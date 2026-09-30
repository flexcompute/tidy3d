from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import tidy3d as td
from tidy3d.components.autograd.collection import collect_adjoint_sample_sets
from tidy3d.components.autograd.field_map import TracerKeys
from tidy3d.components.simulation import constants as simulation_constants
from tidy3d.components.workflow import Workflow
from tidy3d.exceptions import AdjointError, DataError, WebError
from tidy3d.web.api import webapi
from tidy3d.web.api.container import Batch, Job
from tidy3d.web.api.states import ERROR_STATES

from .constants import SAMPLE_SETS_FILE, SIM_FIELDS_KEYS_FILE
from .io_utils import get_cached_vjp_traced_fields, get_vjp_traced_fields

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tidy3d.components.base import Tidy3dBaseModel
    from tidy3d.em.translate.sample_sets import GeometrySampleSets


# above this estimated payload, the upload size line is logged at info level
_SAMPLE_SETS_SIZE_INFO_BYTES = 50e6


def _validate_sample_sets_size(num_bytes: float, sample_sets: GeometrySampleSets) -> None:
    """Reject an oversized sample-set artifact before upload.

    Client preflight of the backend limit (``MAX_ADJOINT_SAMPLE_SETS_SIZE_GB``, enforced
    authoritatively by the solver before it loads the artifact), with guidance in public
    vocabulary.
    """
    max_bytes = simulation_constants.MAX_ADJOINT_SAMPLE_SETS_SIZE_GB * 1e9
    if num_bytes <= max_bytes:
        return

    top = sorted(sample_sets.entries, key=lambda e: e.sample_sets.num_points, reverse=True)
    largest = ", ".join(
        f"structure {e.structure_index} ({e.sample_sets.num_points:,} points at "
        f"{e.sample_sets.spacing:.3g} um spacing)"
        for e in top[:3]
    )
    # surface point count scales with 1 / spacing**2
    spacing_factor = float(np.sqrt(num_bytes / max_bytes))
    raise AdjointError(
        f"The adjoint surface sample sets for this simulation are {num_bytes / 1e9:.2f} GB, "
        f"above the {max_bytes / 1e9:.1f} GB limit for server-side gradients. Largest "
        f"contributors: {largest}. Shape gradients record fields at every surface sample "
        "point, so the size grows with the traced surface area divided by the square of the "
        f"sampling spacing: coarsen the spacing by at least ~{spacing_factor:.1f}x. The "
        "spacing is 'config.adjoint.default_wavelength_fraction' times the shortest material "
        "length scale, bounded below by 'config.adjoint.minimum_spacing_fraction' times the "
        "shortest adjoint wavelength and by the local grid step. For PEC and other metals the "
        "material length scale is negligible, so raise "
        "'config.adjoint.minimum_spacing_fraction' (or coarsen the grid near the traced "
        "surfaces); for dielectrics raise 'config.adjoint.default_wavelength_fraction'. "
        "Tracing only the surfaces that need gradients (keeping large static parts untraced) "
        "also reduces the size."
    )


def _autograd_forward_sidecar_artifacts(
    simulation: td.Simulation,
    sim_fields_keys: list[tuple],
    sample_sets: GeometrySampleSets | None = None,
) -> dict[str, Tidy3dBaseModel | Path]:
    """Build sidecar artifacts uploaded alongside an autograd forward task.

    ``sample_sets`` is the artifact instance the strategy prepared the forward with
    (``prepare_forward``), so the upload ships the exact object monitors were staged
    from. When absent (direct callers outside the strategies), it is collected here
    without exclusions: sidecars only ride remote-gradient forward uploads, and
    custom vjps / numerical structures (the only sources of exclusions) force
    ``local_gradient=True``.

    The sample-set artifact is serialized here, and that file is both what the size
    limit is checked against and what is uploaded, so the preflight measures exactly
    the bytes the backend receives, before any task is created. The caller owns the
    file and removes it after upload with :func:`_remove_sidecar_files`.
    """
    if sample_sets is None:
        sample_sets = collect_adjoint_sample_sets(simulation, sim_fields_keys)
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    sample_sets_file = Path(fname)
    try:
        sample_sets.to_file(fname)
        num_bytes = sample_sets_file.stat().st_size
        size_message = (
            f"Uploading adjoint sample-set artifact: {sample_sets.num_points} surface points, "
            f"{num_bytes / 1e6:.1f} MB."
        )
        if num_bytes > _SAMPLE_SETS_SIZE_INFO_BYTES:
            td.log.info(size_message)
        else:
            td.log.debug(size_message)
        _validate_sample_sets_size(num_bytes, sample_sets)
    except BaseException:
        sample_sets_file.unlink(missing_ok=True)
        raise
    return {
        SIM_FIELDS_KEYS_FILE: TracerKeys(keys=sim_fields_keys),
        SAMPLE_SETS_FILE: sample_sets_file,
    }


def _remove_sidecar_files(sidecar_artifacts: Mapping[str, Tidy3dBaseModel | Path]) -> None:
    """Delete the sidecar files pre-serialized by :func:`_autograd_forward_sidecar_artifacts`."""
    for artifact in sidecar_artifacts.values():
        if isinstance(artifact, Path):
            artifact.unlink(missing_ok=True)


def parse_run_kwargs(*, include_workflow: bool = False, **run_kwargs: Any) -> dict[str, Any]:
    """Parse the ``run_kwargs`` to extract what should be passed to the ``Job``/``Batch`` init."""
    job_fields = [
        *list(Job._upload_fields.default),
        "solver_version",
        "pay_type",
        "lazy",
    ]
    if include_workflow:
        job_fields.append("workflow")
    job_init_kwargs = {k: v for k, v in run_kwargs.items() if k in job_fields}
    return job_init_kwargs


def _build_batch(
    simulations: dict[str, td.Simulation], *, num_workers: int | None, **kwargs: Any
) -> Batch:
    """Construct ``Batch`` while preserving the model default when ``num_workers`` is omitted."""
    if kwargs.pop("store_preprocess_cache", False):
        raise DataError("Preprocess-cache export is not supported for autograd batches.")
    batch_kwargs = dict(simulations=simulations, **kwargs)
    if num_workers is not None:
        batch_kwargs["num_workers"] = num_workers
    return Batch(**batch_kwargs)


def _with_result_cache_disabled(batch: Batch) -> Batch:
    """Return a copy whose jobs cannot restore generic simulation results from local cache."""
    jobs = {}
    for task_name, job in batch.jobs.items():
        workflow = Workflow(
            steps=tuple(step.updated_copy(cacheable=False, deep=False) for step in job.steps),
        )
        jobs[task_name] = job.updated_copy(workflow=workflow, deep=False)
    return batch.updated_copy(jobs_cached=jobs, deep=False)


def _raise_for_failed_batch_tasks(batch: Batch, *, operation: str) -> None:
    """Raise with task details when a monitored autograd batch contains failures."""
    failed_tasks = []
    for task_name, status in batch._terminal_status_by_task.items():
        if status not in ERROR_STATES:
            continue
        task_id = batch.jobs[task_name].task_id
        try:
            webapi.get_status(task_id)
        except WebError as exc:
            reason = str(exc)
        except Exception:
            reason = "Error details could not be obtained."
        else:
            reason = "The server did not provide an error message."
        failed_tasks.append((task_name, task_id, status, reason))

    if not failed_tasks:
        return

    details = "; ".join(
        f"'{task_name}' (task_id={task_id}, status={status}): {reason}"
        for task_name, task_id, status, reason in failed_tasks
    )
    raise WebError(f"{operation} task(s) failed: {details}")


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs: Any
) -> tuple[td.SimulationData, str]:
    """Run a simulation without any tracers using regular web.run()."""

    job_init_kwargs = parse_run_kwargs(include_workflow=True, **run_kwargs)
    job = Job(simulation=simulation, task_name=task_name, **job_init_kwargs)
    td.log.info(f"running {job.simulation_type} simulation with '_run_tidy3d()'")
    if job.simulation_type == "autograd_fwd":
        sidecar_artifacts = _autograd_forward_sidecar_artifacts(
            simulation,
            run_kwargs["sim_fields_keys"],
            sample_sets=run_kwargs.get("sample_sets"),
        )
        try:
            job._upload_and_cache(verbose_estimate_cost=False, _sidecar_artifacts=sidecar_artifacts)
        finally:
            _remove_sidecar_files(sidecar_artifacts)
    path_arg = run_kwargs.get("path")
    if path_arg is None:
        path = webapi._resolve_output_path(None, job._task_type_hint())
    else:
        path = Path(path_arg)
    priority = run_kwargs.get("priority")
    vgpu_allocation = run_kwargs.get("vgpu_allocation")
    ignore_memory_limit = run_kwargs.get("ignore_memory_limit")
    if task_name.endswith("_adjoint"):
        suffixes = "".join(path.suffixes)
        base_name = path.name
        base_without_suffix = base_name[: -len(suffixes)] if suffixes else base_name
        path = path.with_name(f"{base_without_suffix}_adjoint{suffixes}")
    data = job.run(
        path,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )
    return data, job.task_id


def _run_async_tidy3d(
    simulations: dict[str, td.Simulation], **run_kwargs: Any
) -> tuple[td.web.api.container.BatchData, dict[str, str | None]]:
    """Run a batch of simulations using regular web.run()."""

    disable_result_cache = run_kwargs.pop("disable_result_cache", False)
    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    path_dir = run_kwargs.pop("path_dir", None)
    priority = run_kwargs.get("priority")
    vgpu_allocation = run_kwargs.get("vgpu_allocation")
    ignore_memory_limit = run_kwargs.get("ignore_memory_limit")
    num_workers = run_kwargs.get("num_workers")
    batch = _build_batch(simulations=simulations, num_workers=num_workers, **batch_init_kwargs)
    td.log.info(f"running {batch.simulation_type} batch with '_run_async_tidy3d()'")
    if disable_result_cache:
        batch = _with_result_cache_disabled(batch)

    if batch.simulation_type == "autograd_fwd":
        sims = {
            task_name: sim.updated_copy(simulation_type="autograd_fwd", deep=False)
            for task_name, sim in batch.simulations.items()
        }
        batch = batch.updated_copy(simulations=sims)

        sample_sets_dict = run_kwargs.get("sample_sets_dict") or {}
        sidecar_artifacts_by_task = {}
        try:
            for task_name, sim_fields_keys in run_kwargs["sim_fields_keys_dict"].items():
                sidecar_artifacts_by_task[task_name] = _autograd_forward_sidecar_artifacts(
                    sims[task_name], sim_fields_keys, sample_sets=sample_sets_dict.get(task_name)
                )
            batch._upload_jobs(_sidecar_artifacts_by_task=sidecar_artifacts_by_task)
        finally:
            for sidecar_artifacts in sidecar_artifacts_by_task.values():
                _remove_sidecar_files(sidecar_artifacts)

    if path_dir is not None:
        batch_data = batch.run(
            path_dir,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )
    else:
        batch_data = batch.run(
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
        )

    if run_kwargs.get("is_adjoint", False):
        _raise_for_failed_batch_tasks(batch, operation="Adjoint simulation")

    task_ids = getattr(batch_data, "task_ids", None)
    if task_ids is None:
        task_ids = {key: job.task_id for key, job in batch.jobs.items()}
    else:
        task_ids = dict(task_ids)
    return batch_data, task_ids


def _run_async_tidy3d_bwd(
    simulations: dict[str, td.Simulation],
    **run_kwargs: Any,
) -> dict[str, dict]:
    """Run a batch of adjoint simulations using regular web.run()."""

    verbose = run_kwargs.get("verbose", True)
    vjp_traced_fields_dict = {}
    simulations_to_run = {}
    for task_name, simulation in simulations.items():
        cached = get_cached_vjp_traced_fields(simulation, verbose=verbose)
        if cached is None:
            simulations_to_run[task_name] = simulation
        else:
            vjp_traced_fields_dict[task_name] = cached

    if not simulations_to_run:
        return vjp_traced_fields_dict

    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    _ = run_kwargs.pop("path_dir", None)
    num_workers = run_kwargs.get("num_workers")
    batch = _build_batch(
        simulations=simulations_to_run, num_workers=num_workers, **batch_init_kwargs
    )
    batch = _with_result_cache_disabled(batch)
    td.log.info(f"running {batch.simulation_type} batch with '_run_async_tidy3d_bwd()'")

    priority = run_kwargs.get("priority")
    vgpu_allocation = run_kwargs.get("vgpu_allocation")
    ignore_memory_limit = run_kwargs.get("ignore_memory_limit")
    batch.start(
        priority=priority, vgpu_allocation=vgpu_allocation, ignore_memory_limit=ignore_memory_limit
    )
    batch.monitor()
    _raise_for_failed_batch_tasks(batch, operation="Adjoint simulation")

    for task_name, job in batch.jobs.items():
        task_id = job.task_id
        vjp = get_vjp_traced_fields(
            task_id_adj=task_id,
            verbose=batch.verbose,
            cache_simulation=job.simulation,
        )
        vjp_traced_fields_dict[task_name] = vjp

    return vjp_traced_fields_dict
