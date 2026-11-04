from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from flexcompute.core._migration.em.derivatives import (
    BoundaryShapeKernel,
    CollarKernel,
    write_program,
)

import tidy3d as td
from tidy3d.components.autograd.field_map import TracerKeys
from tidy3d.components.simulation import constants as simulation_constants
from tidy3d.components.workflow import Workflow
from tidy3d.exceptions import AdjointError, DataError, WebError
from tidy3d.web.api import webapi
from tidy3d.web.api.container import Batch, Job
from tidy3d.web.api.states import ERROR_STATES

from .constants import SIM_DERIVATIVE_PROGRAM_FILE, SIM_FIELDS_KEYS_FILE
from .io_utils import get_cached_derivative_results, get_derivative_results

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flexcompute.core._migration.em.derivatives import DerivativeProgram

    from tidy3d.components.base import Tidy3dBaseModel


# above this payload, the upload size line is logged at info level
_PROGRAM_SIZE_INFO_BYTES = 50e6


def _validate_program_size(num_bytes: float, program: DerivativeProgram) -> None:
    """Reject an oversized derivative program before upload.

    Client preflight of the backend limit (``MAX_ADJOINT_SAMPLE_SETS_SIZE_GB``), with guidance
    in public vocabulary. The program carries every surface sample point, and for a PEC
    surface the collar cells around it, which dominate its size.
    """
    max_bytes = simulation_constants.MAX_ADJOINT_SAMPLE_SETS_SIZE_GB * 1e9
    if num_bytes <= max_bytes:
        return

    # per structure: surface samples, and collar cells (with their samples) for PEC surfaces
    samples: dict[int, int] = {}
    cells: dict[int, int] = {}
    for key, kernel in program.terms:
        index = key[1]
        if isinstance(kernel, BoundaryShapeKernel):
            samples[index] = samples.get(index, 0) + sum(len(s.normals) for s in kernel.sample_sets)
        elif isinstance(kernel, CollarKernel):
            samples[index] = samples.get(index, 0) + sum(len(s.points) for s in kernel.sample_sets)
            cells[index] = cells.get(index, 0) + len(kernel.cell_centers)
    totals = sorted(
        ((samples.get(i, 0) + cells.get(i, 0), i) for i in samples.keys() | cells.keys()),
        reverse=True,
    )
    largest = ", ".join(
        f"structure {i} ({samples.get(i, 0):,} surface samples"
        + (f", {cells[i]:,} collar cells)" if i in cells else ")")
        for _, i in totals[:3]
    )
    # point counts scale with 1 / spacing**2 (or 1 / grid step**2 for metals)
    factor = float(np.sqrt(num_bytes / max_bytes))
    advice = []
    if cells:
        advice.append(
            "PEC and other metal surfaces are sampled at the local grid step and wrapped in a "
            "collar of grid cells a few cells deep, so their size grows with the traced metal "
            "surface area divided by the square of the grid step: coarsen the grid near the "
            f"traced metal by at least ~{factor:.1f}x, or trace fewer metal structures."
        )
    if any(index not in cells for index in samples):
        advice.append(
            "Dielectric surfaces are sampled every 'config.adjoint.default_wavelength_fraction' "
            "times the shortest material length scale (bounded below by "
            "'config.adjoint.minimum_spacing_fraction' times the shortest adjoint wavelength "
            "and by the local grid step), so their size grows with the traced surface area "
            f"divided by the square of that spacing: coarsen it by at least ~{factor:.1f}x by "
            "raising 'config.adjoint.default_wavelength_fraction'."
        )
    raise AdjointError(
        f"The derivative program for this simulation is {num_bytes / 1e9:.2f} GB, above the "
        f"{max_bytes / 1e9:.1f} GB limit for server-side gradients. Largest contributors: "
        f"{largest}. "
        + " ".join(advice)
        + " Tracing only the surfaces that need gradients (keeping large static parts "
        "untraced) also reduces the size."
    )


def _autograd_forward_sidecar_artifacts(
    program: DerivativeProgram,
) -> dict[str, Tidy3dBaseModel | Path]:
    """Build sidecar artifacts uploaded alongside an autograd forward task.

    The program is serialized here, and that file is both what the size limit is checked
    against and what is uploaded, so the preflight measures exactly the bytes the backend
    receives, before any task is created. The caller owns the file and removes it after upload
    with :func:`_remove_sidecar_files`.
    """
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    program_file = Path(fname)
    try:
        write_program(program, program_file)
        num_bytes = program_file.stat().st_size
        size_message = f"Uploading the derivative program: {num_bytes / 1e6:.1f} MB."
        if num_bytes > _PROGRAM_SIZE_INFO_BYTES:
            td.log.info(size_message)
        else:
            td.log.debug(size_message)
        _validate_program_size(num_bytes, program)
    except BaseException:
        program_file.unlink(missing_ok=True)
        raise
    return {SIM_FIELDS_KEYS_FILE: TracerKeys(keys=()), SIM_DERIVATIVE_PROGRAM_FILE: program_file}


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
        sidecar_artifacts = _autograd_forward_sidecar_artifacts(run_kwargs["derivative_program"])
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

        sidecar_artifacts_by_task = {}
        try:
            for task_name, program in run_kwargs["derivative_programs_dict"].items():
                sidecar_artifacts_by_task[task_name] = _autograd_forward_sidecar_artifacts(program)
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
    """Run a batch of adjoint simulations and return each one's derivative program results."""

    verbose = run_kwargs.get("verbose", True)
    results_dict = {}
    simulations_to_run = {}
    for task_name, simulation in simulations.items():
        cached = get_cached_derivative_results(simulation, verbose=verbose)
        if cached is None:
            simulations_to_run[task_name] = simulation
        else:
            results_dict[task_name] = cached

    if not simulations_to_run:
        return results_dict

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
        results_dict[task_name] = get_derivative_results(
            task_id_adj=job.task_id,
            verbose=batch.verbose,
            cache_simulation=job.simulation,
        )

    return results_dict
