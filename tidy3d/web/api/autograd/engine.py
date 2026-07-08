from __future__ import annotations

from pathlib import Path
from typing import Any

import tidy3d as td
from tidy3d.components.autograd.field_map import TracerKeys
from tidy3d.components.workflow import Workflow
from tidy3d.web.api import webapi
from tidy3d.web.api.container import Batch, Job

from .constants import SIM_FIELDS_KEYS_FILE
from .io_utils import get_cached_vjp_traced_fields, get_vjp_traced_fields


def _sim_fields_keys_artifacts(sim_fields_keys: list[tuple]) -> dict[str, TracerKeys]:
    """Build sidecar artifacts needed before autograd forward metadata processing."""
    return {SIM_FIELDS_KEYS_FILE: TracerKeys(keys=sim_fields_keys)}


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


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs: Any
) -> tuple[td.SimulationData, str]:
    """Run a simulation without any tracers using regular web.run()."""

    job_init_kwargs = parse_run_kwargs(include_workflow=True, **run_kwargs)
    job = Job(simulation=simulation, task_name=task_name, **job_init_kwargs)
    td.log.info(f"running {job.simulation_type} simulation with '_run_tidy3d()'")
    if job.simulation_type == "autograd_fwd":
        job._upload_and_cache(
            verbose_estimate_cost=False,
            _sidecar_artifacts=_sim_fields_keys_artifacts(run_kwargs["sim_fields_keys"]),
        )
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

        sim_fields_key_artifacts = {
            task_name: _sim_fields_keys_artifacts(sim_fields_keys)
            for task_name, sim_fields_keys in run_kwargs["sim_fields_keys_dict"].items()
        }
        batch._upload_jobs(_sidecar_artifacts_by_task=sim_fields_key_artifacts)

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

    for task_name, job in batch.jobs.items():
        task_id = job.task_id
        vjp = get_vjp_traced_fields(
            task_id_adj=task_id,
            verbose=batch.verbose,
            cache_simulation=job.simulation,
        )
        vjp_traced_fields_dict[task_name] = vjp

    return vjp_traced_fields_dict
