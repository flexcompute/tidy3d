from __future__ import annotations

from pathlib import Path
from typing import Any

import tidy3d as td
from tidy3d.web.api import webapi
from tidy3d.web.api.container import Batch, Job

from .io_utils import get_vjp_traced_fields, upload_sim_fields_keys


def parse_run_kwargs(**run_kwargs: Any) -> dict[str, Any]:
    """Parse the ``run_kwargs`` to extract what should be passed to the ``Job``/``Batch`` init."""
    job_fields = [*list(Job._upload_fields.default), "solver_version", "pay_type", "lazy"]
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


def _run_tidy3d(
    simulation: td.Simulation, task_name: str, **run_kwargs: Any
) -> tuple[td.SimulationData, str]:
    """Run a simulation without any tracers using regular web.run()."""

    job_init_kwargs = parse_run_kwargs(**run_kwargs)
    job = Job(simulation=simulation, task_name=task_name, **job_init_kwargs)
    td.log.info(f"running {job.simulation_type} simulation with '_run_tidy3d()'")
    if job.simulation_type == "autograd_fwd":
        verbose = run_kwargs.get("verbose", False)
        upload_sim_fields_keys(run_kwargs["sim_fields_keys"], task_id=job.task_id, verbose=verbose)
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
) -> tuple[td.web.api.container.BatchData, dict[str, str]]:
    """Run a batch of simulations using regular web.run()."""

    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    path_dir = run_kwargs.pop("path_dir", None)
    priority = run_kwargs.get("priority")
    vgpu_allocation = run_kwargs.get("vgpu_allocation")
    ignore_memory_limit = run_kwargs.get("ignore_memory_limit")
    num_workers = run_kwargs.get("num_workers")
    batch = _build_batch(simulations=simulations, num_workers=num_workers, **batch_init_kwargs)
    td.log.info(f"running {batch.simulation_type} batch with '_run_async_tidy3d()'")

    if batch.simulation_type == "autograd_fwd":
        verbose = run_kwargs.get("verbose", False)
        # Need to upload to get the task_ids
        sims = {
            task_name: sim.updated_copy(simulation_type="autograd_fwd", deep=False)
            for task_name, sim in batch.simulations.items()
        }
        batch = batch.updated_copy(simulations=sims)

        batch.upload()
        task_ids = {key: job.task_id for key, job in batch.jobs.items()}
        for task_name, sim_fields_keys in run_kwargs["sim_fields_keys_dict"].items():
            task_id = task_ids[task_name]
            upload_sim_fields_keys(sim_fields_keys, task_id=task_id, verbose=verbose)

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

    task_ids = {key: job.task_id for key, job in batch.jobs.items()}
    return batch_data, task_ids


def _run_async_tidy3d_bwd(
    simulations: dict[str, td.Simulation],
    **run_kwargs: Any,
) -> dict[str, dict]:
    """Run a batch of adjoint simulations using regular web.run()."""

    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    _ = run_kwargs.pop("path_dir", None)
    num_workers = run_kwargs.get("num_workers")
    batch = _build_batch(simulations=simulations, num_workers=num_workers, **batch_init_kwargs)
    td.log.info(f"running {batch.simulation_type} batch with '_run_async_tidy3d_bwd()'")

    priority = run_kwargs.get("priority")
    vgpu_allocation = run_kwargs.get("vgpu_allocation")
    ignore_memory_limit = run_kwargs.get("ignore_memory_limit")
    batch.start(
        priority=priority, vgpu_allocation=vgpu_allocation, ignore_memory_limit=ignore_memory_limit
    )
    batch.monitor()

    vjp_traced_fields_dict = {}
    for task_name, job in batch.jobs.items():
        task_id = job.task_id
        vjp = get_vjp_traced_fields(task_id_adj=task_id, verbose=batch.verbose)
        vjp_traced_fields_dict[task_name] = vjp

    return vjp_traced_fields_dict
