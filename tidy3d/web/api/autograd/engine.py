from __future__ import annotations

from pathlib import Path
from typing import Any

import tidy3d as td
from tidy3d.web.api.container import DEFAULT_DATA_PATH, Batch, Job

from .io_utils import get_vjp_traced_fields, upload_sim_fields_keys


def parse_run_kwargs(**run_kwargs: Any) -> dict[str, Any]:
    """Parse the ``run_kwargs`` to extract what should be passed to the ``Job``/``Batch`` init."""
    job_fields = [*list(Job._upload_fields), "solver_version", "pay_type", "lazy"]
    job_init_kwargs = {k: v for k, v in run_kwargs.items() if k in job_fields}
    return job_init_kwargs


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
    path = Path(run_kwargs.get("path", DEFAULT_DATA_PATH))
    priority = run_kwargs.get("priority")
    if task_name.endswith("_adjoint"):
        suffixes = "".join(path.suffixes)
        base_name = path.name
        base_without_suffix = base_name[: -len(suffixes)] if suffixes else base_name
        path = path.with_name(f"{base_without_suffix}_adjoint{suffixes}")
    data = job.run(path, priority=priority)
    return data, job.task_id


def _run_async_tidy3d(
    simulations: dict[str, td.Simulation], **run_kwargs: Any
) -> tuple[td.web.api.container.BatchData, dict[str, str]]:
    """Run a batch of simulations using regular web.run()."""

    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    path_dir = run_kwargs.pop("path_dir", None)
    priority = run_kwargs.get("priority")
    batch = Batch(simulations=simulations, **batch_init_kwargs)
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
        batch_data = batch.run(path_dir, priority=priority)
    else:
        batch_data = batch.run(priority=priority)

    task_ids = {key: job.task_id for key, job in batch.jobs.items()}
    return batch_data, task_ids


def _run_async_tidy3d_bwd(
    simulations: dict[str, td.Simulation],
    **run_kwargs: Any,
) -> dict[str, dict]:
    """Run a batch of adjoint simulations using regular web.run()."""

    batch_init_kwargs = parse_run_kwargs(**run_kwargs)
    _ = run_kwargs.pop("path_dir", None)
    batch = Batch(simulations=simulations, **batch_init_kwargs)
    td.log.info(f"running {batch.simulation_type} batch with '_run_async_tidy3d_bwd()'")

    priority = run_kwargs.get("priority")
    batch.start(priority=priority)
    batch.monitor()

    vjp_traced_fields_dict = {}
    for task_name, job in batch.jobs.items():
        task_id = job.task_id
        vjp = get_vjp_traced_fields(task_id_adj=task_id, verbose=batch.verbose)
        vjp_traced_fields_dict[task_name] = vjp

    return vjp_traced_fields_dict
