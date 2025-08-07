"""Unified orchestration helpers for running single or batched simulations.

These helpers centralize the upload → start → monitor → load flow so that both
autograd wrappers and high-level containers can reuse the same execution path,
with optional hooks for custom steps (e.g., uploading sim_fields_keys).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional

from tidy3d.web.api import webapi as web
from tidy3d.web.core.types import PayType

if TYPE_CHECKING:  # only for type checkers; avoid import-time cycles
    from tidy3d.web.api.container import BatchData


def run_job(
    *,
    simulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: Optional[str] = None,
    verbose: bool = True,
    progress_callback_upload: Optional[Callable[[float], None]] = None,
    progress_callback_download: Optional[Callable[[float], None]] = None,
    solver_version: Optional[str] = None,
    worker_group: Optional[str] = None,
    simulation_type: str = "tidy3d",
    parent_tasks: Optional[list[str]] = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str = PayType.AUTO,
    priority: Optional[int] = None,
    post_upload: Optional[Callable[[str], None]] = None,
):
    """Run a single simulation using the canonical web API flow with an optional post-upload hook.

    Returns a tuple of (SimulationDataType, task_id).
    """

    task_id = web.upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback=progress_callback_upload,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        solver_version=solver_version,
        reduce_simulation=reduce_simulation,
    )

    if post_upload is not None:
        post_upload(task_id)

    web.start(
        task_id,
        solver_version=solver_version,
        worker_group=worker_group,
        pay_type=pay_type,
        priority=priority,
    )

    web.monitor(task_id, verbose=verbose)

    # Ensure parent directory exists to match legacy Job semantics
    try:
        from tidy3d.web.api.container import Job

        Job._check_path_dir(path)
    except Exception:
        pass

    data = web.load(
        task_id=task_id,
        path=path,
        verbose=verbose,
        progress_callback=progress_callback_download,
    )

    return data, task_id


def run_batch(
    *,
    simulations: dict[str, object],
    folder_name: str = "default",
    path_dir: str = ".",
    callback_url: Optional[str] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    parent_tasks: Optional[dict[str, list[str]]] = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str = PayType.AUTO,
    solver_version: Optional[str] = None,
    post_upload: Optional[Callable[[dict[str, str]], None]] = None,
) -> tuple[BatchData, dict[str, str]]:
    """Run a batch of simulations using the canonical Batch flow with optional post-upload hook.

    Returns a tuple of (BatchData, {task_name: task_id}).
    """

    # Local import to avoid circular import at module import time
    from tidy3d.web.api.container import Batch

    batch = Batch(
        simulations=simulations,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        solver_version=solver_version,
        num_workers=num_workers,
    )

    batch.upload()
    task_ids = {key: job.task_id for key, job in batch.jobs.items()}

    if post_upload is not None:
        post_upload(task_ids)

    batch.to_file(batch._batch_path(path_dir=path_dir))
    batch.start()
    batch.monitor()
    batch_data = batch.load(path_dir=path_dir)
    return batch_data, task_ids
