"""Interface to run several jobs in batch using simplified syntax."""

from __future__ import annotations

import concurrent
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, Optional, Union

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.exceptions import DataError
from tidy3d.log import get_logging_console, log
from tidy3d.web.core.constants import TaskId
from tidy3d.web.core.types import PayType

from .batch_data import DEFAULT_DATA_DIR, BatchData
from .tidy3d_stub import SimulationType

if TYPE_CHECKING:
    pass

# Constants
BATCH_MONITOR_PROGRESS_REFRESH_TIME = 0.02


def upload_async(
    simulations: dict[str, SimulationType],
    folder_name: str,
    callback_url: Optional[str],
    num_workers: int,
    verbose: bool,
    simulation_type: str,
    parent_tasks: Optional[dict[str, list[str]]],
    reduce_simulation: Literal["auto", True, False],
    pay_type: Union[PayType, str],
) -> dict[str, TaskId]:
    """Upload a series of simulations and return task IDs."""
    from . import webapi as web

    # Minimal folder validation
    os.makedirs(folder_name, exist_ok=True)

    task_ids = {}

    def upload_single(task_name, simulation):
        parent_task_ids = parent_tasks.get(task_name, []) if parent_tasks else []
        task_id = web.upload(
            simulation=simulation,
            task_name=task_name,
            folder_name=folder_name,
            callback_url=callback_url,
            verbose=verbose,
            simulation_type=simulation_type,
            parent_tasks=parent_task_ids,
            reduce_simulation=reduce_simulation,
        )
        return task_name, task_id

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(upload_single, task_name, simulation)
            for task_name, simulation in simulations.items()
        ]

        # progressbar (number of tasks uploaded)
        if verbose:
            console = get_logging_console()
            progress_columns = (
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            )
            with Progress(*progress_columns, console=console, transient=False) as progress:
                pbar_message = f"Uploading data for {len(simulations)} tasks"
                pbar = progress.add_task(pbar_message, total=len(simulations))
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    task_name, task_id = future.result()
                    task_ids[task_name] = task_id
                    completed += 1
                    progress.update(pbar, completed=completed)
        else:
            for future in concurrent.futures.as_completed(futures):
                task_name, task_id = future.result()
                task_ids[task_name] = task_id

    return task_ids


def start_async(task_ids: dict[str, TaskId], num_workers: int, verbose: bool) -> None:
    """Start running all tasks.

    Note
    ----
    To monitor the running simulations, can call monitor_async.
    """
    from . import webapi as web

    if verbose:
        console = get_logging_console()
        console.log(f"Started working on {len(task_ids)} tasks.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for task_id in task_ids.values():
            executor.submit(web.start, task_id, verbose=verbose)


def monitor_async(task_ids: dict[str, TaskId], verbose: bool) -> None:
    """Monitor progress of each of the running tasks."""
    from . import webapi as web

    def pbar_description(
        task_name: str, status: str, max_name_length: int, status_width: int
    ) -> str:
        """Make a progressbar description based on the status."""
        # if task name too long, truncate and add ...
        if len(task_name) > max_name_length - 3:  # -3 to leave room for ...
            task_name = task_name[: (max_name_length - 3)] + "..."

        # right-align status
        task_part = f"{task_name:<{max_name_length}}"

        if "error" in status or "diverge" in status or "aborted" in status:
            status_part = f"→ [red]{status:<{status_width}}"
        elif status == "success":
            status_part = f"→ [green]{status:<{status_width}}"
        elif status == "queued" or status == "queued_solver" or status == "aborting":
            status_part = f"→ [yellow]{status:<{status_width}}"
        elif status in ["preprocess", "postprocess", "running"]:
            status_part = f"→ [blue]{status:<{status_width}}"
        else:
            status_part = f"→ {status:<{status_width}}"

        return f"{task_part} {status_part}"

    run_statuses = [
        "draft",
        "queued",
        "preprocess",
        "queued_solver",
        "running",
        "postprocess",
        "visualize",
        "success",
        "aborting",
    ]
    end_statuses = (
        "success",
        "error",
        "errored",
        "diverged",
        "diverge",
        "deleted",
        "draft",
        "aborted",
    )

    max_task_name = max(len(task_name) for task_name in task_ids.keys())
    max_name_length = min(30, max(max_task_name, 15))
    status_width = max(
        max(len(status) for status in run_statuses), max(len(status) for status in end_statuses)
    )

    if verbose:
        console = get_logging_console()

        # Note: Cost estimation not available in async mode
        console.log("Monitoring batch progress. Cost estimation not available in async mode.")

        progress_columns = (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )

        with Progress(*progress_columns, console=console, transient=False) as progress:
            # create progress bars
            pbar_tasks = {}
            task_infos = {}
            for task_name, task_id in task_ids.items():
                task_info = web.get_info(task_id)
                task_infos[task_name] = task_info
                status = task_info.status
                description = pbar_description(task_name, status, max_name_length, status_width)
                completed = run_statuses.index(status) if status in run_statuses else 0
                pbar = progress.add_task(
                    description, total=len(run_statuses) - 1, completed=completed
                )
                pbar_tasks[task_name] = pbar

            while any(task_info.status not in end_statuses for task_info in task_infos.values()):
                updates = []
                for task_name, task_id in task_ids.items():
                    task_info = web.get_info(task_id)
                    task_infos[task_name] = task_info
                    status = task_info.status
                    if status in run_statuses:
                        updates.append(
                            (
                                pbar_tasks[task_name],
                                pbar_description(task_name, status, max_name_length, status_width),
                                run_statuses.index(status),
                            )
                        )

                for pbar, description, completed in updates:
                    progress.update(
                        pbar, description=description, completed=completed, refresh=False
                    )

                progress.refresh()
                time.sleep(BATCH_MONITOR_PROGRESS_REFRESH_TIME)

            updates = []
            for task_name, task_info in task_infos.items():
                updates.append(
                    (
                        pbar_tasks[task_name],
                        pbar_description(
                            task_name, task_info.status, max_name_length, status_width
                        ),
                        len(run_statuses) - 1,
                    )
                )

            for pbar, description, completed in updates:
                progress.update(pbar, description=description, completed=completed, refresh=False)

            progress.refresh()
            console.log("Batch complete.")

    else:
        task_infos = {task_name: web.get_info(task_id) for task_name, task_id in task_ids.items()}
        while any(task_info.status not in end_statuses for task_info in task_infos.values()):
            time.sleep(web.REFRESH_TIME)
            task_infos = {
                task_name: web.get_info(task_id) for task_name, task_id in task_ids.items()
            }


def download_async(
    task_ids: dict[str, TaskId],
    path_dir: str,
    num_workers: int,
    verbose: bool,
    replace_existing: bool = False,
) -> None:
    """Download results of each task."""
    from . import webapi as web

    os.makedirs(path_dir, exist_ok=True)

    def _job_data_path(task_id, path_dir):
        return os.path.join(path_dir, f"{task_id}.hdf5")

    num_existing = 0
    for task_id in task_ids.values():
        job_path_str = _job_data_path(task_id=task_id, path_dir=path_dir)
        if os.path.exists(job_path_str):
            num_existing += 1
    if num_existing > 0:
        files_plural = "files have" if num_existing > 1 else "file has"
        log.warning(
            f"{num_existing} {files_plural} already been downloaded "
            f"and will be skipped. To forcibly overwrite existing files, invoke "
            "the load or download function with `replace_existing=True`.",
            log_once=True,
        )

    def download_single(task_name, task_id):
        job_path_str = _job_data_path(task_id=task_id, path_dir=path_dir)
        if os.path.exists(job_path_str):
            if replace_existing:
                log.info(f"File '{job_path_str}' already exists. Overwriting.")
            else:
                log.info(f"File '{job_path_str}' already exists. Skipping.")
                return None

        task_info = web.get_info(task_id)
        if "error" in task_info.status:
            log.warning(f"Not downloading '{task_name}' as the task errored.")
            return None

        web.download(task_id, path=job_path_str, verbose=verbose)
        return task_name

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(download_single, task_name, task_id)
            for task_name, task_id in task_ids.items()
        ]

        if verbose:
            console = get_logging_console()
            progress_columns = (
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            )
            with Progress(*progress_columns, console=console, transient=False) as progress:
                pbar_message = f"Downloading data for {len(task_ids)} tasks"
                pbar = progress.add_task(pbar_message, total=len(task_ids))
                completed = 0
                for _ in concurrent.futures.as_completed(futures):
                    completed += 1
                    progress.update(pbar, completed=completed)


def load_async(
    task_ids: dict[str, TaskId],
    simulations: dict[str, SimulationType],
    path_dir: str,
    num_workers: int,
    verbose: bool,
    replace_existing: bool = False,
) -> BatchData:
    """Download results and load them into BatchData object."""
    from . import webapi as web

    os.makedirs(path_dir, exist_ok=True)

    def _job_data_path(task_id, path_dir):
        return os.path.join(path_dir, f"{task_id}.hdf5")

    if task_ids is None:
        raise DataError("Can't load batch results, hasn't been uploaded.")

    task_paths = {}
    filtered_task_ids = {}
    for task_name, task_id in task_ids.items():
        task_info = web.get_info(task_id)
        if "error" in task_info.status:
            log.warning(f"Not loading '{task_name}' as the task errored.")
            continue

        task_paths[task_name] = _job_data_path(task_id=task_id, path_dir=path_dir)
        filtered_task_ids[task_name] = task_id

    data = BatchData(task_paths=task_paths, task_ids=filtered_task_ids, verbose=verbose)

    # Handle ModeSolver patching
    for task_name, simulation in simulations.items():
        if task_name in filtered_task_ids and isinstance(simulation, ModeSolver):
            job_data = data[task_name]
            simulation._patch_data(data=job_data)

    download_async(
        task_ids,
        path_dir=path_dir,
        num_workers=num_workers,
        verbose=verbose,
        replace_existing=replace_existing,
    )

    return data


def run_async(
    simulations: dict[str, SimulationType],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: Optional[str] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    parent_tasks: Optional[dict[str, list[str]]] = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: Union[PayType, str] = PayType.AUTO,
) -> BatchData:
    """Submits a set of Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] objects to server,
    starts running, monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    .. TODO add example and see also reference.

    Parameters
    ----------
    simulations : Dict[str, Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]]
        Mapping of task name to simulation.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : str
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type: Union[PayType, str] = PayType.AUTO
        Specify the payment method.

    Returns
    ------
    :class:`BatchData`
        Contains the Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

    See Also
    --------

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    :class:`Batch`
        Interface for submitting several :class:`Simulation` objects to sever.
    """
    if simulation_type is None:
        simulation_type = "tidy3d"

    # if number of workers not specified, just use the number of simulations
    if num_workers is not None:
        log.warning(
            "The 'num_workers' kwarg does not have an effect anymore as all "
            "simulations will now be uploaded in a single batch."
        )

    # Use the new async functions with raw arguments
    task_ids = upload_async(
        simulations=simulations,
        folder_name=folder_name,
        callback_url=callback_url,
        num_workers=num_workers or len(simulations),
        verbose=verbose,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
    )

    start_async(task_ids, num_workers or len(simulations), verbose)

    monitor_async(task_ids, verbose)

    return load_async(
        task_ids=task_ids,
        simulations=simulations,
        path_dir=path_dir,
        num_workers=num_workers or len(simulations),
        verbose=verbose,
        replace_existing=False,
    )
