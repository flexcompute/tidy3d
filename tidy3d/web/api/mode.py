"""Web API for mode solver"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
from datetime import datetime
from typing import Callable, List, Optional, Union

import pydantic.v1 as pydantic
from botocore.exceptions import ClientError
from joblib import Parallel, delayed
from rich.progress import Progress

from ...components.data.monitor_data import ModeSolverData
from ...components.eme.simulation import EMESimulation
from ...components.medium import AbstractCustomMedium
from ...components.simulation import Simulation
from ...components.types import Literal
from ...exceptions import SetupError, WebError
from ...log import get_logging_console, log
from ...plugins.mode.mode_solver import MODE_MONITOR_NAME, ModeSolver
from ...version import __version__
from ..core.core_config import get_logger_console
from ..core.environment import Env
from ..core.http_util import http
from ..core.s3utils import download_file, download_gz_file, upload_file
from ..core.task_core import Folder
from ..core.types import PayType, ResourceLifecycle, Submittable

SIMULATION_JSON = "simulation.json"
SIM_FILE_HDF5_GZ = "simulation.hdf5.gz"
MODESOLVER_API = "tidy3d/modesolver/py"
MODESOLVER_JSON = "mode_solver.json"
MODESOLVER_HDF5 = "mode_solver.hdf5"
MODESOLVER_GZ = "mode_solver.hdf5.gz"

MODESOLVER_LOG = "output/result.log"
MODESOLVER_RESULT = "output/result.hdf5"
MODESOLVER_RESULT_GZ = "output/mode_solver_data.hdf5.gz"

DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 10  # in seconds


def run(
    mode_solver: ModeSolver,
    task_name: str = "Untitled",
    mode_solver_name: str = "mode_solver",
    folder_name: str = "Mode Solver",
    results_file: str = "mode_solver.hdf5",
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: Union[PayType, str] = PayType.AUTO,
) -> ModeSolverData:
    """Submits a :class:`.ModeSolver` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.ModeSolverData` object.

    Parameters
    ----------
    mode_solver : :class:`.ModeSolver`
        Mode solver to upload to server.
    task_name : str = "Untitled"
        Name of task.
    mode_solver_name: str = "mode_solver"
        The name of the mode solver to create the in task.
    folder_name : str = "Mode Solver"
        Name of folder to store task on web UI.
    results_file : str = "mode_solver.hdf5"
        Path to download results file (.hdf5).
    verbose : bool = True
        If ``True``, will print status, otherwise, will run silently.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    reduce_simulation : Literal["auto", True, False] = "auto"
        Restrict simulation to mode solver region. If "auto", then simulation is automatically
        restricted if it contains custom mediums.
    pay_type: Union[PayType, str] = PayType.AUTO
        Which method to pay the simulation.
    Returns
    -------
    :class:`.ModeSolverData`
        Mode solver data with the calculated results.
    """
    log_level = "DEBUG" if verbose else "INFO"
    if verbose:
        console = get_logging_console()

    if reduce_simulation == "auto":
        sim_mediums = mode_solver.simulation.scene.mediums
        contains_custom = any(isinstance(med, AbstractCustomMedium) for med in sim_mediums)
        reduce_simulation = contains_custom

        if reduce_simulation:
            log.warning(
                "The associated 'Simulation' object contains custom mediums. It will be "
                "automatically restricted to the mode solver plane to reduce data for uploading. "
                "To force uploading the original 'Simulation' object use 'reduce_simulation=False'."
                " Setting 'reduce_simulation=True' will force simulation reduction in all cases and"
                " silence this warning."
            )

    if reduce_simulation:
        mode_solver = mode_solver.reduced_simulation_copy

    task = ModeSolverTask.create(mode_solver, task_name, mode_solver_name, folder_name)
    if verbose:
        console.log(
            f"Mode solver created with task_id='{task.task_id}', solver_id='{task.solver_id}'."
        )
    task.upload(verbose=verbose, progress_callback=progress_callback_upload)
    task.submit(pay_type=pay_type)

    # Wait for task to finish
    prev_status = "draft"
    status = task.status
    while status not in ("success", "error", "diverged", "deleted"):
        if status != prev_status:
            log.log(log_level, f"Mode solver status: {status}")
            if verbose:
                console.log(f"Mode solver status: {status}")
            prev_status = status
        time.sleep(0.5)
        status = task.get_info().status

    if status == "error":
        raise WebError("Error running mode solver.")

    log.log(log_level, f"Mode solver status: {status}")
    if verbose:
        console.log(f"Mode solver status: {status}")

    if status != "success":
        # Our cache discards None, so the user is able to re-run
        return None

    return task.get_result(
        to_file=results_file, verbose=verbose, progress_callback=progress_callback_download
    )


def run_batch(
    mode_solvers: List[ModeSolver],
    task_name: str = "BatchModeSolver",
    folder_name: str = "BatchModeSolvers",
    results_files: List[str] = None,
    verbose: bool = True,
    max_workers: int = DEFAULT_NUM_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
) -> List[ModeSolverData]:
    """
    Submits a batch of ModeSolver to the server concurrently, manages progress, and retrieves results.

    Parameters
    ----------
    mode_solvers : List[ModeSolver]
        List of mode solvers to be submitted to the server.
    task_name : str
        Base name for tasks. Each task in the batch will have a unique index appended to this base name.
    folder_name : str
        Name of the folder where tasks are stored on the server's web UI.
    results_files : List[str], optional
        List of file paths where the results for each ModeSolver should be downloaded. If None, a default path based on the folder name and index is used.
    verbose : bool
        If True, displays a progress bar. If False, runs silently.
    max_workers : int
        Maximum number of concurrent workers to use for processing the batch of simulations.
    max_retries : int
        Maximum number of retries for each simulation in case of failure before giving up.
    retry_delay : int
        Delay in seconds between retries when a simulation fails.
    progress_callback_upload : Callable[[float], None], optional
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None], optional
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.


    Returns
    -------
    List[ModeSolverData]
        A list of ModeSolverData objects containing the results from each simulation in the batch. ``None`` is placed in the list for simulations that fail after all retries.
    """
    console = get_logging_console()

    if not all(isinstance(x, ModeSolver) for x in mode_solvers):
        console.log(
            "Validation Error: All items in `mode_solvers` must be instances of `ModeSolver`."
        )
        return []

    num_mode_solvers = len(mode_solvers)

    if results_files is None:
        results_files = [f"mode_solver_batch_results_{i}.hdf5" for i in range(num_mode_solvers)]

    def handle_mode_solver(index, progress, pbar):
        retries = 0
        while retries <= max_retries:
            try:
                result = run(
                    mode_solver=mode_solvers[index],
                    task_name=f"{task_name}_{index}",
                    mode_solver_name=f"mode_solver_batch_{index}",
                    folder_name=folder_name,
                    results_file=results_files[index],
                    verbose=False,
                    progress_callback_upload=progress_callback_upload,
                    progress_callback_download=progress_callback_download,
                )
                if verbose:
                    progress.update(pbar, advance=1)
                return result
            except Exception as e:
                console.log(f"Error in mode solver {index}: {str(e)}")
                if retries < max_retries:
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    console.log(f"The {index}-th mode solver failed after {max_retries} tries.")
                    if verbose:
                        progress.update(pbar, advance=1)
                    return None

    if verbose:
        console.log(f"[cyan]Running a batch of [deep_pink4]{num_mode_solvers} mode solvers.\n")

        # Create the common folder before running the parallel computation
        _ = Folder.create(folder_name=folder_name)

        with Progress(console=console) as progress:
            pbar = progress.add_task("Status:", total=num_mode_solvers)
            results = Parallel(n_jobs=max_workers, backend="threading")(
                delayed(handle_mode_solver)(i, progress, pbar) for i in range(num_mode_solvers)
            )
            # Make sure the progress bar is complete
            progress.update(pbar, completed=num_mode_solvers, refresh=True)
            console.log("[green]A batch of `ModeSolver` tasks completed successfully!")
    else:
        results = Parallel(n_jobs=max_workers, backend="threading")(
            delayed(handle_mode_solver)(i, None, None) for i in range(num_mode_solvers)
        )

    return results


class ModeSolverTask(ResourceLifecycle, Submittable, extra=pydantic.Extra.allow):
    """Interface for managing the running of a :class:`.ModeSolver` task on server."""

    task_id: str = pydantic.Field(
        None,
        title="task_id",
        description="Task ID number, set when the task is created, leave as None.",
        alias="refId",
    )

    solver_id: str = pydantic.Field(
        None,
        title="solver",
        description="Solver ID number, set when the task is created, leave as None.",
        alias="id",
    )

    real_flex_unit: float = pydantic.Field(
        None, title="real FlexCredits", description="Billed FlexCredits.", alias="charge"
    )

    created_at: Optional[datetime] = pydantic.Field(
        title="created_at", description="Time at which this task was created.", alias="createdAt"
    )

    status: str = pydantic.Field(
        None,
        title="status",
        description="Mode solver task status.",
    )

    file_type: str = pydantic.Field(
        None,
        title="file_type",
        description="File type used to upload the mode solver.",
        alias="fileType",
    )

    mode_solver: ModeSolver = pydantic.Field(
        None,
        title="mode_solver",
        description="Mode solver being run by this task.",
    )

    @classmethod
    def create(
        cls,
        mode_solver: ModeSolver,
        task_name: str = "Untitled",
        mode_solver_name: str = "mode_solver",
        folder_name: str = "Mode Solver",
    ) -> ModeSolverTask:
        """Create a new mode solver task on the server.

        Parameters
        ----------
        mode_solver: :class".ModeSolver"
            The object that will be uploaded to server in the submitting phase.
        task_name: str = "Untitled"
            The name of the task.
        mode_solver_name: str = "mode_solver"
            The name of the mode solver to create the in task.
        folder_name: str = "Mode Solver"
            The name of the folder to store the task.

        Returns
        -------
        :class:`ModeSolverTask`
            :class:`ModeSolverTask` object containing information about the created task.
        """
        folder = Folder.get(folder_name, create=True)

        mode_solver.validate_pre_upload()
        if isinstance(mode_solver.simulation, Simulation):
            mode_solver.simulation.validate_pre_upload(source_required=False)
        elif isinstance(mode_solver.simulation, EMESimulation):
            # TODO: replace this with native web api support
            raise SetupError(
                "'EMESimulation' is not yet supported in the "
                "remote mode solver web api. Please instead call 'ModeSolver.to_fdtd_mode_solver' "
                "before using the web api; this replaces the 'EMESimulation' with a 'Simulation' "
                "that can be used in the remote mode solver. "
                "Alternatively, you can add a 'ModeSolverMonitor' to the 'EMESimulation' "
                "and use the EME solver web api."
            )
            # mode_solver.simulation.validate_pre_upload()
        else:
            raise SetupError("Simulation type not supported in the remote mode solver web api.")

        response_body = {
            "projectId": folder.folder_id,
            "taskName": task_name,
            "protocolVersion": __version__,
            "modeSolverName": mode_solver_name,
            "fileType": "Gz",
            "source": "Python",
        }

        resp = http.post(MODESOLVER_API, response_body)

        # TODO: actually fix the root cause later.
        # sometimes POST for mode returns None and crashes the log.info call.
        # For now this catches it and raises a better error that we can debug.
        if resp is None:
            raise WebError(
                "'ModeSolver' POST request returned 'None'. If you received this error, please "
                "raise an issue on the Tidy3D front end GitHub repository referencing the following"
                f"response body '{response_body}'."
            )

        log.info(
            "Mode solver created with task_id='%s', solver_id='%s'.", resp["refId"], resp["id"]
        )
        return ModeSolverTask(**resp, mode_solver=mode_solver)

    @classmethod
    def get(
        cls,
        task_id: str,
        solver_id: str,
        to_file: str = "mode_solver.hdf5",
        sim_file: str = "simulation.hdf5",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> ModeSolverTask:
        """Get mode solver task from the server by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of the task on server.
        solver_id: str
            Unique identifier of the mode solver in the task.
        to_file: str = "mode_solver.hdf5"
            File to store the mode solver downloaded from the task.
        sim_file: str = "simulation.hdf5"
            File to store the simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`ModeSolverTask`
            :class:`ModeSolverTask` object containing information about the task.
        """
        resp = http.get(f"{MODESOLVER_API}/{task_id}/{solver_id}")
        task = ModeSolverTask(**resp)
        mode_solver = task.get_modesolver(to_file, sim_file, verbose, progress_callback)
        return task.copy(update={"mode_solver": mode_solver})

    def get_info(self) -> ModeSolverTask:
        """Get the current state of this task on the server.

        Returns
        -------
        :class:`ModeSolverTask`
            :class:`ModeSolverTask` object containing information about the task, without the mode
            solver.
        """
        resp = http.get(f"{MODESOLVER_API}/{self.task_id}/{self.solver_id}")
        return ModeSolverTask(**resp, mode_solver=self.mode_solver)

    def upload(
        self, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Upload this task's 'mode_solver' to the server.

        Parameters
        ----------
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while uploading the data.
        """
        mode_solver = self.mode_solver.copy()

        sim = mode_solver.simulation

        # Upload simulation.hdf5.gz for GUI display
        file, file_name = tempfile.mkstemp(".hdf5.gz")
        os.close(file)
        try:
            sim.to_hdf5_gz(file_name)
            upload_file(
                self.task_id,
                file_name,
                SIM_FILE_HDF5_GZ,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)

        # Upload a single HDF5 file with the full data
        file, file_name = tempfile.mkstemp(".hdf5.gz")
        os.close(file)
        try:
            mode_solver.to_hdf5_gz(file_name)
            upload_file(
                self.solver_id,
                file_name,
                MODESOLVER_GZ,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)

    def submit(
        self,
        pay_type: Union[PayType, str] = PayType.AUTO,
    ):
        """Start the execution of this task.

        The mode solver must be uploaded to the server with the :meth:`ModeSolverTask.upload` method
        before this step.
        """
        # convert right before sending to API
        pay_type = PayType(pay_type) if not isinstance(pay_type, PayType) else pay_type

        http.post(
            f"{MODESOLVER_API}/{self.task_id}/{self.solver_id}/run",
            {
                "enableCaching": Env.current.enable_caching,
                "payType": pay_type.value,
            },
        )

    def delete(self):
        """Delete the mode solver and its corresponding task from the server."""
        # Delete mode solver
        http.delete(f"{MODESOLVER_API}/{self.task_id}/{self.solver_id}")
        # Delete parent task
        http.delete(f"tidy3d/tasks/{self.task_id}")

    def abort(self):
        """Abort the mode solver and its corresponding task from the server."""
        return http.put(
            "tidy3d/tasks/abort", json={"taskType": "MODE_SOLVER", "taskId": self.solver_id}
        )

    def get_modesolver(
        self,
        to_file: str = "mode_solver.hdf5",
        sim_file: str = "simulation.hdf5",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> ModeSolver:
        """Get mode solver associated with this task from the server.

        Parameters
        ----------
        to_file: str = "mode_solver.hdf5"
            File to store the mode solver downloaded from the task.
        sim_file: str = "simulation.hdf5"
            File to store the simulation downloaded from the task, if any.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`ModeSolver`
            :class:`ModeSolver` object associated with this task.
        """
        if self.file_type == "Gz":
            file, file_path = tempfile.mkstemp(".hdf5.gz")
            os.close(file)
            try:
                download_file(
                    self.solver_id,
                    MODESOLVER_GZ,
                    to_file=file_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
                mode_solver = ModeSolver.from_hdf5_gz(file_path)
            finally:
                os.unlink(file_path)

        elif self.file_type == "Hdf5":
            file, file_path = tempfile.mkstemp(".hdf5")
            os.close(file)
            try:
                download_file(
                    self.solver_id,
                    MODESOLVER_HDF5,
                    to_file=file_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
                mode_solver = ModeSolver.from_hdf5(file_path)
            finally:
                os.unlink(file_path)

        else:
            file, file_path = tempfile.mkstemp(".json")
            os.close(file)
            try:
                download_file(
                    self.solver_id,
                    MODESOLVER_JSON,
                    to_file=file_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
                mode_solver_dict = ModeSolver.dict_from_json(file_path)
            finally:
                os.unlink(file_path)

            download_file(
                self.task_id,
                SIMULATION_JSON,
                to_file=sim_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
            mode_solver_dict["simulation"] = Simulation.from_json(sim_file)
            mode_solver = ModeSolver.parse_obj(mode_solver_dict)

        # Store requested mode solver file
        mode_solver.to_file(to_file)

        return mode_solver

    def get_result(
        self,
        to_file: str = "mode_solver_data.hdf5",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> ModeSolverData:
        """Get mode solver results for this task from the server.

        Parameters
        ----------
        to_file: str = "mode_solver_data.hdf5"
            File to store the mode solver downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`.ModeSolverData`
            Mode solver data with the calculated results.
        """

        file = None
        try:
            file = download_gz_file(
                resource_id=self.solver_id,
                remote_filename=MODESOLVER_RESULT_GZ,
                to_file=to_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        except ClientError:
            if verbose:
                console = get_logger_console()
                console.log(f"Unable to download '{MODESOLVER_RESULT_GZ}'.")

        if not file:
            try:
                file = download_file(
                    resource_id=self.solver_id,
                    remote_filename=MODESOLVER_RESULT,
                    to_file=to_file,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                raise WebError(
                    "Failed to download the simulation data file from the server. "
                    "Please confirm that the task was successfully run."
                ) from e

        data = ModeSolverData.from_hdf5(to_file)
        data = data.copy(
            update={"monitor": self.mode_solver.to_mode_solver_monitor(name=MODE_MONITOR_NAME)}
        )

        self.mode_solver._cached_properties["data_raw"] = data

        # Perform symmetry expansion
        self.mode_solver._cached_properties.pop("data", None)
        return self.mode_solver.data

    def get_log(
        self,
        to_file: str = "mode_solver.log",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> pathlib.Path:
        """Get execution log for this task from the server.

        Parameters
        ----------
        to_file: str = "mode_solver.log"
            File to store the mode solver downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        return download_file(
            self.solver_id,
            MODESOLVER_LOG,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )
