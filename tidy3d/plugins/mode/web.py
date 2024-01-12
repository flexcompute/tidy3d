"""Web API for mode solver"""

from __future__ import annotations
from typing import Optional, Callable

from datetime import datetime
import os
import pathlib
import tempfile
import time

import pydantic as pydantic

from ...components.simulation import Simulation
from ...components.data.monitor_data import ModeSolverData
from ...exceptions import WebError
from ...log import log, get_logging_console
from ...web.file_util import compress_file_to_gzip, extract_gz_file
from ...web.http_management import http
from ...web.s3utils import download_file, upload_file
from ...web.simulation_task import Folder, SIMULATION_JSON, SIM_FILE_HDF5_GZ
from ...web.types import ResourceLifecycle, Submittable

from .mode_solver import ModeSolver, MODE_MONITOR_NAME
from ...version import __version__

MODESOLVER_API = "tidy3d/modesolver/py"
MODESOLVER_JSON = "mode_solver.json"
MODESOLVER_HDF5 = "mode_solver.hdf5"
MODESOLVER_GZ = "mode_solver.hdf5.gz"

MODESOLVER_LOG = "output/result.log"
MODESOLVER_RESULT = "output/result.hdf5"


def run(
    mode_solver: ModeSolver,
    task_name: str = "Untitled",
    mode_solver_name: str = "mode_solver",
    folder_name: str = "Mode Solver",
    results_file: str = "mode_solver.hdf5",
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
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
        If `True`, will print status, otherwise, will run silently.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Returns
    -------
    :class:`.ModeSolverData`
        Mode solver data with the calculated results.
    """

    log_level = "DEBUG" if verbose else "INFO"
    if verbose:
        console = get_logging_console()

    task = ModeSolverTask.create(mode_solver, task_name, mode_solver_name, folder_name)
    if verbose:
        console.log(
            f"Mode solver created with task_id='{task.task_id}', solver_id='{task.solver_id}'."
        )
    task.upload(verbose=verbose, progress_callback=progress_callback_upload)
    task.submit()

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

        mode_solver.simulation.validate_pre_upload(source_required=False)
        resp = http.post(
            MODESOLVER_API,
            {
                "projectId": folder.folder_id,
                "taskName": task_name,
                "protocolVersion": __version__,
                "modeSolverName": mode_solver_name,
                "fileType": "Gz",
                "source": "Python",
            },
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

        file, file_name = tempfile.mkstemp()
        gz_file, gz_file_name = tempfile.mkstemp()
        os.close(file)
        os.close(gz_file)

        sim.to_hdf5(file_name)
        try:
            # Upload simulation.hdf5.gz for GUI display
            # compress .hdf5 to .hdf5.gz
            compress_file_to_gzip(file_name, gz_file_name)

            upload_file(
                self.task_id,
                gz_file_name,
                SIM_FILE_HDF5_GZ,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)
            os.unlink(gz_file_name)

        # Upload a single HDF5 file with the full data
        file, file_name = tempfile.mkstemp()
        gz_file, gz_file_name = tempfile.mkstemp()
        os.close(file)
        os.close(gz_file)
        mode_solver.to_hdf5(file_name)

        try:
            # compress .hdf5 to .hdf5.gz
            compress_file_to_gzip(file_name, gz_file_name)

            upload_file(
                self.solver_id,
                gz_file_name,
                MODESOLVER_GZ,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)
            os.unlink(gz_file_name)

    def submit(self):
        """Start the execution of this task.

        The mode solver must be uploaded to the server with the :meth:`ModeSolverTask.upload` method
        before this step.
        """
        http.post(f"{MODESOLVER_API}/{self.task_id}/{self.solver_id}/run")

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
            File to store the simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`ModeSolver`
            :class:`ModeSolver` object associated with this task.

        Note
        ----
        If the simulation contains custom datasets (such as custom media), an HDF5 file will be
        stored in the same path as 'to_file', but with '.hdf5' extension, and neither 'to_file' or
        'sim_file' will be created.
        """
        if self.file_type == "Gz":
            hdf5_gz_file, hdf5_gz_file_path = tempfile.mkstemp()
            os.close(hdf5_gz_file)
            # keep hdf5_file_path
            hdf5_file, hdf5_file_path = tempfile.mkstemp()
            os.close(hdf5_file)
            try:
                download_file(
                    self.solver_id,
                    MODESOLVER_GZ,
                    to_file=hdf5_gz_file_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
                extract_gz_file(hdf5_gz_file_path, hdf5_file_path)
                mode_solver = ModeSolver.from_hdf5(hdf5_file_path)
                if to_file.endswith(".json"):
                    mode_solver.to_json(to_file)

            finally:
                os.unlink(hdf5_gz_file_path)
                os.unlink(hdf5_file_path)

        elif self.file_type == "Hdf5":
            hdf5_file, hdf5_file_path = tempfile.mkstemp()
            os.close(hdf5_file)
            try:
                download_file(
                    self.solver_id,
                    MODESOLVER_HDF5,
                    to_file=hdf5_file_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
                mode_solver = ModeSolver.from_hdf5(hdf5_file_path)
                if to_file.endswith(".json"):
                    mode_solver.to_json(to_file)

            finally:
                os.unlink(hdf5_file_path)

        else:
            download_file(
                self.solver_id,
                MODESOLVER_JSON,
                to_file=to_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
            download_file(
                self.task_id,
                SIMULATION_JSON,
                to_file=sim_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
            mode_solver_dict = ModeSolver.dict_from_json(to_file)
            mode_solver_dict["simulation"] = Simulation.from_json(sim_file)

            mode_solver = ModeSolver.parse_obj(mode_solver_dict)

        # Overwrite downloaded file with valid contents
        mode_solver.to_file(to_file)

        return mode_solver

    def get_result(
        self,
        to_file: str = "mode_solver.hdf5",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> ModeSolverData:
        """Get mode solver results for this task from the server.

        Parameters
        ----------
        to_file: str = "mode_solver.hdf5"
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
        download_file(
            self.solver_id,
            MODESOLVER_RESULT,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        data = ModeSolverData.from_hdf5(to_file)
        data = data.copy(
            update={"monitor": self.mode_solver.to_mode_solver_monitor(name=MODE_MONITOR_NAME)}
        )

        self.mode_solver._cached_properties["data_raw"] = data

        # Perform symmetry expansion
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
