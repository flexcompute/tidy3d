"""Tidy3d webapi types."""
from __future__ import annotations

import os
import pathlib
import tempfile
from datetime import datetime
from typing import List, Optional, Callable, Tuple
import pydantic as pd
from pydantic import Extra, Field, parse_obj_as

from tidy3d import Simulation
from tidy3d.version import __version__
from tidy3d.exceptions import WebError

from .cache import FOLDER_CACHE
from .http_management import http
from .s3utils import download_file, upload_file
from .types import Queryable, ResourceLifecycle, Submittable
from .types import Tidy3DResource
from .file_util import compress_file_to_gzip, extract_gz_file, read_simulation_from_hdf5

from ..log import get_logging_console

SIMULATION_JSON = "simulation.json"
SIMULATION_DATA_HDF5 = "output/monitor_data.hdf5"
RUNNING_INFO = "output/solver_progress.csv"
SIM_LOG_FILE = "output/tidy3d.log"
SIM_FILE_HDF5 = "simulation.hdf5"
SIM_FILE_HDF5_GZ = "simulation.hdf5.gz"


class Folder(Tidy3DResource, Queryable, extra=Extra.allow):
    """Tidy3D Folder."""

    folder_id: str = Field(..., title="Folder id", description="folder id", alias="projectId")
    folder_name: str = Field(
        ..., title="Folder name", description="folder name", alias="projectName"
    )

    @classmethod
    def list(cls) -> []:
        """List all folders.

        Returns
        -------
        folders : [Folder]
            List of folders
        """
        resp = http.get("tidy3d/projects")
        return (
            parse_obj_as(
                List[Folder],
                resp,
            )
            if resp
            else None
        )

    @classmethod
    def get(cls, folder_name: str, create: bool = False):
        """Get folder by name.

        Parameters
        ----------
        folder_name : str
            Name of the folder.
        create : str
            If the folder doesn't exist, create it.

        Returns
        -------
        folder : Folder
        """
        folder = FOLDER_CACHE.get(folder_name)
        if not folder:
            resp = http.get(f"tidy3d/project?projectName={folder_name}")
            if resp:
                folder = Folder(**resp)
        if create and not folder:
            resp = http.post("tidy3d/projects", {"projectName": folder_name})
            if resp:
                folder = Folder(**resp)
        FOLDER_CACHE[folder_name] = folder
        return folder

    @classmethod
    def create(cls, folder_name: str):
        """Create a folder, return existing folder if there is one has the same name.

        Parameters
        ----------
        folder_name : str
            Name of the folder.

        Returns
        -------
        folder : Folder
        """
        return Folder.get(folder_name, True)

    def delete(self):
        """Remove this folder."""

        http.delete(f"tidy3d/projects/{self.folder_id}")

    def list_tasks(self) -> List[Tidy3DResource]:
        """List all tasks in this folder.

        Returns
        -------
        tasks : List[:class:`.SimulationTask`]
            List of tasks in this folder
        """
        resp = http.get(f"tidy3d/projects/{self.folder_id}/tasks")
        return (
            parse_obj_as(
                List[SimulationTask],
                resp,
            )
            if resp
            else None
        )


class SimulationTask(ResourceLifecycle, Submittable, extra=Extra.allow):
    """Interface for managing the running of a :class:`.Simulation` task on server."""

    task_id: Optional[str] = Field(
        ...,
        title="task_id",
        description="Task ID number, set when the task is uploaded, leave as None.",
        alias="taskId",
    )
    status: Optional[str] = Field(title="status", description="Simulation task status.")

    real_flex_unit: float = Field(
        None, title="real FlexCredits", description="Billed FlexCredits.", alias="realCost"
    )

    created_at: Optional[datetime] = Field(
        title="created_at", description="Time at which this task was created.", alias="createdAt"
    )

    simulation: Optional[Simulation] = Field(
        title="simulation", description="A copy of the Simulation being run as this task."
    )

    folder_name: Optional[str] = Field(
        "default",
        title="Folder Name",
        description="Name of the folder associated with this task.",
        alias="projectName",
    )

    folder: Optional[Folder]

    callback_url: str = Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    # simulation_type: str = pd.Field(
    #     None,
    #     title="Simulation Type",
    #     description="Type of simulation, used internally only.",
    # )

    # parent_tasks: Tuple[TaskId, ...] = pd.Field(
    #     None,
    #     title="Parent Tasks",
    #     description="List of parent task ids for the simulation, used internally only."
    # )

    @pd.root_validator(pre=True)
    def _error_if_jax_sim(cls, values):
        """Raise error if user tries to submit simulation that's a JaxSimulation."""
        sim = values.get("simulation")
        if sim is None:
            return values
        if "JaxSimulation" in str(type(sim)):
            raise ValueError(
                "'JaxSimulation' not compatible with regular webapi functions. "
                "Either convert it to Simulation with 'jax_sim.to_simulation()[0]' or use "
                "the 'adjoint.run' function to run JaxSimulations."
            )
        return values

    @classmethod
    def create(
        cls,
        simulation: Simulation,
        task_name: str,
        folder_name: str = "default",
        callback_url: str = None,
        simulation_type: str = "tidy3d",
        parent_tasks: List[str] = None,
        file_type: str = "Gz",
    ) -> SimulationTask:
        """Create a new task on the server.

        Parameters
        ----------
        simulation: :class".Simulation"
            The :class:`.Simulation` will be uploaded to server in the submitting phase.
            If Simulation is too large to fit into memory, pass None to this parameter
            and use :meth:`.SimulationTask.upload_file` instead.
        task_name: str
            The name of the task.
        folder_name: str,
            The name of the folder to store the task. Default is "default".
        callback_url: str
            Http PUT url to receive simulation finish event. The body content is a json file with
            fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
        simulation_type : str
            Type of simulation being uploaded.
        parent_tasks : List[str]
            List of related task ids.
        file_type: str
            the simulation file type Json, Hdf5, Gz

        Returns
        -------
        :class:`SimulationTask`
            :class:`SimulationTask` object containing info about status, size,
            credits of task and others.
        """
        folder = Folder.get(folder_name, create=True)
        resp = http.post(
            f"tidy3d/projects/{folder.folder_id}/tasks",
            {
                "taskName": task_name,
                "callbackUrl": callback_url,
                "simulationType": simulation_type,
                "parentTasks": parent_tasks,
                "fileType": file_type,
            },
        )

        return SimulationTask(**resp, simulation=simulation, folder=folder)

    @classmethod
    def get(cls, task_id: str) -> SimulationTask:
        """Get task from the server by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of task on server.

        Returns
        -------
        :class:`.SimulationTask`
            :class:`.SimulationTask` object containing info about status,
             size, credits of task and others.
        """
        resp = http.get(f"tidy3d/tasks/{task_id}/detail")
        return SimulationTask(**resp) if resp else None

    @classmethod
    def get_running_tasks(cls) -> List[SimulationTask]:
        """Get a list of running tasks from the server"

        Returns
        -------
        List[:class:`.SimulationTask`]
            :class:`.SimulationTask` object containing info about status,
             size, credits of task and others.
        """
        resp = http.get("tidy3d/py/tasks")
        if not resp:
            return []
        return parse_obj_as(List[SimulationTask], resp)

    def delete(self):
        """Delete current task from server."""
        if not self.task_id:
            raise ValueError("Task id not found.")
        http.delete(f"tidy3d/tasks/{self.task_id}")

    def get_simulation(self) -> Optional[Simulation]:
        """Download simulation from server.

        Returns
        -------
        :class:`.Simulation`
            :class:`.Simulation` object containing info about status, size,
            credits of task and others.
        """
        if self.simulation:
            return self.simulation

        with tempfile.NamedTemporaryFile(suffix=".json") as temp:
            self.get_simulation_json(temp.name)
            if pathlib.Path(temp.name).exists():
                self.simulation = Simulation.from_file(temp.name)
                return self.simulation
        return None

    def get_simulation_json(self, to_file: str, verbose: bool = True) -> pathlib.Path:
        """Get json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progress bars.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        assert self.task_id
        hdf5_file, hdf5_file_path = tempfile.mkstemp()
        os.close(hdf5_file)
        try:
            self.get_simulation_hdf5(hdf5_file_path)
            if os.path.exists(hdf5_file_path):
                json_string = read_simulation_from_hdf5(hdf5_file_path)
                with open(to_file, "w") as file:
                    # Write the string to the file
                    file.write(json_string.decode("utf-8"))
                    if verbose:
                        console = get_logging_console()
                        console.log(f"Generate {to_file} successfully.")
            else:
                raise WebError("Failed to download simulation.json.")
        finally:
            os.unlink(hdf5_file_path)

    def upload_simulation(
        self, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Upload :class:`.Simulation` object to Server.

        Parameters
        ----------
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while uploading the data.
        """
        assert self.task_id
        assert self.simulation

        # Also upload hdf5.gz containing all data.
        file, file_name = tempfile.mkstemp()
        gz_file, gz_file_name = tempfile.mkstemp()
        os.close(file)
        os.close(gz_file)
        self.simulation.to_hdf5(file_name)
        try:
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

    def upload_file(
        self,
        local_file: str,
        remote_filename: str,
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> None:
        """
        Upload file to platform. Using this method when the json file is too large to parse
         as :class".simulation".
        Parameters
        ----------
        local_file: str
            local file path.
        remote_filename: str
            file name on the server
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while uploading the data.
        """
        assert self.task_id

        upload_file(
            self.task_id,
            local_file,
            remote_filename,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def submit(
        self,
        solver_version: str = None,
        worker_group: str = None,
    ):
        """Kick off this task.

        If this task instance contain a :class:`.Simulation`, it will be uploaded to server before
        starting the task. Otherwise, this method assumes that the Simulation has been uploaded by
        the upload_file function, so the task will be kicked off directly.

        Parameters
        ----------
        solver_version: str = None
            target solver version.
        worker_group: str = None
            worker group
        """
        if self.simulation:
            # Also upload hdf5.gz containing all data.
            file, file_name = tempfile.mkstemp()
            gz_file, gz_file_name = tempfile.mkstemp()
            os.close(file)
            os.close(gz_file)
            self.simulation.to_hdf5(file_name)
            try:
                # compress .hdf5 to .hdf5.gz
                compress_file_to_gzip(file_name, gz_file_name)

                upload_file(
                    self.task_id,
                    gz_file_name,
                    SIM_FILE_HDF5_GZ,
                    verbose=False,
                    progress_callback=None,
                )
            finally:
                os.unlink(file_name)
                os.unlink(gz_file_name)

        if solver_version:
            protocol_version = None
        else:
            protocol_version = __version__

        http.post(
            f"tidy3d/tasks/{self.task_id}/submit",
            {
                "solverVersion": solver_version,
                "workerGroup": worker_group,
                "protocolVersion": protocol_version,
            },
        )

    def estimate_cost(self, solver_version=None) -> float:
        """Compute the maximum flex unit charge for a given task, assuming the simulation runs for
        the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.

        Parameters
        ----------
        solver_version: str
            target solver version.

        Returns
        -------
        flex_unit_cost: float
            estimated cost in FlexCredits
        """

        if solver_version:
            protocol_version = None
        else:
            protocol_version = __version__

        assert self.task_id
        resp = http.post(
            f"tidy3d/tasks/{self.task_id}/metadata",
            {
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
            },
        )
        return resp

    def get_sim_data_hdf5(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> pathlib.Path:
        """Get output/monitor_data.hdf5 file from Server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        assert self.task_id
        return download_file(
            self.task_id,
            SIMULATION_DATA_HDF5,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_simulation_hdf5(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> pathlib.Path:
        """Get simulation.hdf5 file from Server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        assert self.task_id
        hdf5_gz_file, hdf5_gz_file_path = tempfile.mkstemp()
        os.close(hdf5_gz_file)
        try:
            download_file(
                self.task_id,
                SIM_FILE_HDF5_GZ,
                to_file=hdf5_gz_file_path,
                verbose=verbose,
                progress_callback=progress_callback,
            )
            if os.path.exists(hdf5_gz_file_path):
                extract_gz_file(hdf5_gz_file_path, to_file)
            else:
                raise WebError("Failed to download simulation.hdf5")
        finally:
            os.unlink(hdf5_gz_file_path)

    def get_running_info(self) -> Tuple[float, float]:
        """Gets the % done and field_decay for a running task.

        Returns
        -------
        perc_done : float
            Percentage of run done (in terms of max number of time steps).
            Is ``None`` if run info not available.
        field_decay : float
            Average field intensity normalized to max value (1.0).
            Is ``None`` if run info not available.
        """
        assert self.task_id

        resp = http.get(f"tidy3d/tasks/{self.task_id}/progress")
        perc_done = resp.get("perc_done")
        field_decay = resp.get("field_decay")
        return perc_done, field_decay

    def get_log(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> pathlib.Path:
        """Get log file from Server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """

        assert self.task_id
        return download_file(
            self.task_id,
            SIM_LOG_FILE,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def abort(self):
        """Abort current task from server."""
        if not self.task_id:
            raise ValueError("Task id not found.")
        return http.put("tidy3d/tasks/abort", json={"taskType": "FDTD", "taskId": self.task_id})
