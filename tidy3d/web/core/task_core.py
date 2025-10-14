"""Tidy3d webapi types."""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
from datetime import datetime
from typing import Callable, Optional, Union

from botocore.exceptions import ClientError
from pydantic.v1 import Extra, Field, parse_obj_as

import tidy3d as td
from tidy3d.exceptions import ValidationError
from tidy3d.web.common import REFRESH_TIME

from . import http_util
from .cache import FOLDER_CACHE
from .constants import (
    SIM_ERROR_FILE,
    SIM_FILE_HDF5_GZ,
    SIM_LOG_FILE,
    SIMULATION_DATA_HDF5_GZ,
)
from .core_config import get_logger_console
from .environment import Env
from .exceptions import WebError, WebNotFoundError
from .file_util import read_simulation_from_hdf5
from .http_util import get_version as _get_protocol_version
from .http_util import http
from .s3utils import download_file, download_gz_file, upload_file
from .stub import TaskStub
from .task_info import BatchDetail
from .types import PayType, Queryable, ResourceLifecycle, Submittable, Tidy3DResource


class Folder(Tidy3DResource, Queryable, extra=Extra.allow):
    """Tidy3D Folder."""

    folder_id: str = Field(..., title="Folder id", description="folder id", alias="projectId")
    folder_name: str = Field(
        ..., title="Folder name", description="folder name", alias="projectName"
    )

    @classmethod
    def list(cls, projects_endpoint: str = "tidy3d/projects") -> []:
        """List all folders.

        Returns
        -------
        folders : [Folder]
            List of folders
        """
        resp = http.get(projects_endpoint)
        return (
            parse_obj_as(
                list[Folder],
                resp,
            )
            if resp
            else None
        )

    @classmethod
    def get(
        cls,
        folder_name: str,
        create: bool = False,
        projects_endpoint: str = "tidy3d/projects",
        project_endpoint: str = "tidy3d/project",
    ):
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
            resp = http.get(project_endpoint, params={"projectName": folder_name})
            if resp:
                folder = Folder(**resp)
        if create and not folder:
            resp = http.post(projects_endpoint, {"projectName": folder_name})
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

    def delete(self, projects_endpoint: str = "tidy3d/projects"):
        """Remove this folder."""

        http.delete(f"{projects_endpoint}/{self.folder_id}")

    def delete_old(self, days_old: int) -> int:
        """Remove folder contents older than ``days_old``."""

        return http.delete(
            f"tidy3d/tasks/{self.folder_id}/tasks",
            params={"daysOld": days_old},
        )

    def list_tasks(self, projects_endpoint: str = "tidy3d/projects") -> list[Tidy3DResource]:
        """List all tasks in this folder.

        Returns
        -------
        tasks : List[:class:`.SimulationTask`]
            List of tasks in this folder
        """
        resp = http.get(f"{projects_endpoint}/{self.folder_id}/tasks")
        return (
            parse_obj_as(
                list[SimulationTask],
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
    folder_id: Optional[str] = Field(
        None,
        title="folder_id",
        description="Folder ID number, set when the task is uploaded, leave as None.",
        alias="folderId",
    )
    status: Optional[str] = Field(title="status", description="Simulation task status.")

    real_flex_unit: float = Field(
        None, title="real FlexCredits", description="Billed FlexCredits.", alias="realCost"
    )

    created_at: Optional[datetime] = Field(
        title="created_at", description="Time at which this task was created.", alias="createdAt"
    )

    task_type: Optional[str] = Field(
        title="task_type", description="The type of task.", alias="taskType"
    )

    folder_name: Optional[str] = Field(
        "default",
        title="Folder Name",
        description="Name of the folder associated with this task.",
        alias="folderName",
    )

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

    @classmethod
    def create(
        cls,
        task_type: str,
        task_name: str,
        folder_name: str = "default",
        callback_url: Optional[str] = None,
        simulation_type: str = "tidy3d",
        parent_tasks: Optional[list[str]] = None,
        file_type: str = "Gz",
        port_name_list: Optional[list[str]] = None,
        projects_endpoint: str = "tidy3d/projects",
    ) -> SimulationTask:
        """Create a new task on the server.

        Parameters
        ----------
        task_type: :class".TaskType"
            The type of task.
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

        # handle backwards compatibility, "tidy3d" is the default simulation_type
        if simulation_type is None:
            simulation_type = "tidy3d"

        folder = Folder.get(folder_name, create=True)
        payload = {
            "taskName": task_name,
            "taskType": task_type,
            "callbackUrl": callback_url,
            "simulationType": simulation_type,
            "parentTasks": parent_tasks,
            "fileType": file_type,
        }
        # Component modeler: include port names if provided
        if port_name_list:
            # Align with backend contract: expect 'portNames' (not 'portNameList')
            payload["portNames"] = port_name_list

        resp = http.post(f"{projects_endpoint}/{folder.folder_id}/tasks", payload)
        # RF group creation may return group-level info without 'taskId'.
        # Use 'groupId' (or 'batchId' as fallback) as the resource id for subsequent uploads.
        if "taskId" not in resp and task_type == "RF":
            # Prefer using 'batchId' as the resource id for uploads (S3 STS expects a task-like id).
            if "batchId" in resp:
                resp["taskId"] = resp["batchId"]
            elif "groupId" in resp:
                resp["taskId"] = resp["groupId"]
            else:
                raise WebError("Missing resource ID for task creation. Contact customer support.")
        return SimulationTask(**resp, taskType=task_type, folder_name=folder_name)

    @classmethod
    def get(cls, task_id: str, verbose: bool = True) -> SimulationTask:
        """Get task from the server by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of task on server.
        verbose:
            If `True`, will print progressbars and status, otherwise, will run silently.

        Returns
        -------
        :class:`.SimulationTask`
            :class:`.SimulationTask` object containing info about status,
             size, credits of task and others.
        """
        try:
            resp = http.get(f"tidy3d/tasks/{task_id}/detail")
        except WebNotFoundError as e:
            td.log.error(f"The requested task ID '{task_id}' does not exist.")
            raise e

        task = SimulationTask(**resp) if resp else None
        return task

    @classmethod
    def get_running_tasks(cls) -> list[SimulationTask]:
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
        return parse_obj_as(list[SimulationTask], resp)

    def delete(self, versions: bool = False):
        """Delete current task from server.

        Parameters
        ----------
        versions : bool = False
            If ``True``, delete all versions of the task in the task group. Otherwise, delete only the version associated with the current task ID.
        """
        if not self.task_id:
            raise ValueError("Task id not found.")

        task_details = http.get(f"tidy3d/tasks/{self.task_id}")

        if task_details and "groupId" in task_details and "version" in task_details:
            group_id = task_details["groupId"]
            version = task_details["version"]
            if versions:
                http.delete("tidy3d/group", json={"groupIds": [group_id]})
            else:
                http.delete(f"tidy3d/group/{group_id}/versions", json={"versions": [version]})
        else:  # Fallback to old method if we can't get the groupId and version
            http.delete(f"tidy3d/tasks/{self.task_id}")

    def get_simulation_json(self, to_file: str, verbose: bool = True) -> pathlib.Path:
        """Get json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: str
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        hdf5_file, hdf5_file_path = tempfile.mkstemp(".hdf5")
        os.close(hdf5_file)
        try:
            self.get_simulation_hdf5(hdf5_file_path)
            if os.path.exists(hdf5_file_path):
                json_string = read_simulation_from_hdf5(hdf5_file_path)
                with open(to_file, "w") as file:
                    # Write the string to the file
                    file.write(json_string.decode("utf-8"))
                    if verbose:
                        console = get_logger_console()
                        console.log(f"Generate {to_file} successfully.")
            else:
                raise WebError("Failed to download simulation.json.")
        finally:
            os.unlink(hdf5_file_path)

    def upload_simulation(
        self,
        stub: TaskStub,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        remote_sim_file: str = SIM_FILE_HDF5_GZ,
    ) -> None:
        """Upload :class:`.Simulation` object to Server.

        Parameters
        ----------
        stub: :class:`TaskStub`
            An instance of TaskStub.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while uploading the data.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")
        if not stub:
            raise WebError("Expected field 'simulation' is unset.")
        # Also upload hdf5.gz containing all data.
        file, file_name = tempfile.mkstemp()
        os.close(file)
        try:
            # upload simulation
            # compress .hdf5 to .hdf5.gz
            stub.to_hdf5_gz(file_name)
            upload_file(
                self.task_id,
                file_name,
                remote_sim_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)

    def upload_file(
        self,
        local_file: str,
        remote_filename: str,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
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
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        upload_file(
            self.task_id,
            local_file,
            remote_filename,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def submit(
        self,
        solver_version: Optional[str] = None,
        worker_group: Optional[str] = None,
        pay_type: Union[PayType, str] = PayType.AUTO,
        priority: Optional[int] = None,
    ):
        """Kick off this task.

        It will be uploaded to server before
        starting the task. Otherwise, this method assumes that the Simulation has been uploaded by
        the upload_file function, so the task will be kicked off directly.

        Parameters
        ----------
        solver_version: str = None
            target solver version.
        worker_group: str = None
            worker group
        pay_type: Union[PayType, str] = PayType.AUTO
            Which method to pay the simulation.
        priority: int = None
            Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
            It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
        """
        pay_type = PayType(pay_type) if not isinstance(pay_type, PayType) else pay_type

        if solver_version:
            protocol_version = None
        else:
            protocol_version = http_util.get_version()

        http.post(
            f"tidy3d/tasks/{self.task_id}/submit",
            {
                "solverVersion": solver_version,
                "workerGroup": worker_group,
                "protocolVersion": protocol_version,
                "enableCaching": Env.current.enable_caching,
                "payType": pay_type.value,
                "priority": priority,
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
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        if solver_version:
            protocol_version = None
        else:
            protocol_version = http_util.get_version()

        resp = http.post(
            f"tidy3d/tasks/{self.task_id}/metadata",
            {
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
            },
        )
        return resp

    def get_sim_data_hdf5(
        self,
        to_file: str,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        remote_data_file: str = SIMULATION_DATA_HDF5_GZ,
    ) -> pathlib.Path:
        """Get simulation data file from Server.

        Parameters
        ----------
        to_file: str
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        file = None
        try:
            file = download_gz_file(
                resource_id=self.task_id,
                remote_filename=remote_data_file,
                to_file=to_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        except ClientError:
            if verbose:
                console = get_logger_console()
                console.log(f"Unable to download '{remote_data_file}'.")

        if not file:
            try:
                file = download_file(
                    resource_id=self.task_id,
                    remote_filename=remote_data_file[:-3],
                    to_file=to_file,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                raise WebError(
                    "Failed to download the simulation data file from the server. "
                    "Please confirm that the task was successfully run."
                ) from e

        return file

    def get_simulation_hdf5(
        self,
        to_file: str,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        remote_sim_file: str = SIM_FILE_HDF5_GZ,
    ) -> pathlib.Path:
        """Get simulation.hdf5 file from Server.

        Parameters
        ----------
        to_file: str
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        return download_gz_file(
            resource_id=self.task_id,
            remote_filename=remote_sim_file,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_running_info(self) -> tuple[float, float]:
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
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        resp = http.get(f"tidy3d/tasks/{self.task_id}/progress")
        perc_done = resp.get("perc_done")
        field_decay = resp.get("field_decay")
        return perc_done, field_decay

    def get_log(
        self,
        to_file: str,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> pathlib.Path:
        """Get log file from Server.

        Parameters
        ----------
        to_file: str
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """

        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        return download_file(
            self.task_id,
            SIM_LOG_FILE,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_error_json(self, to_file: str, verbose: bool = True) -> pathlib.Path:
        """Get error json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: str
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        return download_file(
            self.task_id,
            SIM_ERROR_FILE,
            to_file=to_file,
            verbose=verbose,
        )

    def abort(self):
        """Aborting current task from server."""
        if not self.task_id:
            raise ValueError("Task id not found.")
        return http.put(
            "tidy3d/tasks/abort", json={"taskType": self.task_type, "taskId": self.task_id}
        )

    def validate_post_upload(self, parent_tasks: Optional[list[str]] = None):
        """Perform checks after task is uploaded and metadata is processed."""
        if self.task_type == "HEAT_CHARGE" and parent_tasks:
            try:
                if len(parent_tasks) > 1:
                    raise ValueError(
                        "A single parent 'task_id' corresponding to the task in which the meshing "
                        "was run must be provided."
                    )
                try:
                    # get mesh task info
                    mesh_task = SimulationTask.get(parent_tasks[0], verbose=False)
                    assert mesh_task.task_type == "VOLUME_MESH"
                    assert mesh_task.status == "success"
                    # get up-to-date task info
                    task = SimulationTask.get(self.task_id, verbose=False)
                    if task.fileMd5 != mesh_task.childFileMd5:
                        raise ValidationError(
                            "Simulation stored in parent task 'VolumeMesher' does not match the "
                            "current simulation."
                        )
                except Exception as e:
                    raise ValidationError(
                        "The parent task must be a 'VolumeMesher' task which has been successfully "
                        "run and is associated to the same 'HeatChargeSimulation' as provided here."
                    ) from e

            except Exception as e:
                raise WebError(f"Provided 'parent_tasks' failed validation: {e!s}") from e


class BatchTask:
    """Provides a client-side interface for managing a remote batch task.

    This class acts as a wrapper around the API endpoints for a specific batch,
    allowing users to check, submit, monitor, and download data from it.

    Note:
        The 'batch_type' (e.g., "RF_SWEEP") must be provided by the caller to
        most methods, as it dictates which backend service handles the request.
    """

    def __init__(self, batch_id: str):
        self.batch_id = batch_id

    @staticmethod
    def is_batch(resource_id: str, batch_type: str) -> bool:
        """Checks if a given resource ID corresponds to a valid batch task.

        This is a utility function to verify a batch task's existence before
        instantiating the class.

        Parameters
        ----------
        resource_id : str
            The unique identifier for the resource.
        batch_type : str
            The type of the batch to check (e.g., "RF_SWEEP").

        Returns
        -------
        bool
            ``True`` if the resource is a valid batch task, ``False`` otherwise.
        """
        try:
            # TODO PROPERLY FIXME
            # Disable non critical logs due to check for resourceId, until we have a dedicated API for this
            resp = http.get(
                f"tidy3d/tasks/{resource_id}/batch-detail",
                params={"batchType": batch_type},
                suppress_404=True,
            )
            status = bool(resp and isinstance(resp, dict) and "status" in resp)
            return status
        except Exception:
            return False

    def detail(self, batch_type: str) -> BatchDetail:
        """Fetches the detailed information and status of the batch.

        Parameters
        ----------
        batch_type : str
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        BatchDetail
            An object containing the batch's latest data.
        """
        resp = http.get(
            f"tidy3d/tasks/{self.batch_id}/batch-detail",
            params={"batchType": batch_type},
        )
        # Some backends may return null for collection fields; coerce to sensible defaults
        if isinstance(resp, dict):
            if resp.get("tasks") is None:
                resp["tasks"] = []
        return BatchDetail(**(resp or {}))

    def check(
        self,
        solver_version: Optional[str] = None,
        protocol_version: Optional[str] = None,
        batch_type: str = "",
    ):
        """Submits a request to validate the batch configuration on the server.

        Parameters
        ----------
        solver_version : Optional[str], default=None
            The version of the solver to use for validation.
        protocol_version : Optional[str], default=None
            The data protocol version. Defaults to the current version.
        batch_type : str, default=""
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        Any
            The server's response to the check request.
        """
        if protocol_version is None:
            protocol_version = _get_protocol_version()
        return http.post(
            f"tidy3d/projects/{self.batch_id}/batch-check",
            {
                "batchType": batch_type,
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
            },
        )

    def submit(
        self,
        solver_version: Optional[str] = None,
        protocol_version: Optional[str] = None,
        worker_group: Optional[str] = None,
        batch_type: str = "",
    ):
        """Submits the batch for execution on the server.

        Parameters
        ----------
        solver_version : Optional[str], default=None
            The version of the solver to use for execution.
        protocol_version : Optional[str], default=None
            The data protocol version. Defaults to the current version.
        worker_group : Optional[str], default=None
            Optional identifier for a specific worker group to run on.
        batch_type : str, default=""
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        Any
            The server's response to the submit request.
        """
        if protocol_version is None:
            protocol_version = _get_protocol_version()
        return http.post(
            f"tidy3d/projects/{self.batch_id}/batch-submit",
            {
                "batchType": batch_type,
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
                "workerGroup": worker_group,
            },
        )

    def postprocess(
        self,
        solver_version: Optional[str] = None,
        protocol_version: Optional[str] = None,
        worker_group: Optional[str] = None,
        batch_type: str = "",
    ):
        """Initiates post-processing for a completed batch run.

        Parameters
        ----------
        solver_version : Optional[str], default=None
            The version of the solver to use for post-processing.
        protocol_version : Optional[str], default=None
            The data protocol version. Defaults to the current version.
        worker_group : Optional[str], default=None
            Optional identifier for a specific worker group to run on.
        batch_type : str, default=""
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        Any
            The server's response to the post-process request.
        """
        if protocol_version is None:
            protocol_version = _get_protocol_version()
        return http.post(
            f"tidy3d/projects/{self.batch_id}/postprocess",
            {
                "batchType": batch_type,
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
                "workerGroup": worker_group,
            },
        )

    def wait_for_validate(
        self, timeout: Optional[float] = None, batch_type: str = ""
    ) -> BatchDetail:
        """Waits for the batch to complete the validation stage by polling its status.

        Parameters
        ----------
        timeout : Optional[float], default=None
            Maximum time in seconds to wait. If ``None``, waits indefinitely.
        batch_type : str, default=""
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        BatchDetail
            The final object after validation completes or a timeout occurs.

        Notes
        -----
        This method blocks until the batch status is 'validate_success',
        'validate_warn', 'validate_fail', or another terminal state like 'blocked'
        or 'aborted', or until the timeout is reached.
        """
        start = datetime.now().timestamp()
        while True:
            d = self.detail(batch_type=batch_type)
            status = d.totalStatus
            if status in ("validate_success", "validate_warn", "validate_fail"):
                return d
            if status in ("blocked", "aborting", "aborted"):
                return d
            if timeout is not None and (datetime.now().timestamp() - start) > timeout:
                return d
            time.sleep(REFRESH_TIME)

    def wait_for_run(self, timeout: Optional[float] = None, batch_type: str = "") -> BatchDetail:
        """Waits for the batch to complete the execution stage by polling its status.

        Parameters
        ----------
        timeout : Optional[float], default=None
            Maximum time in seconds to wait. If ``None``, waits indefinitely.
        batch_type : str, default=""
            The type of the batch (e.g., "RF_SWEEP").

        Returns
        -------
        BatchDetail
            The final object after the run completes or a timeout occurs.

        Notes
        -----
        This method blocks until the batch status reaches a terminal run state like
        'run_success', 'run_failed', 'diverged', 'blocked', or 'aborted',
        or until the timeout is reached.
        """
        start = datetime.now().timestamp()
        while True:
            d = self.detail(batch_type=batch_type)
            status = d.totalStatus
            if status in (
                "run_success",
                "run_failed",
                "diverged",
                "blocked",
                "aborting",
                "aborted",
            ):
                return d
            if timeout is not None and (datetime.now().timestamp() - start) > timeout:
                return d
            time.sleep(REFRESH_TIME)

    def get_data_hdf5(
        self,
        remote_data_file_gz: str,
        to_file: str,
        verbose: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> pathlib.Path:
        """Downloads a batch data artifact, with a fallback mechanism.

        Parameters
        ----------
        remote_data_file_gz : str
            Remote gzipped filename to download (e.g., 'output/cm_data.hdf5.gz').
        to_file : str
            Local path where the downloaded file will be saved.
        verbose : bool, default=True
            If ``True``, shows progress logs and messages.
        progress_callback : Optional[Callable[[float], None]], default=None
            Optional callback function for progress updates, which receives the
            download percentage as a float.

        Returns
        -------
        pathlib.Path
            An object pointing to the downloaded local file.

        Raises
        ------
        WebError
            If both the gzipped and uncompressed file downloads fail.

        Notes
        -----
        This method first attempts to download the gzipped version of a file.
        If that fails, it falls back to downloading the uncompressed version.
        """
        file = None
        try:
            file = download_gz_file(
                resource_id=self.batch_id,
                remote_filename=remote_data_file_gz,
                to_file=to_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        except ClientError:
            if verbose:
                console = get_logger_console()
                console.log(f"Unable to download '{remote_data_file_gz}'.")

        if not file:
            try:
                file = download_file(
                    resource_id=self.batch_id,
                    remote_filename=remote_data_file_gz[:-3],
                    to_file=to_file,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                raise WebError(
                    "Failed to download the batch data file from the server. "
                    "Please confirm that the batch has been successfully postprocessed."
                ) from e

        return file
