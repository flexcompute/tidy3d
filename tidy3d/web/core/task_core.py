"""Tidy3d webapi types."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from pydantic import Field, TypeAdapter

import tidy3d as td
from tidy3d.config import config
from tidy3d.config.sections import VALID_VGPU_ALLOCATIONS
from tidy3d.exceptions import ValidationError, format_chained_exception_message

from . import http_util
from .cache import FOLDER_CACHE
from .constants import (
    SIM_ERROR_FILE,
    SIM_FILE_HDF5_GZ,
    SIM_LOG_FILE,
    SIM_VALIDATION_FILE,
    SIMULATION_DATA_HDF5_GZ,
)
from .core_config import get_logger_console
from .exceptions import WebError, WebNotFoundError
from .file_util import read_simulation_from_hdf5
from .http_util import get_version as _get_protocol_version
from .http_util import http
from .s3utils import download_file, download_gz_file, upload_file
from .task_info import BatchDetail, TaskInfo
from .types import PayType, Queryable, ResourceLifecycle, Submittable, Tidy3DResource

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike

    import requests

    from .stub import TaskStub


def _serialize_additional_payload(
    additional_payload: dict[str, Any] | str | None,
) -> str | None:
    """Serialize additional submit payloads to JSON strings."""

    if additional_payload is None or isinstance(additional_payload, str):
        return additional_payload
    return json.dumps(additional_payload)


class Folder(Tidy3DResource, Queryable, extra="allow"):
    """Tidy3D Folder."""

    folder_id: str = Field(
        title="Folder id",
        description="folder id",
        alias="projectId",
    )
    folder_name: str = Field(
        title="Folder name",
        description="folder name",
        alias="projectName",
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
        return TypeAdapter(list[Folder]).validate_python(resp) if resp else None

    @classmethod
    def get(
        cls,
        folder_name: str,
        create: bool = False,
        projects_endpoint: str = "tidy3d/projects",
        project_endpoint: str = "tidy3d/project",
    ) -> Folder:
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
    def create(cls, folder_name: str) -> Folder:
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

    def delete(self, projects_endpoint: str = "tidy3d/projects") -> None:
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
        tasks : list[:class:`.SimulationTask`]
            List of tasks in this folder
        """
        resp = http.get(f"{projects_endpoint}/{self.folder_id}/tasks")
        return TypeAdapter(list[SimulationTask]).validate_python(resp) if resp else None


class WebTask(ResourceLifecycle, Submittable, extra="allow"):
    """Interface for managing the running a task on the server."""

    task_id: str | None = Field(
        None,
        title="task_id",
        description="Task ID number, set when the task is uploaded, leave as None.",
        alias="taskId",
    )

    @classmethod
    def create(
        cls,
        task_type: str,
        task_name: str,
        folder_name: str = "default",
        callback_url: str | None = None,
        simulation_type: str = "tidy3d",
        parent_tasks: list[str] | None = None,
        file_type: str = "Gz",
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
        parent_tasks : list[str]
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
        if task_type in ["RF", "TERMINAL_CM", "MODAL_CM"]:
            payload = {
                "groupName": task_name,
                "folderId": folder.folder_id,
                "fileType": file_type,
                "taskType": task_type,
            }
            resp = http.post("rf/task", payload)
        else:
            payload = {
                "taskName": task_name,
                "taskType": task_type,
                "callbackUrl": callback_url,
                "simulationType": simulation_type,
                "parentTasks": parent_tasks,
                "fileType": file_type,
            }
            resp = http.post(f"{projects_endpoint}/{folder.folder_id}/tasks", payload)
        return SimulationTask(**resp, taskType=task_type, folder_name=folder_name)

    def get_url(self) -> str:
        base = str(config.web.website_endpoint or "")
        if isinstance(self, BatchTask):
            return "/".join([base.rstrip("/"), f"rf?taskId={self.task_id}"])
        return "/".join([base.rstrip("/"), f"workbench?taskId={self.task_id}"])

    def get_folder_url(self) -> str | None:
        folder_id = getattr(self, "folder_id", None)
        if not folder_id:
            return None
        base = str(config.web.website_endpoint or "")
        return "/".join([base.rstrip("/"), f"folders/{folder_id}"])

    def get_log(
        self,
        to_file: PathLike,
        verbose: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> pathlib.Path:
        """Get log file from Server.

        Parameters
        ----------
        to_file: PathLike
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

        target_path = pathlib.Path(to_file)

        return download_file(
            self.task_id,
            SIM_LOG_FILE,
            to_file=target_path,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_data_hdf5(
        self,
        to_file: PathLike,
        remote_data_file_gz: PathLike = SIMULATION_DATA_HDF5_GZ,
        verbose: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> pathlib.Path:
        """Download data artifact (simulation or batch) with gz fallback handling.

        Parameters
        ----------
        remote_data_file_gz : PathLike
            Gzipped remote filename.
        to_file : PathLike
            Local target path.
        verbose : bool
            Whether to log progress.
        progress_callback : Optional[Callable[[float], None]]
            Progress callback.

        Returns
        -------
        pathlib.Path
            Saved local path.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")
        target_path = pathlib.Path(to_file)
        file = None
        try:
            file = download_gz_file(
                resource_id=self.task_id,
                remote_filename=remote_data_file_gz,
                to_file=target_path,
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
                    resource_id=self.task_id,
                    remote_filename=str(remote_data_file_gz)[:-3],
                    to_file=target_path,
                    verbose=verbose,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                raise WebError(
                    format_chained_exception_message(
                        "Failed to download the data file from the server. Please confirm "
                        "that the task completed successfully.",
                        e,
                    )
                ) from e
        return file

    @staticmethod
    def is_batch(resource_id: str) -> bool:
        """Checks if a given resource ID corresponds to a valid batch task.

        This is a utility function to verify a batch task's existence before
        instantiating the class.

        Parameters
        ----------
        resource_id : str
            The unique identifier for the resource.

        Returns
        -------
        bool
            ``True`` if the resource is a valid batch task, ``False`` otherwise.
        """
        try:
            # TODO PROPERLY FIXME
            # Disable non critical logs due to check for resourceId, until we have a dedicated API for this
            resp = http.get(
                f"rf/task/{resource_id}/statistics",
                suppress_404=True,
            )
            status = bool(resp and isinstance(resp, dict) and "status" in resp)
            return status
        except Exception:
            return False

    def delete(self, versions: bool = False) -> None:
        """Delete current task from server.

        Parameters
        ----------
        versions : bool = False
            If ``True``, delete all versions of the task in the task group. Otherwise, delete only
            the version associated with the current task ID.
        """
        if not self.task_id:
            raise ValueError("Task id not found.")

        task_details = self.detail().model_dump()

        if task_details and "groupId" in task_details:
            group_id = task_details["groupId"]
            if versions:
                http.delete("tidy3d/group", json={"groupIds": [group_id]})
                return
            elif "version" in task_details:
                version = task_details["version"]
                http.delete(f"tidy3d/group/{group_id}/versions", json={"versions": [version]})
                return

        # Fallback to old method if we can't get the groupId and version
        http.delete(f"tidy3d/tasks/{self.task_id}")


class SimulationTask(WebTask):
    """Interface for managing the running of solver tasks on the server."""

    folder_id: str | None = Field(
        None,
        title="folder_id",
        description="Folder ID number, set when the task is uploaded, leave as None.",
        alias="folderId",
    )
    status: str | None = Field(None, title="status", description="Simulation task status.")

    real_flex_unit: float | None = Field(
        None, title="real FlexCredits", description="Billed FlexCredits.", alias="realCost"
    )

    created_at: datetime | None = Field(
        None,
        title="created_at",
        description="Time at which this task was created.",
        alias="createdAt",
    )

    task_type: str | None = Field(
        None, title="task_type", description="The type of task.", alias="taskType"
    )

    folder_name: str | None = Field(
        "default",
        title="Folder Name",
        description="Name of the folder associated with this task.",
        alias="folderName",
    )

    callback_url: str | None = Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    # simulation_type: str = Field(
    #     None,
    #     title="Simulation Type",
    #     description="Type of simulation, used internally only.",
    # )

    # parent_tasks: Tuple[TaskId, ...] = Field(
    #     None,
    #     title="Parent Tasks",
    #     description="List of parent task ids for the simulation, used internally only."
    # )

    @classmethod
    def get(cls, task_id: str, verbose: bool = True) -> SimulationTask | None:
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
             size, credits of task and others, or ``None`` if the detail
             response is empty.
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
        return TypeAdapter(list[SimulationTask]).validate_python(resp)

    def detail(self) -> TaskInfo:
        """Fetches the detailed information and status of the task.

        Returns
        -------
        TaskInfo
            An object containing the task's latest data.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")
        resp = http.get(f"tidy3d/tasks/{self.task_id}/detail")
        if not isinstance(resp, dict):
            raise WebError("Expected task detail response to be a JSON object.")
        return TaskInfo(**{"taskId": self.task_id, "taskType": self.task_type, **resp})

    def get_simulation_json(self, to_file: PathLike, verbose: bool = True) -> None:
        """Get json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: PathLike
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        to_file = pathlib.Path(to_file)

        hdf5_file, hdf5_file_path = tempfile.mkstemp(".hdf5")
        os.close(hdf5_file)
        try:
            self.get_simulation_hdf5(hdf5_file_path)
            if os.path.exists(hdf5_file_path):
                json_string = read_simulation_from_hdf5(hdf5_file_path)
                to_file.parent.mkdir(parents=True, exist_ok=True)
                with to_file.open("w", encoding="utf-8") as file:
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
        progress_callback: Callable[[float], None] | None = None,
        remote_sim_file: PathLike = SIM_FILE_HDF5_GZ,
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
        local_file: PathLike,
        remote_filename: str,
        verbose: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        """
        Upload file to platform. Using this method when the json file is too large to parse
         as :class".simulation".
        Parameters
        ----------
        local_file: PathLike
            Local file path.
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
        solver_version: str | None = None,
        worker_group: str | None = None,
        pay_type: PayType | str = PayType.AUTO,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        additional_payload: dict[str, Any] | str | None = None,
    ) -> None:
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
        vgpu_allocation : int = None
            Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
            Only applies to vGPU license users. If not specified, the system
            automatically determines the optimal GPU count.
        ignore_memory_limit : Optional[bool] = None
            If ``True``, allows the simulation to run even when estimated vGPU memory
            exceeds the allocation limit (up to 2x the limit). Only applies to
            vGPU license users. Default ``None`` leaves the server behaviour unchanged.
        additional_payload : Optional[Union[dict[str, Any], str]] = None
            Additional submit payload. Dict values are JSON-serialized and sent
            under ``additionalPayload``.
        """
        pay_type = PayType(pay_type) if not isinstance(pay_type, PayType) else pay_type

        if solver_version:
            protocol_version = None
        else:
            protocol_version = http_util.get_version()

        payload = {
            "solverVersion": solver_version,
            "workerGroup": worker_group,
            "protocolVersion": protocol_version,
            "enableCaching": config.web.enable_caching,
            "payType": pay_type.value,
            "priority": priority,
            "vgpuAllocation": vgpu_allocation,
            "ignoreMemoryLimit": ignore_memory_limit,
        }
        serialized_additional_payload = _serialize_additional_payload(additional_payload)
        if serialized_additional_payload is not None:
            payload["additionalPayload"] = serialized_additional_payload

        http.post(
            f"tidy3d/tasks/{self.task_id}/submit",
            payload,
        )

    def estimate_cost(self, solver_version: str | None = None) -> float:
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

    def get_simulation_hdf5(
        self,
        to_file: PathLike,
        verbose: bool = True,
        progress_callback: Callable[[float], None] | None = None,
        remote_sim_file: PathLike = SIM_FILE_HDF5_GZ,
    ) -> pathlib.Path:
        """Get simulation.hdf5 file from Server.

        Parameters
        ----------
        to_file: PathLike
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

        target_path = pathlib.Path(to_file)

        return download_gz_file(
            resource_id=self.task_id,
            remote_filename=remote_sim_file,
            to_file=target_path,
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
        to_file: PathLike,
        verbose: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> pathlib.Path:
        """Get log file from Server.

        Parameters
        ----------
        to_file: PathLike
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

        target_path = pathlib.Path(to_file)

        return download_file(
            self.task_id,
            SIM_LOG_FILE,
            to_file=target_path,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_error_json(
        self, to_file: PathLike, verbose: bool = True, validation: bool = False
    ) -> pathlib.Path:
        """Get error json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: PathLike
            Save file to path.
        verbose: bool = True
            Whether to display progress bars.
        validation: bool = False
            Whether to get a validation error file or a solver error file.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        if not self.task_id:
            raise WebError("Expected field 'task_id' is unset.")

        target_path = pathlib.Path(to_file)
        target_file = SIM_ERROR_FILE if not validation else SIM_VALIDATION_FILE

        return download_file(
            self.task_id,
            target_file,
            to_file=target_path,
            verbose=verbose,
        )

    def abort(self) -> requests.Response:
        """Abort the current task on the server."""
        if not self.task_id:
            raise ValueError("Task id not found.")
        return http.put(
            "tidy3d/tasks/abort", json={"taskType": self.task_type, "taskId": self.task_id}
        )

    def validate_post_upload(self, parent_tasks: list[str] | None = None) -> None:
        """Perform checks after task is uploaded and metadata is processed."""
        if self.task_type in {"HEAT", "HEAT_CHARGE"} and parent_tasks:
            try:
                if len(parent_tasks) > 1:
                    raise ValueError(
                        "A single parent 'task_id' corresponding to the task in which the meshing "
                        "was run must be provided."
                    )
                try:
                    # get mesh task info
                    mesh_task = SimulationTask.get(parent_tasks[0], verbose=False)
                    if mesh_task is None:
                        raise WebError("Unable to fetch parent task details.")
                    assert mesh_task.task_type == "VOLUME_MESH"
                    assert mesh_task.status == "success"
                    # get up-to-date task info
                    task = SimulationTask.get(self.task_id, verbose=False)
                    if task is None:
                        raise WebError("Unable to fetch task details.")
                    if task.fileMd5 != mesh_task.childFileMd5:
                        raise ValidationError(
                            "Simulation stored in parent task 'VolumeMesher' does not match the "
                            "current simulation."
                        )
                except Exception as e:
                    raise ValidationError(
                        format_chained_exception_message(
                            "The parent task must be a 'VolumeMesher' task which has been "
                            "successfully run and is associated to the same simulation as "
                            "provided here.",
                            e,
                        )
                    ) from e

            except Exception as e:
                raise WebError(
                    format_chained_exception_message("Provided 'parent_tasks' failed validation", e)
                ) from e


class BatchTask(WebTask):
    """Interface for managing a batch task on the server."""

    task_type: str | None = Field(
        None, title="task_type", description="The type of task.", alias="taskType"
    )
    status: str | None = Field(None, title="status", description="The status of the task.")

    @classmethod
    def get(cls, task_id: str, verbose: bool = True) -> BatchTask:
        """Get batch task by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of batch on server.
        verbose:
            If `True`, will print progressbars and status, otherwise, will run silently.

        Returns
        -------
        :class:`.BatchTask` | None
            BatchTask object if found, otherwise None.
        """
        try:
            resp = http.get(f"rf/task/{task_id}/statistics")
        except WebNotFoundError as e:
            td.log.error(f"The requested batch ID '{task_id}' does not exist.")
            raise e
        # Extract taskType from response if available
        if resp:
            task_type = resp.get("taskType") if isinstance(resp, dict) else None
            status = resp.get("status") if isinstance(resp, dict) else None
            return BatchTask(taskId=task_id, taskType=task_type, status=status)
        return None

    def detail(self) -> BatchDetail:
        """Fetches the detailed information and status of the batch.

        Returns
        -------
        BatchDetail
            An object containing the batch's latest data.
        """
        resp = http.get(
            f"rf/task/{self.task_id}/statistics",
        )
        # Some backends may return null for collection fields; coerce to sensible defaults
        if isinstance(resp, dict):
            if resp.get("tasks") is None:
                resp["tasks"] = []
        return BatchDetail(**(resp or {}))

    def check(
        self,
        check_task_type: str,
        solver_version: str | None = None,
        protocol_version: str | None = None,
    ) -> requests.Response:
        """Submits a request to validate the batch configuration on the server.

        Parameters
        ----------
        solver_version : Optional[str], default=None
            The version of the solver to use for validation.
        protocol_version : Optional[str], default=None
            The data protocol version. Defaults to the current version.

        Returns
        -------
        Any
            The server's response to the check request.
        """
        if protocol_version is None:
            protocol_version = _get_protocol_version()
        return http.post(
            f"rf/task/{self.task_id}/check",
            {
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
                "taskType": check_task_type,
            },
        )

    def submit(
        self,
        solver_version: str | None = None,
        protocol_version: str | None = None,
        worker_group: str | None = None,
        pay_type: PayType | str = PayType.AUTO,
        priority: int | None = None,
        vgpu_allocation: int | None = None,
        ignore_memory_limit: bool | None = None,
        additional_payload: dict[str, Any] | str | None = None,
    ) -> requests.Response:
        """Submits the batch for execution on the server.

        Parameters
        ----------
        solver_version : Optional[str], default=None
            The version of the solver to use for execution.
        protocol_version : Optional[str], default=None
            The data protocol version. Defaults to the current version.
        worker_group : Optional[str], default=None
            Optional identifier for a specific worker group to run on.
        priority : Optional[int], default=None
            Priority of the batch in the vGPU queue, where 1 is lowest and 10 is highest.
        vgpu_allocation : Optional[int], default=None
            Number of virtual GPUs to allocate for the batch (1, 2, 4, or 8).
        ignore_memory_limit : Optional[bool], default=None
            Not supported for batch tasks. Passing a non-``None`` value raises
            :class:`NotImplementedError`.
        additional_payload : Optional[Union[dict[str, Any], str]], default=None
            Additional submit payload. Dict values are JSON-serialized and sent
            under ``additionalPayload``.

        Returns
        -------
        Any
            The server's response to the submit request.
        """

        pay_type = PayType(pay_type) if not isinstance(pay_type, PayType) else pay_type
        if priority is not None and (priority < 1 or priority > 10):
            raise ValueError("Priority must be between '1' and '10' if specified.")
        if vgpu_allocation is not None and vgpu_allocation not in VALID_VGPU_ALLOCATIONS:
            raise ValueError(
                f"vgpu_allocation must be one of {list(VALID_VGPU_ALLOCATIONS)} if specified."
            )
        if ignore_memory_limit is not None:
            raise NotImplementedError(
                "The 'ignore_memory_limit' argument is not supported for batch tasks; remove it "
                "before starting the batch."
            )

        if protocol_version is None:
            protocol_version = _get_protocol_version()
        payload = {
            "solverVersion": solver_version,
            "protocolVersion": protocol_version,
            "workerGroup": worker_group,
        }
        if pay_type != PayType.AUTO:
            payload["payType"] = pay_type.value
        if priority is not None:
            payload["priority"] = priority
        if vgpu_allocation is not None:
            payload["vgpuAllocation"] = vgpu_allocation
        serialized_additional_payload = _serialize_additional_payload(additional_payload)
        if serialized_additional_payload is not None:
            payload["additionalPayload"] = serialized_additional_payload

        return http.post(
            f"rf/task/{self.task_id}/submit",
            payload,
        )

    def abort(self) -> requests.Response:
        """Abort the current task on the server."""
        if not self.task_id:
            raise ValueError("Batch id not found.")
        return http.put(f"rf/task/{self.task_id}/abort", {})


class TaskFactory:
    """Factory for obtaining the correct task subclass."""

    _REGISTRY: dict[str, type[WebTask]] = {}

    @classmethod
    def reset(cls) -> None:
        """Clear the cached task kind registry (used in tests)."""
        cls._REGISTRY.clear()

    @classmethod
    def register(cls, task_id: str, kind: type[WebTask]) -> None:
        cls._REGISTRY[task_id] = kind

    @classmethod
    def get_kind(cls, task_id: str, verbose: bool = True) -> type[WebTask]:
        """Return cached task class, fetching and caching if needed."""
        kind = cls._REGISTRY.get(task_id)
        if kind:
            return kind
        if WebTask.is_batch(task_id):
            cls.register(task_id, BatchTask)
            return BatchTask
        task = SimulationTask.get(task_id, verbose=verbose)
        if task:
            cls.register(task_id, SimulationTask)
        return SimulationTask

    @classmethod
    def get(cls, task_id: str, verbose: bool = True) -> WebTask | None:
        kind = cls._REGISTRY.get(task_id)
        if kind is BatchTask:
            return BatchTask.get(task_id, verbose=verbose)
        if kind is SimulationTask:
            task = SimulationTask.get(task_id, verbose=verbose)
            return task
        if WebTask.is_batch(task_id):
            cls.register(task_id, BatchTask)
            return BatchTask.get(task_id, verbose=verbose)
        task = SimulationTask.get(task_id, verbose=verbose)
        if task:
            cls.register(task_id, SimulationTask)
        return task
