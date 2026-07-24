"""Atomic task operations used by higher-level workflow containers."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from tidy3d.components.medium import AbstractCustomMedium
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation, TCADAnalysisTypes
from tidy3d.components.workflow import resolve_workflow
from tidy3d.config import config
from tidy3d.exceptions import DataError, WebError, format_chained_exception_message
from tidy3d.log import get_logging_console, log
from tidy3d.web.api.states import (
    ALL_POST_VALIDATE_STATES,
    COMPLETED_PERCENT,
    DIVERGED_STATES,
    END_STATES,
    ERROR_STATES,
    MAX_STEPS,
    PRE_VALIDATE_STATES,
    STATE_PROGRESS_PERCENTAGE,
    status_to_stage,
)
from tidy3d.web.cache import resolve_local_cache
from tidy3d.web.core.constants import (
    CM_DATA_HDF5_GZ,
    MODE_DATA_HDF5_GZ,
    MODE_FILE_HDF5_GZ,
    MODELER_FILE_HDF5_GZ,
    SIM_FILE_HDF5,
    SIM_FILE_HDF5_GZ,
    SIMULATION_DATA_HDF5_GZ,
)
from tidy3d.web.core.s3utils import upload_file
from tidy3d.web.core.task_core import BatchTask, Folder, SimulationTask, TaskFactory, WebTask
from tidy3d.web.core.task_info import ChargeType, TaskInfo
from tidy3d.web.core.types import TaskType

from .connect_util import REFRESH_TIME, get_grid_points_str, get_time_steps_str, wait_for_connection
from .run_options import (
    log_deprecated_run_args,
    resolve_pay_type,
    resolve_run_start_options,
    resolve_upload_options,
    resolve_vgpu_start_options,
)
from .tidy3d_stub import Tidy3dStub, Tidy3dStubData, task_type_name_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from os import PathLike
    from typing import Literal

    from tidy3d.components.base import Tidy3dBaseModel
    from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType
    from tidy3d.web.cache import CacheEntry
    from tidy3d.web.core.constants import TaskId
    from tidy3d.web.core.task_info import BatchDetail
    from tidy3d.web.core.types import PayType


RUN_REFRESH_TIME = 1.0
SIM_FILE_JSON = "simulation.json"

GUI_SUPPORTED_TASK_TYPES = ["FDTD", "MODE_SOLVER", "HEAT", "TERMINAL_CM"]
BETA_TASK_TYPES = ["HEAT", "EME", "HEAT_CHARGE", "VOLUME_MESH"]
SOLVER_NAME = {
    "FDTD": "FDTD",
    "MODE_SOLVER": "Mode",
    "MODE": "Mode",
    "EME": "EME",
    "HEAT": "Heat",
    "HEAT_CHARGE": "HeatCharge",
    "VOLUME_MESH": "VolumeMesher",
}


@dataclass(frozen=True)
class FlexCreditEstimate:
    """Estimated task cost in FlexCredits."""

    maximum: float
    typical: float | None = None
    task_type: str | None = None
    is_final_billed_cost: bool | None = None
    typical_cost_kind: str | None = None


_FINAL_BILLED_COST_TASK_TYPES = {"MODE", "MODE_SOLVER", "EME", "HEAT"}
_TYPICAL_COST_KIND_CHARGE_SOLVER_ITERATIONS = "charge_solver_iterations"


DEFAULT_DATA_FILENAME = {
    TaskType.FDTD.name: "simulation_data.hdf5",
    TaskType.MODE_SOLVER.name: "simulation_data.hdf5",
    TaskType.MODE.name: "simulation_data.hdf5",
    TaskType.EME.name: "simulation_data.hdf5",
    TaskType.HEAT.name: "simulation_data.hdf5",
    TaskType.HEAT_CHARGE.name: "simulation_data.hdf5",
    TaskType.VOLUME_MESH.name: "simulation_data.hdf5",
    TaskType.MODAL_CM.name: "cm_data.hdf5",
    TaskType.TERMINAL_CM.name: "cm_data.hdf5",
    "COMPONENT_MODELER": "cm_data.hdf5",
    "TERMINAL_COMPONENT_MODELER": "cm_data.hdf5",
    "RF": "cm_data.hdf5",
}


def default_data_filename(task_type: str | None) -> str:
    """Return the default results filename for the given task type."""
    if isinstance(task_type, TaskType):
        task_type = task_type.name
    return DEFAULT_DATA_FILENAME.get(task_type or "", "simulation_data.hdf5")


def _estimate_is_final_billed_cost(estimate: FlexCreditEstimate) -> bool:
    """Return ``True`` when the estimate equals the final billed solver cost."""
    if estimate.is_final_billed_cost is not None:
        return estimate.is_final_billed_cost
    task_type = (estimate.task_type or "").upper()
    return task_type in _FINAL_BILLED_COST_TASK_TYPES


def _estimate_has_charge_solver_iteration_scaling(estimate: FlexCreditEstimate) -> bool:
    """Return whether ``estimate.typical`` comes from charge iteration scaling."""
    return (
        estimate.typical is not None
        and not _estimate_is_final_billed_cost(estimate)
        and estimate.typical_cost_kind == _TYPICAL_COST_KIND_CHARGE_SOLVER_ITERATIONS
    )


def _operation_estimate_is_final_billed_cost(operation: Any) -> bool | None:
    """Return local operation knowledge about whether an estimate is final."""
    if isinstance(operation, HeatChargeSimulation):
        return TCADAnalysisTypes.CHARGE not in operation._get_simulation_types()
    try:
        task_type = task_type_name_of(operation)
    except TypeError:
        return None
    return task_type.upper() in _FINAL_BILLED_COST_TASK_TYPES


def _batch_typical_flex_credit_cost(estimates: Iterable[FlexCreditEstimate]) -> float | None:
    """Return a batch typical cost when it can be reported as a complete batch total."""
    estimates = list(estimates)

    batch_typical_cost = 0.0
    has_typical_estimate = False
    for estimate in estimates:
        if _estimate_is_final_billed_cost(estimate):
            batch_typical_cost += estimate.maximum
        elif estimate.typical is not None:
            has_typical_estimate = True
            batch_typical_cost += estimate.typical
        elif estimate.maximum == 0:
            continue
        else:
            return None
    if not has_typical_estimate:
        return None
    return batch_typical_cost


def _resolve_output_path(path: PathLike | None, task_type: str | None) -> Path:
    """Resolve an explicit output path or a task-type-specific default filename."""
    return Path(path) if path is not None else Path(default_data_filename(task_type))


def _is_web_container(value: object) -> bool:
    """Return whether a value is a public web container shape."""
    return isinstance(value, list | tuple | Mapping)


def _raise_if_upload_container(simulation: object) -> None:
    """Raise a clear error for multi-workflow upload containers."""
    if not _is_web_container(simulation):
        return

    raise ValueError(
        "tidy3d.web.upload() accepts a single workflow object, but received a "
        f"{type(simulation).__name__} container. For multiple workflows, use "
        "tidy3d.web.Batch(simulations=...) and call batch.upload(); use "
        "tidy3d.web.run(...) if you want the full upload/start/monitor/load flow "
        "for a container."
    )


def _raise_if_load_container(task_id: object) -> None:
    """Raise a clear error for task-id load containers."""
    if not _is_web_container(task_id):
        return

    raise ValueError(
        "tidy3d.web.load() accepts a single task id, or None to load an existing local "
        f"file, but received a {type(task_id).__name__} container. Loading task-id "
        "containers is not supported by this function because multi-task state is "
        "managed by Batch. Use Batch.load(...) for batches, or call "
        "tidy3d.web.load(...) once per task id."
    )


def _task_type_from_task(task: WebTask, task_type: str | None = None) -> str | None:
    """Normalize the task type for default-path and artifact resolution."""
    task_type = task_type or getattr(task, "task_type", None)
    if isinstance(task_type, TaskType):
        task_type = task_type.name
    if isinstance(task, BatchTask):
        return task_type or "RF"
    return task_type


def _resolve_download_target(
    task_id: TaskId,
    path: PathLike | None,
    task: WebTask | None = None,
    task_type: str | None = None,
) -> tuple[Path, WebTask, str | None]:
    """Resolve the output path and task metadata needed to download results."""
    if task is None:
        task = TaskFactory.get(task_id, verbose=False)
        if task is None:
            raise ValueError("Task not found.")
    task_type = _task_type_from_task(task, task_type)
    return _resolve_output_path(path, task_type), task, task_type


def _remote_data_file(task: WebTask, task_type: str | None) -> str:
    """Return the remote results artifact name for a task."""
    if isinstance(task, BatchTask):
        return CM_DATA_HDF5_GZ
    return MODE_DATA_HDF5_GZ if task_type == TaskType.MODE_SOLVER.name else SIMULATION_DATA_HDF5_GZ


def _raise_if_modeler_batch_diverged(task: WebTask) -> None:
    """Raise before downloading aggregate modeler data for diverged RF batches."""
    if isinstance(task, BatchTask) and (task.status or "").lower() in DIVERGED_STATES:
        raise WebError(
            "The RF/modeler task diverged, so aggregate component-modeler data "
            "was not produced. Child simulation data may still be available on "
            "the child task IDs."
        )


def _build_website_url(path: str) -> str:
    base = str(config.web.website_endpoint or "")
    if not path:
        return base
    return "/".join([base.rstrip("/"), path.lstrip("/")])


def _get_url(task_id: str) -> str:
    """Get the URL for a task on our server."""
    return _build_website_url(f"workbench?taskId={task_id}")


def _get_folder_url(folder_id: str) -> str:
    """Get the URL for a task folder on our server."""
    return _build_website_url(f"folders/{folder_id}")


def _get_url_rf(resource_id: str) -> str:
    """Get the RF GUI URL for a modeler/batch group."""
    return _build_website_url(f"rf?taskId={resource_id}")


def _get_task_urls(
    task_type: str,
    resource_id: str,
    folder_id: str | None = None,
    group_id: str | None = None,
) -> tuple[str, str | None]:
    """Log task and folder links to the web UI."""
    if task_type in ["RF", "TERMINAL_CM", "MODAL_CM"]:
        url = _get_url_rf(group_id or resource_id)
    else:
        url = _get_url(resource_id)

    if folder_id is not None:
        folder_url = _get_folder_url(folder_id)
    else:
        folder_url = None
    return url, folder_url


def _batch_detail_error(resource_id: str) -> WebError | None:
    """Processes a failed batch job to generate a detailed error."""
    try:
        batch = BatchTask.get(resource_id)
        batch_detail = batch.detail()
        status = batch_detail.status.lower()
    except Exception as e:
        log.error(f"Could not retrieve batch details for '{resource_id}': {e}")
        raise WebError(
            format_chained_exception_message(
                f"Failed to retrieve status for batch '{resource_id}'", e
            )
        ) from e

    if status not in ERROR_STATES:
        return None

    if hasattr(batch_detail, "validateErrors") and batch_detail.validateErrors:
        try:
            error_details = []
            for key, error_str in batch_detail.validateErrors.items():
                error_details.append(f"- Subtask '{key}' failed: {error_str}")

            details_string = "\n".join(error_details)
            full_error_msg = (
                "One or more subtasks failed validation. Please fix the component modeler "
                "configuration.\n"
                f"Details:\n{details_string}"
            )
        except Exception as e:
            raise WebError(
                format_chained_exception_message(
                    "One or more subtasks failed validation. Failed to parse validation errors.",
                    e,
                )
            ) from e
        raise WebError(full_error_msg)

    raise WebError(
        f"Batch '{resource_id}' failed with status '{status}'. Check server logs for details or "
        "contact customer support."
    )


def _copy_simulation_data_from_cache_entry(entry: CacheEntry, path: PathLike) -> bool:
    """Copy cached simulation data from a cache entry to a specified path."""
    if entry is not None:
        try:
            entry.materialize(Path(path))
            return True
        except Exception:
            return False
    return False


def _load_simulation_via_tempfile(task_id: TaskId) -> WorkflowOperationType | None:
    """Load a simulation into a temp file for cache bookkeeping (Windows-safe)."""
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        task = TaskFactory.get(task_id, verbose=False)
        if isinstance(task, BatchTask):
            raise NotImplementedError("Operation not implemented for modeler batches.")
        path = Path(fname)
        task.get_simulation_hdf5(path, verbose=False)
        return Tidy3dStub.from_file(path)
    finally:
        try:
            os.unlink(fname)
        except FileNotFoundError:
            pass


def _cache_simulation_for_load(
    *,
    task_id: TaskId,
    workflow_type: str | None,
    lazy: bool,
    cache_simulation: WorkflowOperationType | None,
) -> tuple[WorkflowOperationType | None, bool]:
    """Return the simulation object to use for cache storage, and whether to store."""
    if cache_simulation is not None:
        return cache_simulation, True

    if workflow_type == TaskType.VOLUME_MESH.name:
        try:
            return _load_simulation_via_tempfile(task_id), True
        except Exception as e:
            log.info(f"Failed to load VolumeMesher for storing results: {e}.")
            return None, False

    if lazy:
        try:
            return _load_simulation_via_tempfile(task_id), True
        except Exception as e:
            log.info(f"Failed to load simulation for storing results: {e}.")
            return None, False

    return None, True


def get_reduced_simulation(
    simulation: WorkflowOperationType,
    reduce_simulation: Literal["auto", True, False],
    *,
    warn_auto: bool = True,
) -> WorkflowOperationType:
    """
    Adjust the given simulation object based on the reduce_simulation parameter. Currently only
    implemented for the mode solver.
    """
    if reduce_simulation == "auto":
        if isinstance(simulation, ModeSimulation):
            sim_mediums = simulation.scene.mediums
        else:
            sim_mediums = simulation.simulation.scene.mediums
        contains_custom = any(isinstance(med, AbstractCustomMedium) for med in sim_mediums)
        reduce_simulation = contains_custom

        if reduce_simulation and warn_auto:
            log.warning(
                f"The {type(simulation)} object contains custom mediums. It will be "
                "automatically restricted to the solver domain to reduce data for uploading. "
                "To force uploading the original object use 'reduce_simulation=False'."
                " Setting 'reduce_simulation=True' will force simulation reduction in all cases and"
                " silence this warning."
            )
    if reduce_simulation:
        return simulation.reduced_simulation_copy
    return simulation


@wait_for_connection
def restore_simulation_if_cached(
    simulation: WorkflowOperationType,
    path: PathLike | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    verbose: bool = True,
) -> tuple[PathLike | None, TaskId | None]:
    """Attempt to restore simulation data from a local cache entry, if available."""
    simulation_cache = resolve_local_cache()
    retrieved_simulation_path = None
    cached_task_id = None
    if simulation_cache is not None:
        sim_for_cache = simulation
        if isinstance(simulation, ModeSolver | ModeSimulation):
            sim_for_cache = get_reduced_simulation(simulation, reduce_simulation)
        entry = simulation_cache.try_fetch(simulation=sim_for_cache, verbose=verbose)
        if entry is not None:
            if path is not None:
                copied = _copy_simulation_data_from_cache_entry(entry, path)
                if copied:
                    retrieved_simulation_path = path
            else:
                retrieved_simulation_path = entry.artifact_path
            cached_task_id = entry.metadata.get("task_id")
            cached_workflow_type = entry.metadata.get("workflow_type")
            if cached_task_id is not None and cached_workflow_type is not None and verbose:
                console = get_logging_console()
                url, _ = _get_task_urls(cached_workflow_type, cached_task_id)
                console.log(
                    "Loading simulation from local cache. "
                    f"View cached task using web UI at [link={url}]'{url}'[/link]."
                )
    return retrieved_simulation_path, cached_task_id


def load_simulation_if_cached(
    simulation: WorkflowOperationType,
    path: PathLike | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    verbose: bool = True,
) -> WorkflowDataType | None:
    """Load simulation results directly from the local cache, if available."""
    restored_path, _ = restore_simulation_if_cached(
        simulation=simulation,
        path=path,
        reduce_simulation=reduce_simulation,
        verbose=verbose,
    )
    if restored_path is None:
        return None

    data = load(
        task_id=None,
        path=str(restored_path),
        verbose=verbose,
    )
    if isinstance(simulation, ModeSolver):
        simulation._patch_data(data=data)
    return data


def _upload_sidecar_artifacts(
    resource_id: TaskId,
    sidecar_artifacts: Mapping[str, Tidy3dBaseModel],
    verbose: bool,
) -> None:
    """Serialize and upload internal sidecar artifacts for an allocated task."""
    for remote_filename, artifact in sidecar_artifacts.items():
        suffix = "".join(Path(remote_filename).suffixes) or ".hdf5"
        handle, fname = tempfile.mkstemp(suffix=suffix)
        os.close(handle)
        try:
            artifact.to_file(fname)
            upload_file(
                resource_id,
                fname,
                remote_filename,
                verbose=verbose,
            )
        finally:
            os.unlink(fname)


def upload(
    simulation: WorkflowOperationType,
    task_name: str | None = None,
    folder_name: str = "default",
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    source_required: bool = True,
    solver_version: str | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    verbose_estimate_cost: bool | None = None,
    _workflow_step: bool = False,
) -> TaskId:
    return _upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback=progress_callback,
        simulation_type=simulation_type,
        parent_tasks=parent_tasks,
        source_required=source_required,
        solver_version=solver_version,
        reduce_simulation=reduce_simulation,
        verbose_estimate_cost=verbose_estimate_cost,
        _workflow_step=_workflow_step,
    )


@wait_for_connection
def _upload(
    simulation: WorkflowOperationType,
    task_name: str | None = None,
    folder_name: str = "default",
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    source_required: bool = True,
    solver_version: str | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    verbose_estimate_cost: bool | None = None,
    _workflow_step: bool = False,
    _sidecar_artifacts: Mapping[str, Tidy3dBaseModel] | None = None,
) -> TaskId:
    """Private upload implementation with optional internal sidecar artifacts."""
    _raise_if_upload_container(simulation)

    if not _workflow_step:
        workflow = resolve_workflow(simulation)
        if len(workflow.steps) > 1:
            raise DataError(
                "'web.upload()' does not support multi-step simulations. "
                "Use 'web.run()', 'web.Job(...).run()', or 'web.Job(...).step()' "
                "to execute them."
            )

    console = get_logging_console() if verbose else None
    log_deprecated_run_args(
        solver_version=solver_version,
    )
    upload_options = resolve_upload_options(
        solver_version=solver_version,
        simulation_type=simulation_type,
    )

    if isinstance(simulation, ModeSolver | ModeSimulation):
        simulation = get_reduced_simulation(simulation, reduce_simulation)

    stub = Tidy3dStub(simulation=simulation)
    stub.validate_pre_upload(source_required=source_required)
    log.debug("Creating task.")

    if task_name is None:
        task_name = stub.get_default_task_name()

    task_type = stub.get_type()

    task = WebTask.create(
        task_type,
        task_name,
        folder_name,
        callback_url,
        upload_options.simulation_type,
        parent_tasks,
        "Gz",
    )

    group_id = getattr(task, "groupId", None)
    resource_id = task.task_id

    if verbose:
        console.log(
            f"Created task '{task_name}' with resource_id '{resource_id}' and task_type '{task_type}'."
        )
        if task_type in BETA_TASK_TYPES:
            solver_name = SOLVER_NAME[task_type]
            console.log(
                f"Tidy3D's {solver_name} solver is currently in the beta stage. "
                f"Cost of {solver_name} simulations is subject to change in the future."
            )
        if task_type in GUI_SUPPORTED_TASK_TYPES:
            url, folder_url = _get_task_urls(task_type, resource_id, task.folder_id, group_id)
            console.log(f"View task using web UI at [link={url}]'{url}'[/link].")
            console.log(f"Task folder: [link={folder_url}]'{task.folder_name}'[/link].")

    remote_sim_file = SIM_FILE_HDF5_GZ
    if task_type == "MODE_SOLVER":
        remote_sim_file = MODE_FILE_HDF5_GZ
    elif task_type in ["RF", "TERMINAL_CM", "MODAL_CM"]:
        remote_sim_file = MODELER_FILE_HDF5_GZ

    task.upload_simulation(
        stub=stub,
        verbose=verbose,
        progress_callback=progress_callback,
        remote_sim_file=remote_sim_file,
    )
    if _sidecar_artifacts is not None:
        _upload_sidecar_artifacts(resource_id, _sidecar_artifacts, verbose=verbose)

    verbose_estimate_cost = verbose if verbose_estimate_cost is None else verbose_estimate_cost
    estimate_cost(
        task_id=resource_id,
        solver_version=upload_options.solver_version,
        verbose=verbose_estimate_cost,
    )

    task.validate_post_upload(parent_tasks=parent_tasks)

    return resource_id


@wait_for_connection
def get_info(task_id: TaskId, verbose: bool = True) -> TaskInfo | BatchDetail:
    task = TaskFactory.get(task_id, verbose=verbose)
    if not task:
        raise ValueError("Task not found.")
    return task.detail()


@wait_for_connection
def start(
    task_id: TaskId,
    solver_version: str | None = None,
    worker_group: str | None = None,
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> None:
    task = TaskFactory.get(task_id)
    if not task:
        raise ValueError("Task not found.")
    log_deprecated_run_args(
        solver_version=solver_version,
        worker_group=worker_group,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )
    dispatch_options = resolve_run_start_options(
        solver_version=solver_version,
        worker_group=worker_group,
    )
    if isinstance(task, BatchTask):
        resolved_pay_type = resolve_pay_type(pay_type, apply_config_default=False)
        resolved_priority = priority
        resolved_vgpu_allocation = vgpu_allocation
        resolved_ignore_memory_limit = ignore_memory_limit
    else:
        resolved_pay_type = resolve_pay_type(pay_type)
        vgpu_options = resolve_vgpu_start_options(
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            apply_config_defaults=True,
        )
        resolved_priority = vgpu_options.priority
        resolved_vgpu_allocation = vgpu_options.vgpu_allocation
        resolved_ignore_memory_limit = vgpu_options.ignore_memory_limit
    task.submit(
        solver_version=dispatch_options.solver_version,
        worker_group=dispatch_options.worker_group,
        pay_type=resolved_pay_type,
        priority=resolved_priority,
        vgpu_allocation=resolved_vgpu_allocation,
        ignore_memory_limit=resolved_ignore_memory_limit,
        additional_payload=dispatch_options.additional_payload,
    )


@wait_for_connection
def get_run_info(task_id: TaskId) -> tuple[float | None, float | None]:
    task = TaskFactory.get(task_id)
    if isinstance(task, BatchTask):
        raise NotImplementedError("Operation not implemented for modeler batches.")
    return task.get_running_info()


def _get_batch_detail_handle_error_status(batch: BatchTask) -> BatchDetail:
    """Get batch detail and raise error if status is in ERROR_STATES."""
    detail = batch.detail()
    status = detail.status.lower()
    if status in ERROR_STATES:
        _batch_detail_error(batch.task_id)
    return detail


def get_status(task_id: TaskId) -> str:
    """Get the status of a task. Raises an error if status is ``error``."""
    task = TaskFactory.get(task_id)
    if isinstance(task, BatchTask):
        return _get_batch_detail_handle_error_status(task).status

    task_info = get_info(task_id)
    status = task_info.status
    if status == "visualize":
        return "success"
    if status in ERROR_STATES:
        try:
            task = SimulationTask(taskId=task_id)
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
                task.get_error_json(to_file=tmp_file.name)
                with open(tmp_file.name) as f:
                    error_content = json.load(f)
                    error_msg = error_content["msg"]
        except Exception:
            error_msg = "Error message could not be obtained, please contact customer support."

        raise WebError(f"Error running task {task_id}! {error_msg}")
    return status


def _batch_detail_progress(detail: BatchDetail) -> tuple[str, str, float]:
    """Compute display status, color status, and progress percentage from BatchDetail subtasks."""
    batch_status = (detail.status or "draft").lower()

    if not detail.tasks:
        return batch_status, batch_status, STATE_PROGRESS_PERCENTAGE.get(batch_status, 0)

    if batch_status in END_STATES:
        pct = STATE_PROGRESS_PERCENTAGE.get(batch_status, COMPLETED_PERCENT)
        return batch_status, batch_status, pct

    n_tasks = len(detail.tasks)
    stage_acc = 0.0
    status_counts: dict[str, int] = {}
    for task in detail.tasks:
        task_status = (task.status or "draft").lower()
        stage_name, idx = status_to_stage(task_status)
        stage_acc += idx / MAX_STEPS
        status_counts[stage_name] = status_counts.get(stage_name, 0) + 1

    task_avg = stage_acc / n_tasks
    pct = task_avg * 0.8 * COMPLETED_PERCENT

    dominant_stage = max(status_counts, key=status_counts.get)
    dominant_count = status_counts[dominant_stage]

    if n_tasks > 1 and dominant_count < n_tasks:
        display_status = f"{dominant_stage} ({dominant_count}/{n_tasks})"
    else:
        display_status = dominant_stage

    return display_status, dominant_stage, pct


def _monitor_modeler_batch(
    task_id: str,
    verbose: bool = True,
    max_detail_tasks: int = 20,
) -> None:
    """Monitor modeler batch progress with aggregate and per-task views."""
    console = get_logging_console() if verbose else None
    task = BatchTask.get(task_id=task_id)
    detail = _get_batch_detail_handle_error_status(task)
    name = detail.name or "modeler_batch"
    group_id = detail.groupId
    status = detail.status.lower()

    if not verbose:
        while status_to_stage(status)[0] not in END_STATES:
            time.sleep(REFRESH_TIME)
            detail = _get_batch_detail_handle_error_status(task)
            status = detail.status.lower()
        return

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=25),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.fields[status]}"),
        TimeElapsedColumn(),
    )
    header = f"Subtasks status - {name}"
    if group_id:
        header += f"\nGroup ID: '{group_id}'"
    console.log(header)
    with Progress(*progress_columns, console=console, transient=False) as progress:
        display_status, _, _ = _batch_detail_progress(detail)
        p_run = progress.add_task("Run Total", total=1.0, status=f" {display_status} ")
        task_bars: dict[str, int] = {}
        prev_display_status = display_status
        console.log(f"Batch status = {status}")

        end_monitor = False
        while not end_monitor:
            total = len(detail.tasks)
            completed_runs = detail.runSuccess or 0
            display_status, _, _ = _batch_detail_progress(detail)
            if display_status != prev_display_status:
                prev_display_status = display_status
                console.log(f"Batch status = {display_status}")
                progress.update(p_run, status=f" {display_status} ")

            if total and total <= max_detail_tasks and detail.tasks:
                name_to_task = {(task.taskName or task.taskId): task for task in detail.tasks or []}
                for name, batch_task in name_to_task.items():
                    if name not in task_bars:
                        task_status = (batch_task.status or "draft").lower()
                        pbar = progress.add_task(
                            f"  {name}",
                            total=1.0,
                            completed=STATE_PROGRESS_PERCENTAGE[task_status] / 100,
                            status=f" {task_status} ",
                        )
                        task_bars[name] = pbar

            if detail.tasks:
                acc = 0.0
                n_members = 0
                for batch_task in detail.tasks or []:
                    n_members += 1
                    task_status = (batch_task.status or "draft").lower()
                    _, idx = status_to_stage(task_status)
                    acc += max(0.0, min(1.0, idx / MAX_STEPS))
                run_frac = ((acc / float(n_members)) if n_members else 0.0) * 0.8
            else:
                run_frac = (completed_runs / total) * 0.8 if total else 0.0

            if status in END_STATES:
                end_monitor = True
                run_frac = 1.0

            progress.update(p_run, completed=run_frac)

            if task_bars and detail.tasks:
                name_to_task = {(task.taskName or task.taskId): task for task in detail.tasks or []}
                for task_name, pbar in task_bars.items():
                    batch_task = name_to_task.get(task_name)
                    if not batch_task:
                        continue
                    task_status = (batch_task.status or "draft").lower()
                    progress.update(
                        pbar,
                        completed=STATE_PROGRESS_PERCENTAGE[task_status] / 100,
                        description=f"  {task_name}",
                        status=f" {task_status} ",
                        refresh=False,
                    )

            progress.refresh()
            time.sleep(REFRESH_TIME)
            detail = _get_batch_detail_handle_error_status(task)
            status = detail.status.lower()

        console.log("Modeler has finished running successfully.")
        real_cost(task.task_id, verbose=verbose)


@wait_for_connection
def monitor(task_id: TaskId, verbose: bool = True, worker_group: str | None = None) -> None:
    """Print the real time task progress until completion."""
    del worker_group

    task_kind = TaskFactory.get_kind(task_id)
    if task_kind is BatchTask:
        return _monitor_modeler_batch(task_id, verbose=verbose)

    console = get_logging_console() if verbose else None
    task_info = get_info(task_id)
    task_name = task_info.taskName
    task_type = task_info.taskType

    def get_estimated_cost() -> float:
        task_info_local = get_info(task_id)
        block_info = task_info_local.taskBlockInfo
        if block_info and block_info.chargeType == ChargeType.FREE:
            est_flex_unit = 0
            grid_points = block_info.maxGridPoints
            time_steps = block_info.maxTimeSteps
            grid_points_str = get_grid_points_str(grid_points)
            time_steps_str = get_time_steps_str(time_steps)
            console.log(
                f"You are running this simulation for FREE. Your current plan allows"
                f" up to {block_info.maxFreeCount} free non-concurrent simulations per"
                f" day (under {grid_points_str} grid points and {time_steps_str}"
                f" time steps)"
            )
        else:
            est_flex_unit = task_info_local.estFlexUnit
        return est_flex_unit

    def monitor_preprocess() -> None:
        status_local = get_status(task_id)
        while status_local not in END_STATES and status_local != "running":
            new_status = get_status(task_id)
            if new_status != status_local:
                status_local = new_status
                if verbose and status_local != "running":
                    console.log(f"status = {status_local}")
            time.sleep(REFRESH_TIME)

    status = get_status(task_id)

    if verbose:
        console.log(f"status = {status}")

    if status in END_STATES:
        return None

    if verbose:
        console.log(
            "To cancel the simulation, use 'web.abort(task_id)' or 'web.delete(task_id)' "
            "or abort/delete the task in the web "
            "UI. Terminating the Python script will not stop the job running on the cloud."
        )
        with console.status(f"[bold green]Waiting for '{task_name}'...", spinner="runner"):
            monitor_preprocess()
    else:
        monitor_preprocess()

    if verbose:
        get_estimated_cost()
        console.log("starting up solver")

    while get_run_info(task_id)[0] is None and get_status(task_id) == "running":
        time.sleep(REFRESH_TIME)

    if verbose:
        console.log("running solver")
        if "FDTD" in task_type:
            with Progress(console=console) as progress:
                pbar_pd = progress.add_task("% done", total=100)
                perc_done, _ = get_run_info(task_id)

                while (
                    perc_done is not None and perc_done < 100 and get_status(task_id) == "running"
                ):
                    perc_done, field_decay = get_run_info(task_id)
                    progress.update(
                        pbar_pd,
                        completed=perc_done,
                        description=f"solver progress (field decay = {field_decay:.2e})",
                    )
                    time.sleep(RUN_REFRESH_TIME)

                perc_done, field_decay = get_run_info(task_id)
                if perc_done is not None and perc_done < 100 and field_decay > 0:
                    console.log(f"early shutoff detected at {perc_done:1.0f}%, exiting.")

                progress.update(
                    pbar_pd,
                    completed=100,
                    refresh=True,
                    description=f"solver progress (field decay = {field_decay:.2e})",
                )
        elif task_type == "EME":
            with Progress(console=console) as progress:
                pbar_pd = progress.add_task("% done", total=100)
                perc_done, _ = get_run_info(task_id)

                while (
                    perc_done is not None and perc_done < 100 and get_status(task_id) == "running"
                ):
                    perc_done, _ = get_run_info(task_id)
                    progress.update(pbar_pd, completed=perc_done, description="solver progress")
                    time.sleep(RUN_REFRESH_TIME)

                progress.update(pbar_pd, completed=100, refresh=True, description="solver progress")
        else:
            while get_status(task_id) == "running":
                _ = get_run_info(task_id)
                time.sleep(RUN_REFRESH_TIME)
    else:
        perc_done, _ = get_run_info(task_id)
        while perc_done is not None and perc_done < 100 and get_status(task_id) == "running":
            perc_done, _ = get_run_info(task_id)
            time.sleep(RUN_REFRESH_TIME)

    if verbose:
        status = get_status(task_id)
        if status != "running":
            console.log(f"status = {status}")

        with console.status(f"[bold green]Finishing '{task_name}'...", spinner="runner"):
            while status not in END_STATES:
                new_status = get_status(task_id)
                if new_status != status:
                    status = new_status
                    console.log(f"status = {status}")
                time.sleep(REFRESH_TIME)

        if task_type in GUI_SUPPORTED_TASK_TYPES:
            url = _get_url(task_id)
            console.log(f"View simulation result at [blue underline][link={url}]'{url}'[/link].")
    else:
        while get_status(task_id) not in END_STATES:
            time.sleep(REFRESH_TIME)

    return None


@wait_for_connection
def download(
    task_id: TaskId,
    path: PathLike | None = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Download results of task to file."""
    path, task, task_type = _resolve_download_target(task_id=task_id, path=path)
    _raise_if_modeler_batch_diverged(task)
    remote_data_file = _remote_data_file(task, task_type)
    task.get_data_hdf5(
        to_file=path,
        remote_data_file_gz=remote_data_file,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    return None


@wait_for_connection
def load(
    task_id: TaskId | None,
    path: PathLike | None = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    replace_existing: bool = True,
    lazy: bool = False,
    cache_simulation: WorkflowOperationType | None = None,
    store_in_cache: bool = True,
    _allow_existing_path_with_task_id: bool = False,
) -> WorkflowDataType:
    """Download and load simulation results into a data object."""
    _raise_if_load_container(task_id)

    from_cache = task_id is None
    task = None
    task_type = None
    path = Path(path) if path is not None else None
    reuse_existing_path = path is not None and path.exists() and not replace_existing
    if from_cache:
        path = _resolve_output_path(path, None)
    elif reuse_existing_path and _allow_existing_path_with_task_id:
        pass
    else:
        path, task, task_type = _resolve_download_target(task_id=task_id, path=path)
        reuse_existing_path = path.exists() and not replace_existing

    if from_cache:
        if not path.exists():
            raise FileNotFoundError("Cached file not found.")
    elif not path.exists() or replace_existing:
        if task is None or not hasattr(task, "get_data_hdf5"):
            download(
                task_id=task_id,
                path=path,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        else:
            remote_data_file = _remote_data_file(task, task_type)
            _raise_if_modeler_batch_diverged(task)
            task.get_data_hdf5(
                to_file=path,
                remote_data_file_gz=remote_data_file,
                verbose=verbose,
                progress_callback=progress_callback,
            )

    if verbose and not from_cache:
        console = get_logging_console()
        console.log(f"Loading results from {path}")

    stub_data = Tidy3dStubData.postprocess(path, lazy=lazy)

    simulation_cache = resolve_local_cache()
    should_store_in_cache = store_in_cache and simulation_cache is not None and not from_cache
    if should_store_in_cache and (not reuse_existing_path or cache_simulation is not None):
        if reuse_existing_path:
            workflow_type = task_type_name_of(cache_simulation)
        else:
            info = get_info(task_id, verbose=False)
            workflow_type = getattr(info, "taskType", None)
        if workflow_type != TaskType.MODE_SOLVER.name:
            simulation, should_store = _cache_simulation_for_load(
                task_id=task_id,
                workflow_type=workflow_type,
                lazy=lazy,
                cache_simulation=cache_simulation,
            )
            if should_store:
                simulation_cache.store_result(
                    stub_data=stub_data,
                    task_id=task_id,
                    path=path,
                    workflow_type=workflow_type,
                    simulation=simulation,
                )

    return stub_data


@wait_for_connection
def abort(task_id: TaskId) -> TaskInfo | None:
    """Abort server-side data associated with task."""
    console = get_logging_console()

    task = TaskFactory.get(task_id, verbose=False)
    if not task:
        return None
    url = task.get_url()
    task.abort()
    console.log(
        f"Task is aborting. View task using web UI at [link={url}]'{url}'[/link] to check the result."
    )
    return TaskInfo(
        **{"taskId": task_id, "taskType": getattr(task, "task_type", None), **task.model_dump()}
    )


@wait_for_connection
def download_json(task_id: TaskId, path: PathLike = SIM_FILE_JSON, verbose: bool = True) -> None:
    """Download the ``.json`` simulation file associated with a task."""
    task = TaskFactory.get(task_id, verbose=False)
    if isinstance(task, BatchTask):
        raise NotImplementedError("Operation not implemented for modeler batches.")
    task.get_simulation_json(path, verbose=verbose)


@wait_for_connection
def delete_old(days_old: int, folder_name: str = "default") -> int:
    """Remove folder contents older than ``days_old``."""
    folder = Folder.get(folder_name, create=True)
    return folder.delete_old(days_old)


@wait_for_connection
def load_simulation(
    task_id: TaskId, path: PathLike = SIM_FILE_JSON, verbose: bool = True
) -> WorkflowOperationType:
    """Download a task's simulation file and load the associated workflow object."""
    task = TaskFactory.get(task_id, verbose=False)
    if isinstance(task, BatchTask):
        raise NotImplementedError("Operation not implemented for modeler batches.")
    path = Path(path)
    if path.suffix == ".json":
        task.get_simulation_json(path, verbose=verbose)
    elif path.suffix == ".hdf5":
        task.get_simulation_hdf5(path, verbose=verbose)
    else:
        raise ValueError("Path suffix must be '.json' or '.hdf5'")
    return Tidy3dStub.from_file(path)


@wait_for_connection
def download_log(
    task_id: TaskId,
    path: PathLike = "tidy3d.log",
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Download the log file associated with a task."""
    task = TaskFactory.get(task_id, verbose=False)
    if isinstance(task, BatchTask):
        raise NotImplementedError("Operation not implemented for modeler batches.")
    task.get_log(path, verbose=verbose, progress_callback=progress_callback)


@wait_for_connection
def download_simulation(
    task_id: TaskId,
    path: PathLike = SIM_FILE_HDF5,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Download the ``.hdf5`` simulation file associated with a task."""
    task = TaskFactory.get(task_id, verbose=False)
    if isinstance(task, BatchTask):
        raise NotImplementedError("Operation not implemented for modeler batches.")

    task_type = _task_type_from_task(task)
    if task_type is None:
        info = get_info(task_id, verbose=False)
        task_type = getattr(info, "taskType", None)

    remote_sim_file = (
        MODE_FILE_HDF5_GZ if task_type == TaskType.MODE_SOLVER.name else SIM_FILE_HDF5_GZ
    )
    task.get_simulation_hdf5(
        path,
        verbose=verbose,
        progress_callback=progress_callback,
        remote_sim_file=remote_sim_file,
    )


@wait_for_connection
def get_tasks(
    num_tasks: int | None = None, order: Literal["new", "old"] = "new", folder: str = "default"
) -> list[dict]:
    """Get metadata of tasks in the requested folder."""
    folder = Folder.get(folder, create=True)
    tasks = folder.list_tasks()
    if not tasks:
        return []
    if order == "new":
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
    elif order == "old":
        tasks = sorted(tasks, key=lambda t: t.created_at)
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return [task.model_dump() for task in tasks]


@wait_for_connection
def delete(task_id: TaskId, versions: bool = False) -> TaskInfo:
    """Delete server-side data associated with task."""
    if not task_id:
        raise ValueError("Task id not found.")
    task = TaskFactory.get(task_id, verbose=False)
    task.delete(versions)
    return TaskInfo(**{"taskId": task.task_id, **task.model_dump()})


def _log_flex_credit_estimate(console: Any, estimate: FlexCreditEstimate) -> None:
    """Log a user-facing FlexCredit estimate."""
    task_type = (estimate.task_type or "").upper()
    if estimate.typical is not None:
        if _estimate_has_charge_solver_iteration_scaling(estimate):
            console.log(
                f"Estimated typical FlexCredit cost: {estimate.typical:1.3f}. "
                "For charge simulations, the billed cost depends on the number of solver "
                "iterations required for convergence."
            )
            console.log(
                f"Maximum FlexCredit cost: {estimate.maximum:1.3f}. This assumes the charge "
                "solver reaches its configured iteration limits for all applied biases. Use "
                "'web.real_cost(task_id)' to get the billed FlexCredit cost after a simulation "
                "run."
            )
        else:
            console.log(f"Estimated typical FlexCredit cost: {estimate.typical:1.3f}.")
            console.log(
                f"Maximum FlexCredit cost: {estimate.maximum:1.3f}. Use "
                "'web.real_cost(task_id)' to get the billed FlexCredit cost after a simulation "
                "run."
            )
        return

    if task_type in {"FDTD", "RF_FDTD"}:
        console.log(
            f"Estimated FlexCredit cost: {estimate.maximum:1.3f}. This assumes the FDTD "
            "solver runs for the full simulation time; if early shutoff is reached, the "
            "billed cost can be lower. Use 'web.real_cost(task_id)' to get the billed "
            "FlexCredit cost after a simulation run."
        )
    elif _estimate_is_final_billed_cost(estimate):
        console.log(
            f"Estimated FlexCredit cost: {estimate.maximum:1.3f}. For this solver type, "
            "the estimate is the final billed cost."
        )
    else:
        console.log(
            f"Estimated FlexCredit cost: {estimate.maximum:1.3f}. Use "
            "'web.real_cost(task_id)' to get the billed FlexCredit cost after a simulation "
            "run."
        )


@wait_for_connection
def estimate_cost_info(
    task_id: TaskId,
    verbose: bool = True,
    solver_version: str | None = None,
    *,
    is_final_billed_cost: bool | None = None,
) -> FlexCreditEstimate:
    """Compute the FlexCredit charge estimate details for a given task."""
    if not isinstance(task_id, str):
        raise ValueError(
            f"Task ID: {task_id} is not a string. You can get it using 'web.upload(<simulation>)'."
        )

    console = get_logging_console() if verbose else None

    task = TaskFactory.get(task_id, verbose=False)
    detail = task.detail()
    if isinstance(task, BatchTask):
        check_task_type = "FDTD" if detail.taskType == "MODAL_CM" else "RF_FDTD"
        status = detail.status.lower()
        if status in {"created", "draft"}:
            task.check(solver_version=solver_version, check_task_type=check_task_type)
        while status in PRE_VALIDATE_STATES:
            detail = task.detail()
            status = detail.status.lower()
            if status in PRE_VALIDATE_STATES:
                time.sleep(REFRESH_TIME)
        if status in ERROR_STATES:
            _batch_detail_error(resource_id=task_id)
        est_flex_unit = detail.estFlexUnit
        estimate = FlexCreditEstimate(
            maximum=est_flex_unit,
            task_type=detail.taskType,
            is_final_billed_cost=is_final_billed_cost,
        )
        if verbose:
            _log_flex_credit_estimate(console, estimate)
        return estimate

    task.estimate_cost(solver_version=solver_version)
    task_info = get_info(task_id)
    status = task_info.metadataStatus

    while status not in ALL_POST_VALIDATE_STATES:
        time.sleep(REFRESH_TIME)
        task_info = get_info(task_id)
        status = task_info.metadataStatus

    if status in ERROR_STATES:
        try:
            task = SimulationTask(taskId=task_id)
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
                task.get_error_json(to_file=tmp_file.name, validation=True)
                with open(tmp_file.name) as f:
                    error_content = json.load(f)
                    error_msg = error_content["validation_error"]
        except Exception:
            error_msg = "Error message could not be obtained, please contact customer support."
        raise WebError(f"Error estimating cost for task {task_id}! {error_msg}")
    typical = task_info.estFlexUnitTypical
    estimate_is_final_billed_cost = is_final_billed_cost
    if (task_info.taskType or "").upper() == TaskType.HEAT_CHARGE.name:
        if estimate_is_final_billed_cost is None and typical is not None and typical <= 0:
            estimate_is_final_billed_cost = True
    if estimate_is_final_billed_cost:
        typical = None
    elif typical is not None and typical <= 0:
        typical = None
    typical_cost_kind = None
    if typical is not None and (task_info.taskType or "").upper() == TaskType.HEAT_CHARGE.name:
        typical_cost_kind = _TYPICAL_COST_KIND_CHARGE_SOLVER_ITERATIONS
    estimate = FlexCreditEstimate(
        maximum=task_info.estFlexUnit,
        typical=typical,
        task_type=task_info.taskType,
        is_final_billed_cost=estimate_is_final_billed_cost,
        typical_cost_kind=typical_cost_kind,
    )
    if verbose:
        _log_flex_credit_estimate(console, estimate)
        fc_mode = task_info.estFlexCreditMode
        fc_post = task_info.estFlexCreditPostProcess
        if fc_mode:
            console.log(f"  {fc_mode:1.3f} FlexCredit of the total cost from mode solves.")
        if fc_post:
            console.log(f"  {fc_post:1.3f} FlexCredit of the total cost from post-processing.")
    return estimate


def estimate_cost(
    task_id: TaskId,
    verbose: bool = True,
    solver_version: str | None = None,
) -> float:
    """Compute the maximum FlexCredit charge for a given task."""
    return estimate_cost_info(
        task_id=task_id,
        verbose=verbose,
        solver_version=solver_version,
    ).maximum


@wait_for_connection
def real_cost(task_id: TaskId, verbose: bool = True) -> float | None:
    """Get the billed cost for given task after it has been run."""
    if not isinstance(task_id, str):
        raise ValueError(
            f"Task ID: {task_id} is not a string. You can get it using 'web.upload(<simulation>)'."
        )

    console = get_logging_console() if verbose else None
    task_info = get_info(task_id)
    flex_unit = task_info.realFlexUnit
    ori_flex_unit = getattr(task_info, "oriRealFlexUnit", flex_unit)
    if not flex_unit:
        log.warning(
            f"Billed FlexCredit for task '{task_id}' is not available. If the task has been "
            "successfully run, it should be available shortly."
        )
    elif verbose:
        console.log(f"Billed flex credit cost: {flex_unit:1.3f}.")
        if flex_unit != ori_flex_unit and "FDTD" in task_info.taskType:
            console.log(
                "Note: the task cost pro-rated due to early shutoff was below the minimum "
                "threshold, due to fast shutoff. Decreasing the simulation 'run_time' should "
                "decrease the estimated, and correspondingly the billed cost of such tasks."
            )
    return flex_unit
