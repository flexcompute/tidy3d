"""Provides lowest level, user-facing interface to server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from requests import HTTPError

from tidy3d.exceptions import WebError
from tidy3d.log import get_logging_console
from tidy3d.web.api import task_api
from tidy3d.web.api.container import Job
from tidy3d.web.core.account import Account
from tidy3d.web.core.http_util import config_toml_path

from .connect_util import wait_for_connection
from .run_options import log_deprecated_run_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from typing import Literal

    from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType
    from tidy3d.web.core.types import PayType

RUN_REFRESH_TIME = task_api.RUN_REFRESH_TIME
REFRESH_TIME = task_api.REFRESH_TIME
SIM_FILE_JSON = task_api.SIM_FILE_JSON
GUI_SUPPORTED_TASK_TYPES = task_api.GUI_SUPPORTED_TASK_TYPES
BETA_TASK_TYPES = task_api.BETA_TASK_TYPES
SOLVER_NAME = task_api.SOLVER_NAME
DEFAULT_DATA_FILENAME = task_api.DEFAULT_DATA_FILENAME

_build_website_url = task_api._build_website_url
_get_url = task_api._get_url
_get_folder_url = task_api._get_folder_url
_get_url_rf = task_api._get_url_rf
_get_task_urls = task_api._get_task_urls
_batch_detail_error = task_api._batch_detail_error
_copy_simulation_data_from_cache_entry = task_api._copy_simulation_data_from_cache_entry
_load_simulation_via_tempfile = task_api._load_simulation_via_tempfile
_get_batch_detail_handle_error_status = task_api._get_batch_detail_handle_error_status
_batch_detail_progress = task_api._batch_detail_progress
_monitor_modeler_batch = task_api._monitor_modeler_batch
Tidy3dStub = task_api.Tidy3dStub
Tidy3dStubData = task_api.Tidy3dStubData
resolve_local_cache = task_api.resolve_local_cache

default_data_filename = task_api.default_data_filename
_resolve_output_path = task_api._resolve_output_path
restore_simulation_if_cached = task_api.restore_simulation_if_cached
load_simulation_if_cached = task_api.load_simulation_if_cached
get_reduced_simulation = task_api.get_reduced_simulation
get_info = task_api.get_info
get_run_info = task_api.get_run_info
get_status = task_api.get_status
abort = task_api.abort
download_json = task_api.download_json
delete_old = task_api.delete_old
load_simulation = task_api.load_simulation
download_log = task_api.download_log
delete = task_api.delete
download_simulation = task_api.download_simulation
get_tasks = task_api.get_tasks
estimate_cost = task_api.estimate_cost
real_cost = task_api.real_cost
_upload = task_api._upload


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
) -> str:
    """
    Upload a simulation to the server without starting it.

    Parameters
    ----------
    simulation : WorkflowOperationType
        Simulation or single-step workflow operation to upload. Multi-step workflows should be
        submitted with :func:`run` or :class:`Job`.
    task_name : Optional[str] = None
        Name of the task. If omitted, a default task name is generated.
    folder_name : str = "default"
        Name of the web UI folder that will contain the task.
    callback_url : Optional[str] = None
        HTTP PUT URL that receives the simulation finish event.
    verbose : bool = True
        Whether to print progress, links, and status messages.
    progress_callback : Callable[[float], None] = None
        Optional callback used while uploading data.
    simulation_type : Optional[str] = None
        Simulation category override. If omitted, the configured default is used.
    parent_tasks : list[str] = None
        Existing upstream task IDs associated with this upload.
    source_required : bool = True
        Whether source-free simulations should be rejected before upload.
    solver_version : Optional[str] = None
        Solver version override. If omitted, the configured default is used.
    reduce_simulation : Literal["auto", True, False] = "auto"
        Whether to reduce structures to the simulation domain when supported.
    verbose_estimate_cost : Optional[bool] = None
        Whether to print cost estimation. If omitted, follows ``verbose``.

    Returns
    -------
    str
        Unique task ID assigned by the server.
    """
    return task_api.upload(
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
    )


def start(
    task_id: str,
    solver_version: str | None = None,
    worker_group: str | None = None,
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> None:
    """
    Start running an uploaded task.

    Parameters
    ----------
    task_id : str
        Unique task ID returned by :func:`upload`.
    solver_version : Optional[str] = None
        Solver version override. If omitted, the configured default is used.
    worker_group : Optional[str] = None
        Worker group override. If omitted, the configured default is used.
    pay_type : Optional[Union[PayType, str]] = None
        Payment method override. If omitted, the configured default is used.
    priority : Optional[int] = None
        vGPU queue priority, where 1 is lowest and 10 is highest.
    vgpu_allocation : Optional[int] = None
        Number of virtual GPUs to allocate for vGPU-license runs.
    ignore_memory_limit : Optional[bool] = None
        Whether to allow runs above the estimated vGPU allocation limit.

    Note
    ----
    For server-side batch task IDs, ``priority`` and ``vgpu_allocation`` are
    forwarded only when explicitly supplied. Configured vGPU defaults are not
    applied implicitly, and ``ignore_memory_limit`` is not supported.
    """
    return task_api.start(
        task_id=task_id,
        solver_version=solver_version,
        worker_group=worker_group,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )


def monitor(task_id: str, verbose: bool = True, worker_group: str | None = None) -> None:
    """
    Print real-time task progress until completion.

    Parameters
    ----------
    task_id : str
        Unique task ID returned by :func:`upload`.
    verbose : bool = True
        Whether to print progress bars and status messages.
    worker_group : Optional[str] = None
        Deprecated worker-group argument retained for compatibility.
    """
    return task_api.monitor(task_id=task_id, verbose=verbose, worker_group=worker_group)


def download(
    task_id: str,
    path: PathLike | None = None,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """
    Download task results to an HDF5 file.

    Parameters
    ----------
    task_id : str
        Unique task ID returned by :func:`upload`.
    path : Optional[PathLike] = None
        Output file path. If omitted, a task-type-specific default filename is used.
    verbose : bool = True
        Whether to print download progress.
    progress_callback : Callable[[float], None] = None
        Optional callback used while downloading data.
    """
    return task_api.download(
        task_id=task_id,
        path=path,
        verbose=verbose,
        progress_callback=progress_callback,
    )


@wait_for_connection
def run(
    simulation: WorkflowOperationType,
    task_name: str | None = None,
    folder_name: str = "default",
    path: PathLike | None = None,
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] | None = None,
    progress_callback_download: Callable[[float], None] | None = None,
    solver_version: str | None = None,
    worker_group: str | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool = False,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> WorkflowDataType:
    """
    Submit a simulation-like object, run it to completion, download results, and load data.

    ``HeatSimulation`` and ``HeatChargeSimulation`` inputs run by first creating a volume
    mesh and then running the solver. The returned object is the final simulation data
    object from the solver step.

    Parameters
    ----------
    simulation : WorkflowOperationType
        Simulation or supported workflow operation to submit to the server.
    task_name : Optional[str] = None
        Name of the task. If omitted, a default task name is generated.
    folder_name : str = "default"
        Name of the web UI folder that will contain the task.
    path : Optional[PathLike] = None
        Path to download the results file (.hdf5), including filename. When ``None``,
        a task-type-specific default filename is used.
    callback_url : Optional[str] = None
        HTTP PUT URL that receives the simulation finish event. The body content is a
        JSON object with fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, print progress bars, links, and status messages. If ``False``,
        run silently.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback used while uploading data.
    progress_callback_download : Callable[[float], None] = None
        Optional callback used while downloading data.
    solver_version : Optional[str] = None
        Solver version override. If ``None``, uses ``td.config.run.solver_version``.
    worker_group : Optional[str] = None
        Worker group override. If ``None``, uses ``td.config.run.worker_group``.
    simulation_type : Optional[str] = None
        Simulation category override. If ``None``, uses ``td.config.run.simulation_type``.
    parent_tasks : list[str] = None
        Existing upstream task IDs associated with this run. For Heat/HeatCharge runs,
        this may contain one existing volume-mesh task ID to reuse for the solver step.
    reduce_simulation : Literal["auto", True, False] = "auto"
        Whether to reduce structures to the simulation domain when supported. Currently
        only implemented for mode-solver uploads.
    pay_type : Optional[Union[PayType, str]] = None
        Payment method override. If ``None``, uses ``td.config.run.pay_type``.
    priority : Optional[int] = None
        Priority in the virtual GPU queue (1 = lowest, 10 = highest). This affects only
        simulations from vGPU licenses and does not impact FlexCredit runs.
    lazy : bool = False
        Whether to load the actual data immediately (``lazy=False``) or return a proxy
        that loads data when accessed (``lazy=True``).
    vgpu_allocation : Optional[int] = None
        Number of virtual GPUs to allocate for vGPU-license runs. If ``None``, uses
        ``td.config.vgpu.vgpu_allocation``. If that is also unset, the system determines
        the GPU count automatically.
    ignore_memory_limit : Optional[bool] = None
        If ``True``, allow the simulation to run even when estimated vGPU memory exceeds
        the allocation limit (up to 2x the limit). If ``None``, uses
        ``td.config.vgpu.ignore_memory_limit``.

    Returns
    -------
    WorkflowDataType
        Data object containing the results for the supplied operation. For Heat/HeatCharge
        runs, this is the final solver-step data object.

    Notes
    -----
    Passing run options directly is deprecated. Set defaults via ``td.config.run`` and
    ``td.config.vgpu`` instead. Non-``None`` values passed here override the config for
    this call.

    Examples
    --------
    Run a simulation and load its results:

    .. code-block:: python

        sim_data = web.run(simulation, task_name="task_name", path="out/sim.hdf5")

    The original simulation object can be accessed from the returned data object:

    .. code-block:: python

        sim_copy = sim_data.simulation

    See Also
    --------
    :meth:`tidy3d.web.api.webapi.monitor`
        Print the real-time task progress until completion.
    :class:`tidy3d.web.api.container.Job`
        Lower-level handle for uploading, running, stepping, loading, and deleting one job.
    :class:`tidy3d.web.api.container.Batch`
        Run and load groups of jobs.
    """
    log_deprecated_run_args(
        solver_version=solver_version,
        worker_group=worker_group,
        simulation_type=simulation_type,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )
    job = Job(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        solver_version=solver_version,
        simulation_type=simulation_type,
        parent_tasks=tuple(parent_tasks) if parent_tasks else None,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        lazy=lazy,
    )
    return job.run(
        path=path,
        progress_callback_upload=progress_callback_upload,
        progress_callback_download=progress_callback_download,
        worker_group=worker_group,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )


def load(
    task_id: str | None,
    path: PathLike | None = None,
    replace_existing: bool = True,
    verbose: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    lazy: bool = False,
) -> WorkflowDataType:
    """
    Download and load simulation results into a data object.

    Parameters
    ----------
    task_id : Optional[str]
        Unique task ID returned by :func:`upload`. If ``None``, ``path`` is assumed to
        point to an existing cached results file.
    path : Optional[PathLike] = None
        Download path to the .hdf5 data file, including filename. When ``None`` and
        ``task_id`` is provided, a task-type-specific default filename is used.
    replace_existing : bool = True
        Whether to download data even if ``path`` already exists, overwriting the file.
    verbose : bool = True
        If ``True``, print download and loading messages. If ``False``, run silently.
    progress_callback : Callable[[float], None] = None
        Optional callback used while downloading data.
    lazy : bool = False
        Whether to load the actual data immediately (``lazy=False``) or return a proxy
        that loads data when accessed (``lazy=True``).

    Returns
    -------
    WorkflowDataType
        Data object containing task results.

    Notes
    -----
    To load already-downloaded results without replacing the file, call this function
    with ``replace_existing=False``.
    """
    return task_api.load(
        task_id=task_id,
        path=path,
        replace_existing=replace_existing,
        verbose=verbose,
        progress_callback=progress_callback,
        lazy=lazy,
    )


@wait_for_connection
def account(verbose: bool = True) -> Account:
    """Get account information including FlexCredit balance and usage limits."""
    account_info = Account.get()
    if verbose and account_info:
        console = get_logging_console()
        credit = account_info.credit
        credit_expiration = account_info.credit_expiration
        cycle_type = account_info.allowance_cycle_type
        cycle_amount = account_info.allowance_current_cycle_amount
        cycle_end_date = account_info.allowance_current_cycle_end_date
        free_simulation_counts = account_info.daily_free_simulation_counts

        message = ""
        if credit is not None:
            message += f"Current FlexCredit balance: {credit:.2f}"
            if credit_expiration is not None:
                message += (
                    f" and expiration date: {credit_expiration.strftime('%Y-%m-%d %H:%M:%S')}. "
                )
            else:
                message += ". "
        if cycle_type is not None and cycle_amount is not None and cycle_end_date is not None:
            cycle_end = cycle_end_date.strftime("%Y-%m-%d %H:%M:%S")
            message += f"{cycle_type} FlexCredit balance: {cycle_amount:.2f} and expiration date: {cycle_end}. "
        if free_simulation_counts is not None:
            message += f"Remaining daily free simulations: {free_simulation_counts}."

        console.log(message)

    return account_info


@wait_for_connection
def test() -> None:
    """Confirm whether Tidy3D authentication is configured."""
    try:
        get_tasks(num_tasks=0)
        console = get_logging_console()
        console.log("Authentication configured successfully!")
    except (WebError, HTTPError) as e:
        url = "https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html"
        msg = (
            str(e)
            + "\n\n"
            + "It looks like the Tidy3D Python interface is not configured with your "
            "unique API key. "
            "To get your API key, sign into 'https://tidy3d.simulation.cloud' and copy it "
            "from your 'Account' page. Then you can configure tidy3d through command line "
            "'tidy3d configure' (recommended). Alternatively, one can manually create the configuration "
            f"file at '{config_toml_path()}' with content like: \n\n"
            "[web]\n"
            "apikey = 'XXX' \n\nHere XXX is your API key copied from your account page within quotes.\n\n"
            f"For details, check the instructions at {url}."
        )
        raise WebError(msg) from e
