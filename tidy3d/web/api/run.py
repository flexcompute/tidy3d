from __future__ import annotations

import typing
from pathlib import Path
from typing import TYPE_CHECKING

from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType
from tidy3d.config import config
from tidy3d.log import get_logging_console
from tidy3d.web.api.autograd.autograd import run as run_autograd
from tidy3d.web.api.autograd.autograd import run_async
from tidy3d.web.api.container import DEFAULT_DATA_DIR
from tidy3d.web.api.container_utils import flatten_container, validate_mapping_key
from tidy3d.web.api.run_options import log_deprecated_run_args
from tidy3d.web.api.tidy3d_stub import task_type_name_of
from tidy3d.web.api.webapi import default_data_filename

if TYPE_CHECKING:
    from os import PathLike

    from tidy3d.web.core.types import PayType

RunInput: typing.TypeAlias = (
    WorkflowOperationType
    | list["RunInput"]
    | tuple["RunInput", ...]
    | dict[typing.Hashable, "RunInput"]
)

RunOutput: typing.TypeAlias = (
    WorkflowDataType
    | list["RunOutput"]
    | tuple["RunOutput", ...]
    | dict[typing.Hashable, "RunOutput"]
)


def _validate_run_mapping_key(key: object) -> None:
    """Reject unhashable keys and simulation objects in run containers."""
    validate_mapping_key(key)
    if isinstance(key, WorkflowOperationType):
        raise ValueError("Dict keys must not be simulations.")


def _collect_by_hash(
    node: RunInput,
) -> dict[str, WorkflowOperationType]:
    """Traverses the structure and collects all simulations into a `{hash: sim}` mapping.
    The latest occurrence of the same hash overwrites the previous one — which is fine
    since identical objects share the same hash."""
    return typing.cast(
        dict[str, WorkflowOperationType],
        flatten_container(
            node,
            is_leaf=lambda value: isinstance(value, WorkflowOperationType),
            validate_dict_key=_validate_run_mapping_key,
            leaf_id=lambda _path, simulation: simulation._hash_self(),
        ),
    )


def _reconstruct_by_hash(node: RunInput, h2data: dict[str, WorkflowDataType]) -> RunOutput:
    """Replaces each leaf node (simulation) with its corresponding data object.

    If a simulation appears multiple times in the input structure, the first
    occurrence reuses the same object; subsequent occurrences receive a `.copy()`
    to avoid shared-state side effects.
    """
    seen = set()

    def _recur(item: RunInput) -> RunOutput:
        if isinstance(item, WorkflowOperationType):
            hash_value = item._hash_self()
            data = h2data[hash_value]
            if hash_value in seen:
                return data.copy()
            seen.add(hash_value)
            return data
        if isinstance(item, tuple):
            return tuple(_recur(value) for value in item)
        if isinstance(item, list):
            return [_recur(value) for value in item]
        if isinstance(item, dict):
            return {key: _recur(value) for key, value in item.items()}
        raise TypeError(f"Unsupported element in reconstruction: {type(item)!r}")

    return _recur(node)


def run(
    simulation: RunInput,
    task_name: str | None = None,
    folder_name: str = "default",
    path: PathLike | None = None,
    callback_url: str | None = None,
    verbose: bool = True,
    progress_callback_upload: typing.Callable[[float], None] | None = None,
    progress_callback_download: typing.Callable[[float], None] | None = None,
    solver_version: str | None = None,
    worker_group: str | None = None,
    simulation_type: str | None = None,
    parent_tasks: list[str] | None = None,
    local_gradient: bool | None = None,
    max_num_adjoint_per_fwd: int | None = None,
    reduce_simulation: typing.Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    max_workers: int | None = None,
    lazy: bool | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> RunOutput:
    """
    Submit one or many simulations and return results in the same container shape.

    This is a convenience wrapper around the autograd runners that accepts a single
    simulation **or** an arbitrarily nested container of simulations
    (`list`, `tuple`, or `dict` values). Internally, all simulations are collected,
    deduplicated by object hash, executed either synchronously (single) or
    asynchronously (batch), and the returned data objects are reassembled to mirror
    the input structure.

    **Path behavior**
      - **Single simulation:** results are downloaded to ``f"{path}.hdf5"`` when ``path`` is
        provided without a suffix.
      - **Multiple simulations:** ``path`` is treated as a **directory**, and each
        task will write its own results file inside that directory.

    **Lazy loading**
      - If ``lazy`` is *not* specified: single runs default to ``False`` (eager load);
        batch runs default to ``True`` (proxy objects that load on first access).

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] | list | tuple | dict
        A simulation or a container whose leaves are simulations.
        Supported containers are ``list``, ``tuple``, and ``dict`` (values only).
        Dict **keys must not** be simulations.
    task_name : Optional[str] = None
        Optional name for a single run. Prefixed for multiple runs.
    folder_name : str = "default"
        Folder shown on the web UI.
    path : Optional[PathLike] = None
        Output path. Interpreted as a file path for single simulations and a directory for multiple simulations.
        Defaults are task-type-specific filenames for single simulations and the current
        directory for multiple simulations.
    callback_url : Optional[str] = None
        Optional HTTP PUT endpoint to receive completion events.
    verbose : bool = True
        If ``True``, print status and progress; otherwise run quietly.
    progress_callback_upload : Optional[Callable[[float], None]] = None
        Callback invoked with byte counts during upload (single-run path only).
    progress_callback_download : Optional[Callable[[float], None]] = None
        Callback invoked with byte counts during download (single-run path only).
    solver_version : Optional[str] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.solver_version`` instead; external users should leave unset.
    worker_group : Optional[str] = None
        Deprecated direct argument for worker group targeting, for internal use only. Internal
        workflows should set ``td.config.run.worker_group`` instead; external users should leave
        unset.
    simulation_type : Optional[str] = None
        Internal simulation type label; external users should leave unset.
    parent_tasks : Optional[List[str]] = None
        Parent task IDs, if any.
    local_gradient : Optional[bool] = None
        Compute gradients locally (more downloads; useful for experimental features).
        Defaults to ``config.adjoint.local_gradient`` when not provided. Remote gradients
        always use server-side defaults.
    max_num_adjoint_per_fwd : Optional[int] = None
        Maximum number of adjoint simulations allowed per forward run. Defaults to
        ``config.adjoint.max_adjoint_per_fwd`` when not provided.
    reduce_simulation : {"auto", True, False} = "auto"
        Whether to reduce structures to the simulation domain (mode solver only).
    pay_type : Optional[Union[PayType, str]] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.pay_type`` instead; external users should leave unset.
    priority : Optional[int] = None
        Queue priority for vGPU (1 = lowest, 10 = highest).
    max_workers : Optional[int] = None
        Maximum parallel submissions for batch runs. ``None`` submits all at once.
    lazy : Optional[bool] = None
        Whether to load the actual data (``lazy=False``) or return a proxy that loads
        the data when accessed (``lazy=True``).
    vgpu_allocation : Optional[int] = None
        Number of virtual GPUs to allocate for the simulation (1, 2, 4, or 8).
        Only applies to vGPU license users. If not specified, uses
        ``td.config.vgpu.vgpu_allocation``.
        If that is also unset, the system
        automatically determines the optimal GPU count.
    ignore_memory_limit : Optional[bool] = None
        If ``True``, allows the simulation to run even when estimated vGPU memory
        exceeds the allocation limit (up to 2x the limit). Only applies to
        vGPU license users. If ``None``, uses ``td.config.vgpu.ignore_memory_limit``.

    Returns
    -------
    RunOutput
        A data object (or nested container of data objects) matching the input
        container shape. Leaves are instances of the corresponding
        :class:`WorkflowDataType`.

    Notes
    -----
    - Simulations are indexed by ``hash(sim)``. If the *same object* appears multiple
      times in the input, it is executed once and its data is reused at all positions.
      The *last* occurrence wins if duplicates with the same hash are encountered.
    - For each simulation, a mode-solver compatibility patch is applied so that
      the returned data exposes expected convenience attributes.
    - ``progress_callback_*`` are only used in the single-run code path.
    - Passing ``solver_version``, ``worker_group``, ``pay_type``, ``priority``,
      and vGPU options directly is deprecated. Set defaults via ``td.config.run``
      and ``td.config.vgpu`` instead. Non-``None`` values passed here override
      the config for this call.

    Raises
    ------
    ValueError
        If no simulations are found in ``simulation``.
    TypeError
        If an unsupported container element is encountered, or if a dict key is a
        simulation object.

    Examples
    --------
    Single run (eager by default)

    .. code-block:: python

        sim_data = run(sim, task_name="wg_bend", path="out/bend")
        # writes: "out/bend.hdf5"

    Batch run with nested structure (lazy by default)

    .. code-block:: python

        sims = {
            "coarse": [sim_a, sim_b],
            "fine": sim_c,
        }
        data = run(sims, path="out/batch_dir", max_workers=4)

        # 'data' mirrors 'sims' structure:
        # data["coarse"][0] -> data for sim_a, etc.

    See Also
    --------
    tidy3d.web.api.autograd.autograd.run
        Underlying autograd single-run implementation.
    tidy3d.web.api.autograd.autograd.run_async
        Underlying autograd batch submission implementation.
    """
    h2sim = _collect_by_hash(simulation)
    if not h2sim:
        raise ValueError("No simulation data found in simulation input.")

    if local_gradient is None:
        local_gradient = bool(config.adjoint.local_gradient)

    if max_num_adjoint_per_fwd is None:
        max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

    key_prefix = ""
    if len(h2sim) == 1:
        if path is not None:
            # user may submit the same simulation multiple times and not specify an extension, but dir path
            if not Path(path).suffixes:
                path = f"{path}.hdf5"
                console = get_logging_console()
                console.log(f"Changed output path to {path}")
        h, sim = next(iter(h2sim.items()))
        if path is None:
            path = default_data_filename(task_type_name_of(sim))
        log_deprecated_run_args(simulation_type=simulation_type)
        data = {
            h: run_autograd(
                simulation=sim,
                task_name=task_name,
                folder_name=folder_name,
                path=path,
                callback_url=callback_url,
                verbose=verbose,
                progress_callback_upload=progress_callback_upload,
                progress_callback_download=progress_callback_download,
                solver_version=solver_version,
                worker_group=worker_group,
                simulation_type=simulation_type,
                parent_tasks=parent_tasks,
                local_gradient=local_gradient,
                max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
                reduce_simulation=reduce_simulation,
                pay_type=pay_type,
                priority=priority,
                vgpu_allocation=vgpu_allocation,
                ignore_memory_limit=ignore_memory_limit,
                lazy=lazy if lazy is not None else False,
            )
        }
    else:
        key_prefix = f"{task_name}_" if task_name else ""
        sims = {f"{key_prefix}{h}": s for h, s in h2sim.items()}
        path_dir = Path(path) if path is not None else Path(DEFAULT_DATA_DIR)
        data = run_async(
            simulations=sims,
            folder_name=folder_name,
            path_dir=path_dir,
            callback_url=callback_url,
            num_workers=max_workers,
            verbose=verbose,
            simulation_type=simulation_type,
            solver_version=solver_version,
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            reduce_simulation=reduce_simulation,
            pay_type=pay_type,
            priority=priority,
            vgpu_allocation=vgpu_allocation,
            ignore_memory_limit=ignore_memory_limit,
            lazy=lazy if lazy is not None else True,
        )

    h2data: dict[str, WorkflowDataType] = {}
    for k, sim_data in data.items():
        h = k[len(key_prefix) :] if key_prefix else k
        h2data[h] = sim_data

    return _reconstruct_by_hash(simulation, h2data)
