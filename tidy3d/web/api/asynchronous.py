"""Interface to run several jobs in batch using simplified syntax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .container import DEFAULT_DATA_DIR, Batch
from .run_options import log_deprecated_run_args

if TYPE_CHECKING:
    from os import PathLike
    from typing import Literal

    from tidy3d.components.types.workflow import WorkflowOperationType
    from tidy3d.web.core.types import PayType

    from .container import BatchData


def run_async(
    simulations: dict[str, WorkflowOperationType]
    | tuple[WorkflowOperationType]
    | list[WorkflowOperationType],
    folder_name: str = "default",
    path_dir: PathLike = DEFAULT_DATA_DIR,
    callback_url: str | None = None,
    num_workers: int | None = None,
    verbose: bool = True,
    simulation_type: str | None = None,
    solver_version: str | None = None,
    parent_tasks: dict[str, list[str]] | None = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    lazy: bool = False,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
) -> BatchData:
    """Submits a set of Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] objects to server,
    starts running, monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    .. TODO add example and see also reference.

    Parameters
    ----------
    simulations : Union[dict[str, Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], tuple[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], list[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]]]
        Mapping of task name to simulation or list of simulations.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : PathLike
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of worker threads used by configurable :class:`Batch` stages (for example,
        monitoring downloads). Upload and start use a separate fixed concurrency of 64 workers.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : Optional[str] = None
        Internal simulation type label; external users should leave unset.
    solver_version: Optional[str] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.solver_version`` instead; external users should leave unset.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type : Optional[Union[PayType, str]] = None
        Deprecated direct argument for internal use only. Internal workflows should set
        ``td.config.run.pay_type`` instead; external users should leave unset.
    priority: int = None
        Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
        It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
    lazy : bool = False
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
    ------
    :class:`BatchData`
        Contains the Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`] for each
        Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.

    See Also
    --------

    :class:`Job`:
        Interface for managing the running of a Simulation on server.

    :class:`Batch`
        Interface for submitting several :class:`.Simulation` objects to sever.

    Notes
    -----
    Passing ``solver_version``, ``pay_type``, ``priority``, and vGPU options
    directly is deprecated. Set defaults via ``td.config.run`` and
    ``td.config.vgpu`` instead. Non-``None`` values passed here override the
    config for this call.
    """
    log_deprecated_run_args(
        solver_version=solver_version,
        pay_type=pay_type,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )

    batch = Batch(
        simulations=simulations,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        simulation_type=simulation_type,
        solver_version=solver_version,
        parent_tasks=parent_tasks,
        reduce_simulation=reduce_simulation,
        pay_type=pay_type,
        lazy=lazy,
        **({"num_workers": num_workers} if num_workers is not None else {}),
    )

    batch_data = batch.run(
        path_dir=path_dir,
        priority=priority,
        vgpu_allocation=vgpu_allocation,
        ignore_memory_limit=ignore_memory_limit,
    )
    return batch_data
