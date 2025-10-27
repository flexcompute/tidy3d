"""Interface to run several jobs in batch using simplified syntax."""

from __future__ import annotations

from os import PathLike
from typing import Literal, Optional, Union

from tidy3d.components.types.workflow import WorkflowType
from tidy3d.log import log
from tidy3d.web.core.types import PayType

from .container import DEFAULT_DATA_DIR, Batch, BatchData


def run_async(
    simulations: Union[dict[str, WorkflowType], tuple[WorkflowType], list[WorkflowType]],
    folder_name: str = "default",
    path_dir: PathLike = DEFAULT_DATA_DIR,
    callback_url: Optional[str] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    solver_version: Optional[str] = None,
    parent_tasks: Optional[dict[str, list[str]]] = None,
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: Union[PayType, str] = PayType.AUTO,
    priority: Optional[int] = None,
    lazy: bool = False,
) -> BatchData:
    """Submits a set of Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] objects to server,
    starts running, monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    .. TODO add example and see also reference.

    Parameters
    ----------
    simulations : Union[Dict[str, Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], tuple[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]], list[Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]]]
        Mapping of task name to simulation or list of simulations.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : PathLike
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    solver_version: Optional[str] = None
        Target solver version.
    reduce_simulation: Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type: Union[PayType, str] = PayType.AUTO
        Specify the payment method.
    priority: int = None
        Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
        It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
    lazy : bool = False
        Whether to load the actual data (``lazy=False``) or return a proxy that loads
        the data when accessed (``lazy=True``).

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
    """
    if simulation_type is None:
        simulation_type = "tidy3d"

    # if number of workers not specified, just use the number of simulations
    if num_workers is not None:
        log.warning(
            "The 'num_workers' kwarg does not have an effect anymore as all "
            "simulations will now be uploaded in a single batch."
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
    )

    batch_data = batch.run(path_dir=path_dir, priority=priority)
    return batch_data
