"""Interface to run several jobs in batch using simplified syntax."""
from typing import Dict, List

from .container import DEFAULT_DATA_DIR, BatchData, Batch
from ..components.simulation import Simulation
from ..log import log


# pylint:disable=too-many-arguments, too-many-locals
def run_async(
    simulations: Dict[str, Simulation],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    num_workers: int = None,
    verbose: bool = True,
    simulation_type: str = "tidy3d",
    parent_tasks: Dict[str, List[str]] = None,
) -> BatchData:
    """Submits a set of :class:`.Simulation` objects to server, starts running,
    monitors progress, downloads, and loads results as a :class:`.BatchData` object.

    Parameters
    ----------
    simulations : Dict[str, :class:`.Simulation`]
        Mapping of task name to simulation.
    folder_name : str = "default"
        Name of folder to store each task on web UI.
    path_dir : str
        Base directory where data will be downloaded, by default current working directory.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    num_workers: int = None
        Number of tasks to submit at once in a batch, if None, will run all at the same time.
    verbose : bool = True
        If `True`, will print progressbars and status, otherwise, will run silently.

    Returns
    ------
    :class:`BatchData`
        Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.
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
        parent_tasks=parent_tasks,
    )

    batch_data = batch.run(path_dir=path_dir)
    return batch_data
