"""Provides lowest level, user-facing interface to server."""

from typing import Callable
import asyncio
from functools import wraps, partial


from .tidy3d_stub import SimulationType, SimulationDataType
from .connect_util import wait_for_connection
from ..core.environment import Env
from .webapi import upload, start, load, monitor
from ...log import log

# time between checking run status
RUN_REFRESH_TIME = 1.0
REFRESH_TIME = 5.0

# file names when uploading to S3
SIM_FILE_JSON = "simulation.json"


def _get_url(task_id: str) -> str:
    """Get the URL for a task on our server."""
    return f"{Env.current.website_endpoint}/workbench?taskId={task_id}"


def to_async(func):
    """Makes a function async."""

    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()  # Make event loop of nothing exists
        pfunc = partial(func, *args, **kwargs)  # Return function with variables (event) filled in
        return await loop.run_in_executor(executor, pfunc)

    return run


@wait_for_connection
async def async_run(
    simulation: SimulationType,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
    verbose: bool = False,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
    solver_version: str = None,
    worker_group: str = None,
) -> SimulationDataType:
    """
    Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.SimulationDataType` object.

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`]
        Simulation to upload to server.
    task_name : str
        Name of task.
    folder_name : str = "default"
        Name of folder to store task on web UI.
    path : str = "simulation_data.hdf5"
        Path to download results file (.hdf5), including filename.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = False
        Note: verbose must always be `False` otherwise the progressbars will crash.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version: str = None
        target solver version.
    worker_group: str = None
        worker group

    Returns
    -------
    Union[:class:`.SimulationData`, :class:`.HeatSimulationData`]
        Object containing solver results for the supplied simulation.

    Notes
    -----

        Submitting a simulation to our cloud server is very easily done by a simple web API call.

        .. code-block:: python

            sim_data = tidy3d.web.api.webapi.run(simulation, task_name='my_task', path='out/data.hdf5')

        The :meth:`tidy3d.web.api.webapi.run()` method shows the simulation progress by default.  When uploading a
        simulation to the server without running it, you can use the :meth:`tidy3d.web.api.webapi.monitor`,
        :meth:`tidy3d.web.api.container.Job.monitor`, or :meth:`tidy3d.web.api.container.Batch.monitor` methods to
        display the progress of your simulation(s).

    Examples
    --------

        To access the original :class:`.Simulation` object that created the simulation data you can use:

        .. code-block:: python

            # Run the simulation.
            sim_data = web.run(simulation, task_name='task_name', path='out/sim.hdf5')

            # Get a copy of the original simulation object.
            sim_copy = sim_data.simulation

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.monitor`
        Print the real time task progress until completion.

    :meth:`tidy3d.web.api.container.Job.monitor`
        Monitor progress of running :class:`Job`.

    :meth:`tidy3d.web.api.container.Batch.monitor`
        Monitor progress of each of the running tasks.
    """

    if verbose is not False:
        log.warning(
            "'verbose' must be 'False' in the asynchronous run function as the progress bars are "
            "not compatible with asynchronous execution. Setting 'verbose=False' internally. "
        )
        verbose = False

    task_id = upload(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        callback_url=callback_url,
        verbose=verbose,
        progress_callback=progress_callback_upload,
    )
    start(
        task_id,
        solver_version=solver_version,
        worker_group=worker_group,
    )
    log.info(f"Started asynchronous task with name '{task_name}'.")
    await to_async(monitor)(task_id, verbose=verbose)
    log.info(f"Finished asynchronous task with name '{task_name}'.")
    return load(
        task_id=task_id, path=path, verbose=verbose, progress_callback=progress_callback_download
    )
