"""Interface to run several jobs in batch using asyncio."""
import asyncio
from typing import Dict

from rich.progress import Progress
import nest_asyncio

from .container import DEFAULT_DATA_DIR, BatchData, Job, Batch
from ..components.simulation import Simulation

# note: this is to avoid annoying errors with asyncio.run() on jupyter notebooks
nest_asyncio.apply()


# pylint:disable=too-many-arguments, too-many-locals
async def _run_async(
    simulations: Dict[str, Simulation],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    num_workers: int = None,
    verbose: bool = True,
) -> BatchData:
    """Submits a set of :class:`.Simulation` objects to server, starts running,
    monitors progress, downloads, and loads results as a :class:`.BatchData` object.
    Uses ``asyncio`` to perform these steps asynchronously.

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

    Note
    ----
    This is an experimental feature and may not work on all systems or configurations.
    For more details, see ``https://realpython.com/async-io-python/``.

    Returns
    ------
    :class:`BatchData`
        Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.
    """

    # if number of workers not specified, just use the number of simulations
    if num_workers is None:
        num_workers = len(simulations)

    def get_path(job: Job) -> str:
        """Get the data path associated with a :class:`.Job` using :class:`.Batch` convention."""

        # pylint:disable=protected-access
        return Batch._job_data_path(task_id=job.task_id, path_dir=path_dir)

    # initialize a queue
    queue = asyncio.Queue()

    # store task ids and data paths to initialize a BatchData object with for return
    task_ids = {}
    task_paths = {}

    def start_job(simulation: Simulation, task_name: str) -> None:
        """make a job, start it, and add it to the queue."""

        # make and start a job
        job = Job(
            simulation=simulation,
            task_name=task_name,
            callback_url=callback_url,
            folder_name=folder_name,
            verbose=False,  # note, set verbose=False here because we use a custom progressbar below
        )
        job.start()

        # store the task id and data path
        task_ids[task_name] = job.task_id
        task_paths[task_name] = get_path(job)

        # put the job on the queue
        queue.put_nowait(job)

    async def worker(queue, progress: Progress = None, pbars: dict = None) -> None:
        """Defines a worker that runs a job."""

        while True:

            # grab a job from the queue
            job = await queue.get()

            if verbose and pbars and progress:
                # mark the job as running
                progress.update(
                    pbars[job.task_name], description=f'running "{job.task_name}"', advance=1
                )

            # monitor it
            job.monitor()

            if verbose and pbars and progress:
                # and mark the job as completed
                progress.update(
                    pbars[job.task_name],
                    description=f'done w/ "{job.task_name}"',
                    advance=1,
                    refresh=True,
                )

            queue.task_done()

    async def complete_tasks(progress: Progress = None, pbars: dict = None) -> None:
        """Generate a bunch of workers to work on task and combine results."""

        # create a bunch of workers to asynchonously work on the task
        tasks = []
        for _ in range(num_workers):
            task = asyncio.create_task(worker(queue, progress=progress, pbars=pbars))
            tasks.append(task)

        # Wait until the queue is fully processed
        await queue.join()

        # Cancel our worker tasks
        for task in tasks:
            task.cancel()

        # Wait until all worker tasks are cancelled
        await asyncio.gather(*tasks, return_exceptions=True)

    # main logic: start all the jobs, and then call complete_tasks, with pbar wrapping if verbose
    if verbose:

        with Progress() as progress:

            pbars = {}

            for task_name, simulation in simulations.items():

                # make a progressbar
                pbar = progress.add_task(task_name, total=3)
                pbars[task_name] = pbar

                start_job(simulation=simulation, task_name=task_name)

                # update the progressbar
                progress.update(pbar, description=f'starting "{task_name}"', advance=1)

            await complete_tasks(progress=progress, pbars=pbars)

    else:

        for task_name, simulation in simulations.items():
            start_job(simulation=simulation, task_name=task_name)

        await complete_tasks()

    # return the batch data containing all of the job run details for loading later
    return BatchData(task_ids=task_ids, task_paths=task_paths, verbose=verbose)


# pylint:disable=too-many-arguments
def run_async(
    simulations: Dict[str, Simulation],
    folder_name: str = "default",
    path_dir: str = DEFAULT_DATA_DIR,
    callback_url: str = None,
    num_workers: int = None,
    verbose: bool = True,
) -> BatchData:
    """Submits a set of :class:`.Simulation` objects to server, starts running,
    monitors progress, downloads, and loads results as a :class:`.BatchData` object.
    Uses ``asyncio`` to perform these steps asynchronously.

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

    Note
    ----
    This is an experimental feature and may not work on all systems or configurations.
    For more details, see ``https://realpython.com/async-io-python/``.

    Returns
    ------
    :class:`BatchData`
        Contains the :class:`.SimulationData` of each :class:`.Simulation` in :class:`Batch`.
    """

    return asyncio.run(
        _run_async(
            simulations=simulations,
            folder_name=folder_name,
            path_dir=path_dir,
            callback_url=callback_url,
            num_workers=num_workers,
            verbose=verbose,
        )
    )
