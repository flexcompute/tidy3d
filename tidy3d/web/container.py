"""higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks."""
import os
from abc import ABC
from typing import Dict, Generator, Optional
import time

from rich.console import Console
from rich.progress import Progress

from . import webapi as web
from .task import TaskId, TaskInfo, RunInfo, TaskName
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..components.base import Tidy3dBaseModel


DEFAULT_DATA_PATH = "simulation_data.hdf5"
DEFAULT_DATA_DIR = "."


class WebContainer(Tidy3dBaseModel, ABC):
    """Base class for :class:`Job` and :class:`Batch`, technically not used"""


class Job(WebContainer):
    """Interface for managing the running of a :class:`.Simulation` on server.

    Parameters
    ----------
        simulation : :class:`.Simulation`
            Simulation to upload to server.
        task_name : str
            Name of task.
        folder_name : str = "default"
            Name of folder to store task on web UI.
        callback_url : str = None
            Http PUT url to receive simulation finish event. The body content is a json file with
            fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    """

    simulation: Simulation
    task_name: TaskName
    folder_name: str = "default"
    task_id: TaskId = None
    callback_url: str = None

    def run(
        self, path: str = DEFAULT_DATA_PATH, normalize_index: Optional[int] = 0
    ) -> SimulationData:
        """run :class:`Job` all the way through and return data.

        Parameters
        ----------
        path_dir : str = "./simulation_data.hdf5"
            Base directory where data will be downloaded, by default current working directory.
        normalize_index : int = 0
            If specified, normalizes the frequency-domain data by the amplitude spectrum of the
            source corresponding to ``simulation.sources[normalize_index]``.
            This occurs when the data is loaded into a :class:`SimulationData` object.
            To turn off normalization, set ``normalize_index`` to ``None``.

        Returns
        -------
        Dict[str: :class:`.SimulationData`]
            Dictionary mapping task name to :class:`.SimulationData` for :class:`Job`.
        """

        self.upload()
        self.start()
        self.monitor()
        return self.load(path=path, normalize_index=normalize_index)

    def upload(self) -> None:
        """Upload simulation to server without running.

        Note
        ----
        To start the simulation running, call :meth:`Job.start` after uploaded.
        """
        task_id = web.upload(
            simulation=self.simulation,
            task_name=self.task_name,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
        )
        self.task_id = task_id

    def get_info(self) -> TaskInfo:
        """Return information about a :class:`Job`.

        Returns
        -------
        :class:`TaskInfo`
            :class:`TaskInfo` object containing info about status, size, credits of task and others.
        """

        task_info = web.get_info(task_id=self.task_id)
        return task_info

    @property
    def status(self):
        """Return current status of :class:`Job`."""
        return self.get_info().status

    def start(self) -> None:
        """Start running a :class:`Job`.

        Note
        ----
        To monitor progress of the :class:`Job`, call :meth:`Job.monitor` after started.
        """
        web.start(self.task_id)

    def get_run_info(self) -> RunInfo:
        """Return information about the running :class:`Job`.

        Returns
        -------
        :class:`RunInfo`
            Task run information.
        """
        run_info = web.get_run_info(task_id=self.task_id)
        return run_info

    def monitor(self) -> None:
        """Monitor progress of running :class:`Job`.

        Note
        ----
        To load the output of completed simulation into :class:`.SimulationData`objets,
        call :meth:`Job.load`.
        """

        status = self.status
        console = Console()

        with console.status(f"[bold green]Working on '{self.task_name}'...", spinner="runner") as _:

            while status not in ("success", "error", "diverged", "deleted", "draft"):
                new_status = self.status
                if new_status != status:
                    console.log(f"status = {new_status}")
                    status = new_status
                time.sleep(web.REFRESH_TIME)

    def download(self, path: str = DEFAULT_DATA_PATH) -> None:
        """Download results of simulation.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).

        Note
        ----
        To load the data into :class:`.SimulationData`objets, can call :meth:`Job.load`.
        """
        web.download(task_id=self.task_id, path=path)

    def load(
        self, path: str = DEFAULT_DATA_PATH, normalize_index: Optional[int] = 0
    ) -> SimulationData:
        """Download results from simulation (if not already) and load them into ``SimulationData``
        object.

        Parameters
        ----------
        path : str = "./simulation_data.hdf5"
            Path to download data as ``.hdf5`` file (including filename).
        normalize_index : int = 0
            If specified, normalizes the frequency-domain data by the amplitude spectrum of the
            source corresponding to ``simulation.sources[normalize_index]``.
            This occurs when the data is loaded into a :class:`SimulationData` object.
            To turn off normalization, set ``normalize_index`` to ``None``.

        Returns
        -------
        :class:`.SimulationData`
            Object containing data about simulation.
        """
        return web.load(
            task_id=self.task_id,
            path=path,
            normalize_index=normalize_index,
        )

    def delete(self):
        """Delete server-side data associated with :class:`Job`."""
        web.delete(self.task_id)
        self.task_id = None


class Batch(WebContainer):
    """Interface for submitting several :class:`.Simulation` objects to sever.

    Parameters
    ----------
    simulations : Dict[str, :class:`.Simulation`]
        Mapping of task name to :class:`.Simulation` objects.
    folder_name : ``str`` = './'
        Name of folder to store member of each batch on web UI.
    """

    simulations: Dict[TaskName, Simulation]
    jobs: Dict[TaskName, Job] = None
    folder_name: str = "default"

    def run(self, path_dir: str = DEFAULT_DATA_DIR):
        """Upload and run each simulation in :class:`Batch`.
        Returns generator that can be used to loop through data results.

        Parameters
        ----------
        path_dir : str
            Base directory where data will be downloaded, by default current working directory.

        Yields
        ------
        str, :class:`.SimulationData`
            Yields the name of task
            and its corresponding :class:`.SimulationData` at each iteration.

        Note
        ----
        A typical usage might look like:

        >>> batch_results = batch.run()
        >>> for task_name, sim_data in batch_results:
        ...     # do something with data.

        Note that because ``batch_results`` is a generator, only the current iteration of
        :class:`.SimulationData` is stored in memory at a time.
        """

        self.upload()
        self.start()
        self.monitor()
        self.download(path_dir=path_dir)
        return self.items()

    def upload(self) -> None:
        """Create a series of tasks in the :class:`Batch` and upload them to server.

        Note
        ----
        To start the simulations running, must call :meth:`Batch.start` after uploaded.
        """
        self.jobs = {}
        for task_name, simulation in self.simulations.items():
            job = Job(simulation=simulation, task_name=task_name, folder_name=self.folder_name)
            self.jobs[task_name] = job
            job.upload()

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """Get information about each task in the :class:`Batch`.

        Returns
        -------
        Dict[str, :class:`TaskInfo`]
            Mapping of task name to data about task associated with each task.
        """
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def start(self) -> None:
        """Start running all tasks in the :class:`Batch`.

        Note
        ----
        To monitor the running simulations, can call :meth:`Batch.monitor`.
        """
        for _, job in self.jobs.items():
            job.start()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """get information about a each of the tasks in the :class:`Batch`.

        Returns
        -------
        Dict[str: :class:`RunInfo`]
            Maps task names to run info for each task in the :class:`Batch`.
        """
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self) -> None:  # pylint:disable=too-many-locals
        """Monitor progress of each of the running tasks.

        Note
        ----
        To loop through the data of completed simulations, can call :meth:`Batch.items`.
        """

        def pbar_description(task_name: str, status: str) -> str:
            return f"{task_name}: status = {status}"

        run_statuses = [
            "queued",
            "preprocess",
            "queued_solver",
            "running",
            "postprocess",
            "visualize",
            "success",
        ]
        end_statuses = ("success", "error", "diverged", "deleted", "draft")

        pbar_tasks = {}
        with Progress() as progress:
            for task_name, job in self.jobs.items():
                status = job.status
                description = pbar_description(task_name, status)
                pbar = progress.add_task(description, total=len(run_statuses) - 1)
                pbar_tasks[task_name] = pbar

            statuses = [job.status for _, job in self.jobs.items()]
            while any(s not in end_statuses for s in statuses):
                for status_index, (task_name, job) in enumerate(self.jobs.items()):
                    current_status = statuses[status_index]
                    new_status = job.status
                    if new_status != current_status:
                        completed = run_statuses.index(new_status)
                        pbar = pbar_tasks[task_name]
                        description = pbar_description(task_name, new_status)
                        progress.update(pbar, description=description, completed=completed)
                        statuses[status_index] = new_status
                statuses = [job.status for _, job in self.jobs.items()]
                time.sleep(web.REFRESH_TIME)

            for task_name, job in self.jobs.items():
                pbar = pbar_tasks[task_name]
                description = pbar_description(task_name, job.status)
                progress.update(pbar, description=description, completed=len(run_statuses))

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: str = DEFAULT_DATA_DIR):
        """Default path to data of a single :class:`Job` in :class:`Batch`.

        Parameters
        ----------
        task_id : str
            task_id corresponding to a :class:`Job`.
        path_dir : str = './'
            Base directory where data will be downloaded, by default, the current working directory.

        Returns
        -------
        str
            Full path to the data file.
        """
        return os.path.join(path_dir, f"{str(task_id)}.hdf5")

    def download(self, path_dir: str = DEFAULT_DATA_DIR) -> None:
        """Download results of each task.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default the current working directory.

        Note
        ----
        To load the data into :class:`.SimulationData`objets, can call :meth:`Batch.items`.

        The data for each task will be named as ``{path_dir}/{task_name}.hdf5``.

        """

        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_name, path_dir)
            job.download(path=job_path)

    def load(
        self, path_dir: str = DEFAULT_DATA_DIR, normalize_index: Optional[int] = 0
    ) -> Dict[TaskName, SimulationData]:
        """Download results and load them into :class:`.SimulationData` object.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.

        Returns
        -------
        Dict[str, :class:`.SimulationData`]
            Dictionary mapping task names to :class:`.SimulationData` for :class:`Batch`.

        Note
        ----
        This will return a dictionary of :class:`.SimulationData` objects,
        each of which can hold a large amount of data.
        If many simulations or large amounts of data,
        use ``for task_name, sim_data in Batch.items():``
        to instead loop through :class:`.SimulationData` objects and only store
        current iteration in memory.
        """
        sim_data_dir = {}
        self.download(path_dir=path_dir)
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load(path=job_path, normalize_index=normalize_index)
            sim_data_dir[task_name] = sim_data
        return sim_data_dir

    def delete(self):
        """Delete server-side data associated with each task in the batch."""
        for _, job in self.jobs.items():
            job.delete()
            self.jobs = None

    def items(self, path_dir: str = DEFAULT_DATA_DIR) -> Generator:
        """Generates :class:`.SimulationData` for batch.
        Used like: ``for task_name, sim_data in batch.items(): do something``.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.

        Yields
        ------
        str, :class:`.SimulationData`
            Yields the name of task
            and its corresponding :class:`.SimulationData` at each iteration.
        """
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load(path=job_path)
            yield task_name, sim_data
