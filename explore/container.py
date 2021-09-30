""" higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks. """

import os
from abc import ABC
from typing import Dict
import time

from rich.console import Console
from rich.progress import Progress

from . import webapi as web
from .task import TaskId, TaskInfo, RunInfo, TaskStatus
from ..components.simulation import Simulation
from ..components.data import SimulationData
from ..components.base import Tidy3dBaseModel


class WebContainer(Tidy3dBaseModel, ABC):
    """base class for job and batch, technically not used"""


# type of task_name
TaskName = str


class Job(WebContainer):
    """Holds a task and its simulation"""

    simulation: Simulation
    task_name: TaskName
    task_id: TaskId = None

    def upload(self) -> None:
        """upload simulation to server, record task id"""
        task_id = web.upload(simulation=self.simulation)
        self.task_id = task_id

    def get_info(self) -> TaskInfo:
        """return general information about a job's task"""
        task_info = web.get_info(task_id=self.task_id)
        return task_info

    def run(self) -> None:
        """start running a task"""
        web.run(self.task_id)

    def get_run_info(self) -> RunInfo:
        """get information about a running task"""
        run_info = web.get_run_info(task_id=self.task_id)
        return run_info

    def get_status(self) -> TaskStatus:
        """gets current status"""
        task = web.monitor(task_id=self.task_id)
        return task.info.status

    def monitor(self) -> None:
        """monitor progress of running task"""
        console = Console()
        prev_status = self.get_status()
        console.log(f"status = {prev_status.value}")
        with console.status(f"[bold green]Monitoring Job '{self.task_name}' ...") as console_status:
            for _ in web.run(task_id=self.task_id):
                time.sleep(0.01)
                status = self.get_status()
                if status != prev_status:
                    prev_status = status
                    console.log(f"status = {prev_status.value}")
                if status == TaskStatus.RUN:
                    break

        perc_done = 0
        field_decay = 0
        with Progress() as progress:

            task1 = progress.add_task("[red]% Done", total=100.0)
            task2 = progress.add_task("[green]Field Decay", total=1.0)
            for _ in web.run(task_id=self.task_id):
                time.sleep(0.01)
                task = web.monitor(task_id=self.task_id)
                run_info = task.run_info
                if run_info is None:
                    break

                perc_done_update = run_info.perc_done - perc_done
                perc_done = run_info.perc_done
                feld_decay_update = run_info.field_decay - field_decay
                field_decay = run_info.field_decay

                progress.update(task1, advance=perc_done_update)
                progress.update(task2, advance=feld_decay_update)

        prev_status = TaskStatus.POST

        console.log(f"status = {prev_status.value}")
        prev_status = TaskStatus.SUCCESS
        time.sleep(1)
        console.log(f"status = {prev_status.value}")

    def download(self, path: str) -> None:
        """download results."""
        web.download(task_id=self.task_id, path=path)

    def load_results(self, path: str) -> SimulationData:
        """download results and load them into SimulationData object."""
        if not os.path.exists(path):
            self.download(path=path)
        sim_data = SimulationData.load(path)
        return sim_data

    def delete(self):
        """delete server-side data associated with job"""
        web.delete(self.task_id)
        self.task_id = None


class Batch(WebContainer):
    """Holds a dictionary of jobs"""

    simulations: Dict[TaskName, Simulation]
    jobs: Dict[TaskName, Job] = None

    def __init__(self, **kwargs):
        """hacky way to create jobs if not supplied or use supplied ones if saved"""
        jobs = kwargs.get("jobs")
        if jobs is None:
            jobs = {}
            for task_name, simulation in kwargs["simulations"].items():
                job = Job(simulation=simulation, task_name=task_name)
                jobs[task_name] = job
            kwargs["jobs"] = jobs
        super().__init__(**kwargs)

    def upload(self) -> None:
        """upload simulations to server, record task ids"""
        for _, job in self.jobs.items():
            job.upload()

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """get general information about all job's task"""
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def run(self) -> None:
        """start running a task"""
        for _, job in self.jobs.items():
            job.run()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """get information about a running task"""
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self, mon_frequency=0.1) -> None:
        """monitor progress of running task"""
        console = Console()
        with console.status("[bold green]Monitoring Batch...") as status:
            for task_name, job in self.jobs.items():
                while True:
                    task = job.monitor()
                    time.sleep(mon_frequency)
                console.log(f"task '{task_name}' complete")

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir: str):
        """returns path of a job data given task name and directory path"""
        return os.path.join(path_dir, f"{str(task_id)}.hdf5")

    def download(self, path_dir: str) -> None:
        """download results."""
        # console = Console()
        # with console.status("[bold green]Downloading Batch...") as status:
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_name, path_dir)
            job.download(path=job_path)
            # console.log(f"task '{task_name}' downloaded")

    def load_results(self, path_dir: str) -> Dict[TaskName, SimulationData]:
        """download results and load them into SimulationData object."""
        sim_data_dir = {}
        self.download(path_dir=path_dir)
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            sim_data_dir[task_name] = sim_data
        return sim_data_dir

    def delete(self):
        """delete server-side data associated with job"""
        for _, job in self.jobs.items():
            job.delete()
            self.jobs = None

    def save(self, fname: str) -> None:
        """alias for self.export"""
        self.export(fname=fname)

    def items(self, path_dir: str):
        """simple iterator, `for task_name, sim_data in batch: `do something`"""
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            yield task_name, sim_data
