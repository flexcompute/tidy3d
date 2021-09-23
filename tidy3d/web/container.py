import pydantic
import os

from .task import TaskId, TaskInfo, RunInfo

from .. import web
from ..components import Simulation, SimulationData
from ..components.types import Dict
from ..components.base import Tidy3dBaseModel

""" higher level wrappers for webapi functions for individual (Job) and batch (Batch) tasks. """

class WebContainer(Tidy3dBaseModel):
    pass

    # def export(self, fname: str) -> None:
    #     json_string = self.json(indent=2)
    #     with open(fname, "w") as fp:
    #         fp.write(json_string)

    # @classmethod
    # def load(cls, fname: str):
    #     return cls.parse_file(fname)

# type of task_name
TaskName = str

class Job(WebContainer):

    simulation: Simulation
    task_name: TaskName
    task_id: TaskId = None

    def upload(self) -> None:
        """ upload simulation to server, record task id"""
        task_id = web.upload(simulation=self.simulation)
        self.task_id = task_id

    def get_info(self) -> TaskInfo:
        """ return general information about a job's task"""
        task_info = web.get_info(task_id=self.task_id)
        return task_info

    def run(self) -> None:
        """ start running a task"""
        web.run(self.task_id)

    def get_run_info(self) -> RunInfo:
        """ get information about a running task"""
        run_info = web.get_run_info(task_id=self.task_id)
        return run_info

    def monitor(self) -> None:
        """ monitor progress of running task"""
        web.monitor(task_id=self.task_id)

    def download(self, path: str) -> None:
        """ download results."""
        web.download(task_id=self.task_id, path=path)

    def load_results(self, path: str) -> SimulationData:
        """ download results and load them into SimulationData object."""
        if not os.path.exists(path):
            self.download(path=path)
        sim_data = SimulationData.load(path)
        return sim_data

    def delete(self):
        """ delete server-side data associated with job"""
        web.delete(self.task_id)
        self.task_id = None

class Batch(WebContainer):

    simulations: Dict[TaskName, Simulation]
    jobs: Dict[TaskName, Job] = None

    def __init__(self, **kwargs):
        """ hacky way to create jobs if not supplied or use supplied ones if saved"""
        jobs = kwargs.get('jobs')
        if jobs is None:
            jobs = {}
            for task_name, simulation in kwargs['simulations'].items():
                job = Job(simulation=simulation, task_name=task_name)
                jobs[task_name] = job
            kwargs['jobs'] = jobs
        super().__init__(**kwargs)

    def upload(self) -> None:
        """ upload simulations to server, record task ids"""
        for task_name, job in self.jobs.items():
            job.upload()

    def get_info(self) -> Dict[TaskName, TaskInfo]:
        """ get general information about all job's task"""
        info_dict = {}
        for task_name, job in self.jobs.items():
            task_info = job.get_info()
            info_dict[task_name] = task_info
        return info_dict

    def run(self) -> None:
        """ start running a task"""
        for _, job in self.jobs.items():
            job.run()

    def get_run_info(self) -> Dict[TaskName, RunInfo]:
        """ get information about a running task"""
        run_info_dict = {}
        for task_name, job in self.jobs.items():
            run_info = job.get_run_info()
            run_info_dict[task_name] = run_info
        return run_info_dict

    def monitor(self) -> None:
        """ monitor progress of running task"""
        for task_name, job in self.jobs.items():
            print(f'\nmonitoring task: {task_name}')
            job.monitor()

    @staticmethod
    def _job_data_path(task_id: TaskId, path_dir:str):
        """ returns path of a job data given task name and directory path"""
        return os.path.join(path_dir, f'{str(task_id)}.hdf5')

    def download(self, path_dir: str) -> None:
        """ download results."""
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_name, path_dir)
            job.download(path=job_path)

    def load_results(self, path_dir: str) -> Dict[TaskName, SimulationData]:
        """ download results and load them into SimulationData object."""
        sim_data_dir = {}
        self.download(path_dir=path_dir)
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            sim_data_dir[task_name] = sim_data
        return sim_data_dir

    def delete(self):
        """ delete server-side data associated with job"""
        for task_name, job in self.jobs.items():
            job.delete()
            self.jobs = None

    def save(self, fname: str) -> None:
        # alias for export
        self.export(fname=fname)

    def items(self, path_dir: str):
        """ simple iterator, `for task_name, sim_data in batch: `do something`"""
        for task_name, job in self.jobs.items():
            job_path = self._job_data_path(task_id=job.task_id, path_dir=path_dir)
            sim_data = job.load_results(path=job_path)
            yield task_name, sim_data
