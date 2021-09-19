"""Summary
"""
from tqdm import tqdm
import time

from . import webapi as web
from .job import Job
from ..simulation import Simulation


class Batch:
    """Container for processing set of simulations.
    
    Attributes
    ----------
    jobs : list
        List of ``tidy3d.web.Job`` objects containing jobs to be processed.
    num_jobs : int
        Number of jobs in the :class:`.Batch`.
    """

    def __init__(
        self,
        simulations,
        task_ids=None,
        task_names=None,
        base_dir="out/",
        folder_name="default",
        draft=False,
    ):
        """Construct.
        
        Parameters
        ----------
        simulations : list
            List of :class:`.Simulation` to process in :class:`.Batch`.
        task_ids : None, optional
            List of unique ID of task, can be found either on UI or through ``job.task_id``.
        task_names : list, optional
            List of strings corresponding to task name of each :class:`.Simulation`.
        base_dir : str, optional
            Base of path corresponding to where data for this :class:`.Batch` will be stored.
        folder_name : str, optional
            Server folder to hold the :class:`.Batch`.
        draft : bool, optional
            If ``True``, each job will be submitted but not run. It can then
            be visualized in the web UI and run from there when needed.
        """
        self.num_jobs = len(simulations)
        self.jobs = []
        for job_index in range(self.num_jobs):
            sim = simulations[job_index]
            task_name = None if task_names is None else task_names[job_index]
            task_id = None if task_ids is None else task_ids[job_index]
            new_job = Job(
                simulation=sim,
                task_id=task_id,
                task_name=task_name,
                base_dir=base_dir,
                folder_name=folder_name,
                draft=draft,
            )
            self.jobs.append(new_job)

    @classmethod
    def load_from_file(
        cls,
        filename,
        task_names=None,
        base_dir="out/",
        folder_name="default",
        draft=False,
    ):
        """Load a :class:`.Batch` from file containing newline separated list of ``task_ids``.
        
        Parameters
        ----------
        filename : str
            Full path to file for storing :class:`.Batch` info.
        task_names : list, optional
            List of strings corresponding to task name of each :class:`.Simulation`.
        base_dir : str, optional
            Base of path corresponding to where data for this :class:`.Batch` will be stored.
        folder_name : str, optional
            Folder to store the :class:`.Batch` on the server / UI.
        draft : bool, optional
            If ``True``, each job will be submitted but not run. It can then
            be visualized in the web UI and run from there when needed.
        
        No Longer Returned
        ------------------
        Batch
            :class:`.web.Batch` containing all Jobs from file.
        """
        task_ids = []
        with open(filename, "r") as f:
            for line in f:
                task_ids.append(str(line.strip()))
        return Batch.load_from_task_ids(
            task_ids,
            task_names=task_names,
            base_dir=base_dir,
            folder_name=folder_name,
            draft=draft,
        )

    @classmethod
    def load_from_task_ids(
        cls,
        task_ids,
        task_names=None,
        base_dir="out/",
        folder_name="default",
        draft=False,
    ):
        """Load a :class:`.Batch` from file containing newline separated list of ``task_ids``.
        
        Parameters
        ----------
        task_ids : list
            List of ``task_id`` strings to load into :class:`.Batch`.
        task_names : list, optional
            List of strings corresponding to ``task_name`` of each :class:`.Simulation`.
        base_dir : str, optional
            Base of path corresponding to where data for this batch will be stored.
        folder_name : str, optional
            Folder to store the :class:`.Batch` on the server / UI.
        draft : bool, optional
            If ``True``, each job will be submitted but not run. It can then
            be visualized in the web UI and run from there when needed.        
        
        No Longer Returned
        ------------------
        Batch
            :class:`.web.Batch` containing all jobs from list of ``task_ids``.
        """
        sims = []
        for job_index in range(len(task_ids)):
            task_id = task_ids[job_index]
            json = web.download_json(task_id, target_folder=base_dir)
            fname = base_dir + "simulation.json"
            sim = Simulation.import_json(base_dir + "simulation.json")
            sims.append(sim)
        return Batch(
            sims,
            task_ids=task_ids,
            task_names=task_names,
            base_dir=base_dir,
            folder_name=folder_name,
            draft=draft,
        )

    def get_info(self):
        """Return list of dictionaries storing :class:`.Batch` metadata.
        
        Returns
        -------
        list
            List of dictionaries storing metadata of jobs in :class:`.Batch`.
        """
        info_dicts = []
        for job_index in range(self.num_jobs):
            new_info_dict = self.jobs[job_index].get_info()
            info_dicts.append(new_info_dict)
        return info_dicts

    def monitor(self):
        """Monitor progress of :class:`.Batch` in terms of number of jobs completed.
        """
        done_states = ("success", "error", "diverged", "deleted", "draft")
        with tqdm(total=self.num_jobs) as pbar:
            pbar.set_description("Percentage of jobs completed: ")
            num_done_prev = 0
            while num_done_prev < self.num_jobs:
                info_dicts = self.get_info()
                statuses = [dic["status"] for dic in info_dicts]
                time.sleep(0.3)
                num_done = sum([status in done_states for status in statuses])
                # note: pbar.update(n) moves bar *forward* n values
                # so to update it to num_done, need to feed it the difference
                pbar.update(num_done - num_done_prev)
                num_done_prev = num_done

    def save(self, filename):
        """Save :class:`.Batch` info to file of newline-separated ``task_ids``.
        :class:`.Batch` may be loaded from this file by ``Batch.load_from_file(filename)``.
        
        Parameters
        ----------
        filename : str
            Full path to file to store ``task_id`` data in.
        """
        with open(filename, "w") as f:
            for job in self.jobs:
                f.write(str(job.task_id) + "\n")

    def delete(self):
        """Delete all jobs in :class:`.Batch`.
        """
        for job_index in range(self.num_jobs):
            self.jobs[job_index].delete()

    def load_results(self):
        """downloads results and either loads into list of simulations or returns list of new simulations
        
        Returns
        -------
        list
            List of :class:`.Simulations` containing loaded results.
        """
        sims_loaded = []
        for job_index in range(self.num_jobs):
            print(f"\nloading results for job ({job_index+1}/{self.num_jobs})\n")
            sim_new = self.jobs[job_index].load_results()
            sims_loaded.append(sim_new)
        return sims_loaded

    def download_json(self):
        """Downalod json file of each :class:`.Job` to it's corresponding ``target_folder``.
        """
        for job_index in range(self.num_jobs):
            print(f"\ndownloading json for job ({job_index+1}/{self.num_jobs})\n")
            self.jobs[job_index].download_json()

    def download_results(self):
        """Downalod data of each :class:`.Job` to it's corresponding ``target_folder``.
        """
        for job_index in range(self.num_jobs):
            print(f"\ndownloading results for job ({job_index+1}/{self.num_jobs})\n")
            self.jobs[job_index].download_results()
