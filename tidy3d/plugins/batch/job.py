from . import webapi as web


class Job:
    """Container for single simulation and associated task.
    
    Attributes
    ----------
    simulation : Simulation
        :class:`.Simulation` being managed by the Job.
    target_folder : str
        Path to where this job stores its data.
        ``base_dir`` followed by ``task_id``.
    task_id : str
        Unique ID of task, can be found either on UI or through ``job.task_id``.
    """

    def __init__(
        self,
        simulation,
        base_dir="out/",
        task_id=None,
        task_name=None,
        folder_name="default",
        draft=False,
    ):
        """Construct.
        
        Parameters
        ----------
        simulation : Simulation
            :class:`.Simulation` object to run.
        base_dir : str, optional
            Base of path corresponding to where data for this job will be stored.
        task_id : None, optional
            Unique ID of task, can be found either on UI or through job.task_id.
        task_name : str, optional
            Custom name for the job.
        folder_name : str, optional
            Server folder to hold the job.
        draft : bool, optional
            If ``True``, the job will be submitted but not run. It can then
            be visualized in the web UI and run from there when needed.
        """

        self.simulation = simulation
        sim_dict = simulation.export()
        if task_id is None:
            # submit a task
            info_dict = web.new_project(
                sim_dict, task_name=task_name, folder_name=folder_name, draft=draft
            )
            self.task_id = info_dict["taskId"]
        else:
            # use task_id
            self.task_id = task_id
        self.target_folder = base_dir.replace("/", "") + "/" + str(self.task_id)

    @classmethod
    def load_from_task_id(
        cls, task_id, base_dir="out/", task_name=None, folder_name="default"
    ):
        """Loads a Job from it's task id.
        
        Parameters
        ----------
        task_id : str
            Unique ID of task, can be found either on UI or through job.task_id.
        base_dir : str, optional
            Base path corresponding to where data for this job will be stored.
            Data is stored in ``base_dir/task_id``.
        task_name : str, optional
            Custom name for the task.
        folder_name : str, optional
            Server folder to hold the task.
        
        Returns
        ------------------
        Job
            :class:`.web.Job` instance.
        """
        json = web.download_json(task_id, target_folder=base_dir)
        fname = base_dir + "simulation.json"
        sim = Simulation.import_json(base_dir + "simulation.json")
        return Job(
            sim,
            task_id=task_id,
            base_dir=base_dir,
            task_name=task_name,
            folder_name=folder_name,
        )

    """ convenience methods """

    def _get_target_folder(self, target_folder):
        """Get target folder for this :class:`.Job`.
        If it was specified use that, otherwise use default.
        
        Parameters
        ----------
        target_folder : str
            specified ``target_folder``.
        
        Returns
        -------
        str
            Path to target folder for storing job.
        """
        return self.target_folder if target_folder is None else target_folder

    """ core methods in the API """

    def get_info(self):
        """Returns dictionary containing :class:`.Job` metadata.
        
        Returns
        -------
        dict
            Dictionary storing job metadata.
        """
        return web.get_project(self.task_id)

    def delete(self):
        """Deletes the :class:`.Job` from our server.
        
        Returns
        -------
        dict
            Dictionary storing :class:`.Job` metadata.
        """
        return web.delete_project(self.task_id)

    def monitor(self):
        """Prints status of :class:`.Job` in real time.
        """
        web.monitor_project(self.task_id)

    def load_results(self, target_folder=None):
        """Downloads data to ``target_folder``, loads data into either saved simulation or newly created simulation.
        
        Parameters
        ----------
        target_folder : str, optional
            Path to where this job stores its data.
        
        Returns
        -------
        Simulation
            :class:`.Simulation` with stored data.
        """
        target_folder = self._get_target_folder(target_folder)
        web.download_results(self.task_id, target_folder=target_folder)
        sim_loaded = web.load(self.task_id, self.simulation, target_folder)
        return sim_loaded

    def download_json(self, target_folder=None):
        """Download the json file associated with this job to ``target_folder``.
        
        Parameters
        ----------
        target_folder : None, optional
            Path to where this job stores its data.
        """
        target_folder = self._get_target_folder(target_folder)
        web.download_json(self.task_id, target_folder=target_folder)

    def download_results(self, target_folder=None):
        """Download the job results to ``target_folder``.
        
        Parameters
        ----------
        target_folder : str, optional
            Path to where this job stores its data.
        """
        target_folder = self._get_target_folder(target_folder)
        web.download_results(self.task_id, target_folder=target_folder)

    """ methods we might want to implememnt sometime """

    def _pause(self):
        """Pauses job running on server.
        """
        pass

    def _resume(self):
        """Resumes job running on server.
        """
        pass
