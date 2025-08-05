"""BatchData class for handling batch simulation results."""

from __future__ import annotations

from collections.abc import Mapping

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.web.api import webapi as web
from tidy3d.web.api.tidy3d_stub import SimulationDataType
from tidy3d.web.core.constants import TaskName

# Default data directory for batch operations
DEFAULT_DATA_DIR = "."


class BatchData(Tidy3dBaseModel, Mapping):
    """
    Holds a collection of :class:`.SimulationData` returned by :class:`Batch`.

    Notes
    -----

        When the batch is completed, the output is not a :class:`.SimulationData` but rather a :class:`BatchData`. The
        data within this :class:`BatchData` object can either be indexed directly ``batch_results[task_name]`` or can be looped
        through ``batch_results.items()`` to get the :class:`.SimulationData` for each task.

    See Also
    --------

    :class:`Batch`:
         Interface for submitting several :class:`.Simulation` objects to sever.

    :class:`.SimulationData`:
         Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`.

    **Notebooks**
        * `Running simulations through the cloud <../../notebooks/WebAPI.html>`_
        * `Performing parallel / batch processing of simulations <../../notebooks/ParameterScan.html>`_
    """

    task_paths: dict[TaskName, str] = pd.Field(
        ...,
        title="Data Paths",
        description="Mapping of task_name to path to corresponding data for each task in batch.",
    )

    task_ids: dict[TaskName, str] = pd.Field(
        ..., title="Task IDs", description="Mapping of task_name to task_id for each task in batch."
    )

    verbose: bool = pd.Field(
        True, title="Verbose", description="Whether to print info messages and progressbars."
    )

    def load_sim_data(self, task_name: str) -> SimulationDataType:
        """Load a simulation data object from file by task name."""
        task_data_path = self.task_paths[task_name]
        task_id = self.task_ids[task_name]
        web.get_info(task_id)

        return web.load(task_id=task_id, path=task_data_path, verbose=False)

    def __getitem__(self, task_name: TaskName) -> SimulationDataType:
        """Get the simulation data object for a given ``task_name``."""
        return self.load_sim_data(task_name)

    def __iter__(self):
        """Iterate over the task names."""
        return iter(self.task_paths)

    def __len__(self):
        """Return the number of tasks in the batch."""
        return len(self.task_paths)

    @classmethod
    def load(cls, path_dir: str = DEFAULT_DATA_DIR, replace_existing: bool = False) -> BatchData:
        """Load :class:`Batch` from file, download results, and load them.

        Parameters
        ----------
        path_dir : str = './'
            Base directory where data will be downloaded, by default current working directory.
            A `batch.hdf5` file must be present in the directory.
        replace_existing : bool = False
            Downloads the data even if path exists (overwriting the existing).

        Returns
        -------
        :class:`BatchData`
            Contains Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
            for each Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] in :class:`Batch`.
        """

        # Import here to avoid circular import
        from tidy3d.web.api.container import Batch

        batch_file = Batch._batch_path(path_dir=path_dir)
        batch = Batch.from_file(batch_file)
        return batch.load(path_dir=path_dir, replace_existing=replace_existing)
