""" Batch Level Data """
from typing import Dict

from .base import Tidy3dData
from .sim_data import SimulationData

# TODO: iterating through batch data
# TODO: saving and loading from hdf5 group or json file
# TODO: does equality checing work?
# TODO: docstring examples?
# TODO: merge somehow with web.container.BatchData (where to define?)


class BatchData(Tidy3dData):
    """Collection of :class:`.SimulationData` objects."""

    sim_data_dict: Dict[str, SimulationData]
