from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.simulation_types import SimulationDataType


class IndexSimulationData(Tidy3dBaseModel):
    """Container for a set of simulation data, accessible by a string index."""

    index: tuple[str, ...]
    """Tuple of unique string identifiers for each simulation data object."""

    data: tuple[SimulationDataType, ...]
    """Tuple of :class:`.SimulationData` objects corresponding to each index."""

    @pd.root_validator()
    def _validate_lengths_match(cls, values):
        """Validate that index and data have the same length."""
        index, data = values.get("index"), values.get("data")
        if index is not None and data is not None and len(index) != len(data):
            raise ValueError("Length of 'index' and 'data' must be the same.")
        return values

    def __getitem__(self, index: str) -> SimulationDataType:
        """Allows retrieving simulation data by its index name.

        Parameters
        ----------
        index : str
            The string name of the simulation data to look up.

        Returns
        -------
        SimulationDataType
            The :class:`.SimulationData` object corresponding to the given index name.

        Raises
        ------
        KeyError
            If no simulation data with the given index name is found.
        """
        for i, index_i in enumerate(self.index):
            if index_i == index:
                return self.data[i]
        raise KeyError(f"Index '{index}' not found.")
