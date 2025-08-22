from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.simulation_types import SimulationType


class SimulationMap(Tidy3dBaseModel, Mapping[str, SimulationType]):
    """Container for a set of simulations, accessible by a string index."""

    index: tuple[str, ...]
    """Tuple of unique string identifiers for each simulation."""

    simulation: tuple[SimulationType, ...]
    """Tuple of :class:`.Simulation` objects corresponding to each index."""

    @pd.root_validator()
    def _validate_lengths_match(cls, values):
        """Validate that index and simulation have the same length."""
        index, simulation = values.get("index"), values.get("simulation")
        if index is not None and simulation is not None and len(index) != len(simulation):
            raise ValueError("Length of 'index' and 'simulation' must be the same.")
        return values

    def __getitem__(self, index: Union[str, int]) -> SimulationType:
        """Allows retrieving a simulation by its index name.

        Parameters
        ----------
        index : Union[str, int]
            The string name or integer index of the simulation to look up.

        Returns
        -------
        SimulationType
            The :class:`.Simulation` object corresponding to the given index.

        Raises
        ------
        KeyError
            If no simulation with the given index name is found.
        """
        for i, index_i in enumerate(self.index):
            if index_i == index:
                return self.simulation[i]
        raise KeyError(f"Index '{index}' not found.")
