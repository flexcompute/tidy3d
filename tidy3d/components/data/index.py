"""
This module defines the SimulationDataMap, a specialized container for storing and
accessing simulation data results from a Tidy3D simulation.
"""

from __future__ import annotations

from collections.abc import Mapping

import pydantic.v1 as pd

from tidy3d.components.index import ValueMap
from tidy3d.components.types.simulation import SimulationDataType


class SimulationDataMap(ValueMap, Mapping[str, SimulationDataType]):
    """An immutable dictionary-like container for simulation data.

    Notes
    -----
        It provides standard dictionary
        behaviors like item access (`data["key"]`), iteration (`for key in data`), and
        length checking (`len(data)`).

        It automatically validates that the `keys` and `values`
        tuples have matching lengths upon instantiation.

    Attributes
    ----------
    keys : tuple[str, ...]
        A tuple of unique string identifiers for each simulation data object.
    values : tuple[SimulationDataType, ...]
        A tuple of `SimulationDataType` objects, each corresponding to a key at the
        same index.

    Example
    -------
    >>> from tidy3d import (
    ...     Simulation,
    ...     SimulationData,
    ...     SimulationDataMap,
    ...     Structure,
    ...     Box,
    ...     Medium,
    ...     UniformCurrentSource,
    ...     GaussianPulse,
    ...     FieldMonitor,
    ...     GridSpec,
    ...     BoundarySpec,
    ...     Boundary,
    ...     PML,
    ... )
    >>> import tidy3d as td
    >>> # Simple minimal simulation
    >>> sim1 = Simulation(
    ...     size=(4, 3, 3),
    ...     grid_spec=GridSpec.auto(min_steps_per_wvl=25),
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         UniformCurrentSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             source_time=GaussianPulse(freq0=2e14, fwidth=4e13),
    ...         )
    ...     ],
    ...     monitors=[
    ...         FieldMonitor(
    ...             size=(1, 1, 0),
    ...             center=(0, 0, 0),
    ...             freqs=[2e14],
    ...             name='field'
    ...         ),
    ...     ],
    ...     run_time=1e-12,
    ...     boundary_spec=BoundarySpec.all_sides(boundary=PML()),
    ... )
    >>> sim2 = sim1.updated_copy(run_time=2e-12)
    >>>
    >>> sim_data_1 = td.SimulationData(
    ...     simulation=sim1,
    ...     data=()  # Empty tuple for minimal case
    ... )
    >>> sim_data_2 = td.SimulationData(
    ...     simulation=sim2,
    ...     data=()  # Empty tuple for minimal case
    ... )
    >>> # Instantiate the map
    >>> simulation_data_map = SimulationDataMap(
    ...     keys=("data_1", "data_2"),
    ...     values=(sim_data_1, sim_data_2),
    ... )
    >>>
    >>> # Access a simulation data like a dictionary
    >>> # print(simulation_data_map["data_2"])
    """

    keys_tuple: tuple[str, ...] = pd.Field(
        description="A tuple of unique string identifiers for each simulation data object.",
        alias="keys",
    )
    values_tuple: tuple[SimulationDataType, ...] = pd.Field(
        description=(
            "A tuple of `SimulationDataType` objects, each corresponding to a key at the "
            "same index."
        ),
        alias="values",
    )
