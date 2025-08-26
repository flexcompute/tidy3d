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

    Iit provides standard dictionary
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
    >>> from tidy3d import FieldData, FluxData
    >>> import numpy as np
    >>> # Create some dummy simulation data
    >>> field_data = FieldData(data=np.random.rand(2, 2, 2, 3, 2), f="field")
    >>> flux_data = FluxData(data=np.random.rand(10), f="flux")
    >>>
    >>> # Instantiate the map
    >>> sim_data_map = SimulationDataMap(
    ...     keys=("field_monitor", "flux_monitor"),
    ...     values=(field_data, flux_data),
    ... )
    >>>
    >>> # Access data like a dictionary
    >>> print(sim_data_map["field_monitor"])
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
