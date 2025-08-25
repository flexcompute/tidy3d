"""
This module defines the SimulationDataMap, a specialized container for storing and
accessing simulation data results from a Tidy3D simulation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import SimulationDataType


class SimulationDataMap(Tidy3dBaseModel, Mapping[str, SimulationDataType]):
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

    @pd.root_validator(skip_on_failure=True)
    def _validate_lengths_match(cls, data: dict) -> dict:
        """Pydantic root validator to ensure 'keys' and 'values' have the same length.

        Parameters
        ----------
        data : dict
            The dictionary of field values for the model provided by Pydantic.

        Returns
        -------
        dict
            The validated dictionary of field values.

        Raises
        ------
        ValueError
            If the lengths of the 'keys' and 'values' tuples are not equal.
        """
        keys, values = data.get("keys_tuple"), data.get("values_tuple")
        if len(keys) != len(values):
            raise ValueError("Length of 'keys_tuple' and 'values_tuple' must be the same.")
        return data

    def __getitem__(self, key: str) -> SimulationDataType:
        """Retrieves a `SimulationDataType` object by its corresponding key.

        This allows for dictionary-style access, e.g., `my_map["monitor_name"]`.

        Parameters
        ----------
        key : str
            The string name of the simulation data to retrieve.

        Returns
        -------
        SimulationDataType
            The `SimulationDataType` object corresponding to the given key.

        Raises
        ------
        KeyError
            If no simulation data with the given key is found in the map.
        """
        for i, current_key in enumerate(self.keys_tuple):
            if current_key == key:
                return self.values_tuple[i]
        raise KeyError(f"Key '{key}' not found in the SimulationDataMap.")

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the string keys of the map.

        This allows for standard iteration, e.g., `for key in my_map:`.

        Yields
        ------
        str
            The next key in the map.
        """
        return iter(self.keys_tuple)

    def __len__(self) -> int:
        """Returns the number of key-value pairs stored in the map.

        This allows for using the built-in `len()` function, e.g., `len(my_map)`.

        Returns
        -------
        int
            The total number of items in the map.
        """
        assert len(self.keys_tuple) == len(self.values_tuple), "Internal state mismatch."
        return len(self.keys_tuple)
