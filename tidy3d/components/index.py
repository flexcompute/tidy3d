"""
This module defines the SimulationMap, a specialized container for managing
a set of Tidy3D simulations, allowing them to be accessed by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.simulation import SimulationType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tidy3d.compat import Self


class ValueMap(Tidy3dBaseModel, Mapping[str, Any]):
    """An immutable dictionary-like container for objects.

    Notes
    -----
        This class maps unique string keys to corresponding value objects.
        By inheriting from `collections.abc.Mapping`, it provides standard dictionary
        behaviors like item access (`my_dict["my_key"]`), iteration (`for name in my_dict`), and
        length checking (`len(my_dict)`).

        It automatically validates that the `keys` and `values`
        tuples have matching lengths upon instantiation.

    Attributes
    ----------
    keys : tuple[str, ...]
        A tuple of unique string identifiers for each simulation.
    values : tuple[Any, ...]
        A tuple of `Any`-type objects, each corresponding to a key at the
        same index. Should be overwritten by the subclass instantiation
    """

    keys_tuple: tuple[str, ...] = Field(
        description="A tuple of unique string identifiers for each simulation.", alias="keys"
    )
    values_tuple: tuple[Any, ...] = Field(
        description=(
            "A tuple of `Simulation` objects, each corresponding to a key at the same index."
        ),
        alias="values",
    )

    @model_validator(mode="after")
    def _validate_lengths_match(self) -> Self:
        """Ensure 'keys' and 'values' have the same length.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If the lengths of the 'keys' and 'values' tuples are not equal.
        """
        keys, values = self.keys_tuple, self.values_tuple
        if len(keys) != len(values):
            raise ValueError("Length of 'keys' and 'values' must be the same.")
        return self

    def __getitem__(self, key: str) -> Any:
        """Retrieves a `Simulation` object by its corresponding key.

        This allows for dictionary-style access, e.g., `my_map["my_key"]`.

        Parameters
        ----------
        key : str
            The string name of the item to retrieve.

        Returns
        -------
        SimulationType
            The object corresponding to the given key.

        Raises
        ------
        KeyError
            If no object with the given key name is found in the map.
        """
        for i, current_key in enumerate(self.keys_tuple):
            if current_key == key:
                return self.values_tuple[i]
        raise KeyError(f"Key '{key}' not found in the {type(self)}.")

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
        return len(self.keys_tuple)


class SimulationMap(ValueMap, Mapping[str, SimulationType]):
    """An immutable dictionary-like container for simulations.

    Notes
    -----
        This class maps unique string keys to corresponding `Simulation` objects.
        By inheriting from `collections.abc.Mapping`, it provides standard dictionary
        behaviors like item access (`sims["sim_A"]`), iteration (`for name in sims`), and
        length checking (`len(sims)`).

        It automatically validates that the `keys` and `values`
        tuples have matching lengths upon instantiation.

    Attributes
    ----------
    keys : tuple[str, ...]
        A tuple of unique string identifiers for each simulation.
    values : tuple[SimulationType, ...]
        A tuple of `Simulation` objects, each corresponding to a key at the
        same index.

    Example
    -------
    >>> from tidy3d import (
    ...     Simulation,
    ...     SimulationMap,
    ...     Structure,
    ...     Box,
    ...     Medium,
    ...     UniformCurrentSource,
    ...     GaussianPulse,
    ...     FieldMonitor,
    ...     GridSpec,
    ...     BoundarySpec,
    ...     PML,
    ... )
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
    ...             current_amplitude_definition="total",
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
    >>> # Instantiate the map
    >>> simulation_map = SimulationMap(
    ...     keys=("sim_1", "sim_2"),
    ...     values=(sim1, sim2),
    ... )
    >>>
    >>> # Access a simulation like a dictionary
    >>> # print(simulation_map["sim_1"])
    """

    keys_tuple: tuple[str, ...] = Field(
        description="A tuple of unique string identifiers for each simulation.", alias="keys"
    )
    values_tuple: tuple[discriminated_union(SimulationType), ...] = Field(
        description=(
            "A tuple of `Simulation` objects, each corresponding to a key at the same index."
        ),
        alias="values",
    )
