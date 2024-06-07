"""Defines parameter sweeping utilities for tidy3d."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas
import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel, cached_property
from ...web.api.container import BatchData


class Result(Tidy3dBaseModel):
    """Stores the result of a run over a ``DesignSpace``.
    Can be converted to ``pandas.DataFrame`` with ``Result.to_dataframe()`` for post processing.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> result = tdd.Result(
    ...     dims=('x', 'y', 'z'),
    ...     values=(1, 2, 3, 4),
    ...     coords=((0,1,2), (1,2,3), (2,3,4), (3,4,5)),
    ...     output_names=("output",),
    ... )
    >>> df = result.to_dataframe()
    >>> # df.head() # print out first 5 elements of data
    """

    dims: Tuple[str, ...] = pd.Field(
        (),
        title="Dimensions",
        description="The dimensions of the design variables (indexed by 'name').",
    )

    values: Tuple[Any, ...] = pd.Field(
        (),
        title="Values",
        description="The return values from the design problem function.",
    )

    coords: Tuple[Tuple[Any, ...], ...] = pd.Field(
        (),
        title="Coordinates",
        description="The values of the coordinates corresponding to each of the dims."
        "Note: shaped (D, N) where D is the ``len(dims)`` and N is the ``len(values)``",
    )

    output_names: Tuple[str, ...] = pd.Field(
        None,
        title="Output Names",
        description="Names for each of the outputs stored in ``values``. If not specified, default "
        "values are assigned.",
    )

    fn_source: str = pd.Field(
        None,
        title="Function Source Code",
        description="Source code for the function evaluated in the parameter sweep.",
    )

    task_ids: Tuple[Tuple[str, ...], ...] = pd.Field(
        None,
        title="Task IDs",
        description="Task IDs for the simulation run in each data point. Only available if "
        "the parameter sweep function is split into pre and post processing and run with "
        "'Design.run_batch()', otherwise is ``None``. "
        "To access all of the data, see ``batch_data``.",
    )

    batch_data: BatchData = pd.Field(
        None,
        title="Batch Data",
        description=":class:`BatchData` object storing all of the data for the simulations "
        " used in this ``Result``. Can be iterated through like a dictionary with "
        "``for task_name, sim_data in batch_data.items()``. Only available if "
        "the parameter sweep function is split into pre and post processing and run with "
        "'Design.run_batch()', otherwise is ``None``.",
    )

    @pd.validator("coords", always=True)
    def _coords_and_dims_shape(cls, val, values):
        """Make sure coords and dims have same size."""

        dims = values.get("dims")

        if val is None or dims is None:
            return

        num_dims = len(dims)
        for i, _val in enumerate(val):
            if len(_val) != num_dims:
                raise ValueError(
                    f"Number of 'coords' at index '{i}' ({len(_val)}) "
                    "doesn't match the number of 'dims' ({num_dims})."
                )

        return val

    @pd.validator("coords", always=True)
    def _coords_and_values_shape(cls, val, values):
        """Make sure coords and values have same length."""

        _values = values.get("values")

        if val is None or _values is None:
            return

        num_values = len(_values)
        num_coords = len(val)

        if num_values != num_coords:
            raise ValueError(
                f"'coords' and 'values' must have same number of elements. "
                f"Have {num_coords} and {num_values} elements, respectively."
            )

        return val

    def value_as_dict(self, value) -> Dict[str, Any]:
        """How to convert an output function value as a dictionary."""
        if isinstance(value, dict):
            return value
        keys = self.default_value_keys(value)
        if len(keys) == 1:
            return {keys[0]: value}
        return dict(zip(keys, value))

    @staticmethod
    def default_value_keys(value) -> Tuple[str, ...]:
        """The default keys for a given value."""

        # if a dict already, just use the existing keys as labels
        if isinstance(value, dict):
            return tuple(value.keys())

        # if array-like, ith output has key "output {i}"
        if isinstance(value, (tuple, list, np.ndarray)):
            return tuple(f"output {i}" for i in range(len(value)))

        # if simply single value (float, int, bool, etc) just label "output"
        return ("output",)

    def items(self) -> Tuple[dict, Any]:
        """Iterate through coordinates (args) and values (outputs) one by one."""

        for coord_tuple, val in zip(self.coords, self.values):
            coord_dict = dict(zip(self.dims, coord_tuple))
            yield coord_dict, val

    @cached_property
    def data(self) -> Dict[tuple, Any]:
        """Dict mapping tuple of fn args to their value."""

        result = {}
        for coord_dict, val in self.items():
            coord_tuple = tuple(coord_dict[dim] for dim in self.dims)
            result[coord_tuple] = val

        return result

    def get_value(self, coords: tuple) -> Any:
        """Get a data element indexing by function arg tuple."""
        return self.data[coords]

    def sel(self, **kwargs) -> Any:
        """Get a data element by function kwargs.."""
        coords_tuple = tuple(kwargs[dim] for dim in self.dims)
        return self.get_value(coords_tuple)

    def to_dataframe(self) -> pandas.DataFrame:
        """Data as a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            ``pandas.DataFrame`` corresponding to the ``Result``.
        """

        data = []
        for coord_dict, val in self.items():
            val_dict = self.value_as_dict(val)
            data_i = list(coord_dict.values()) + list(val_dict.values())
            data.append(data_i)

        val_keys = list(self.value_as_dict(self.values[0])) if self.values else [""]
        columns = list(self.dims) + val_keys

        df = pandas.DataFrame(data=data, columns=columns)

        attrs = dict(
            task_ids=self.task_ids,
            output_names=self.output_names,
            fn_source=self.fn_source,
            dims=self.dims,
        )

        df.attrs = attrs
        return df

    @classmethod
    def from_dataframe(cls, df: pandas.DataFrame, dims: List[str] = None) -> Result:
        """Load a result directly from a `pandas.DataFrame` object.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            ```DataFrame`` object to load into a :class:`.Result`.
        dims : List[str] = None
            Set of dimensions corresponding to the function arguments.
            Not required if this dataframe was generated directly from a :class:`.Result`
            without modification. In that case, it contains the dims in its ``.attrs`` metadata.

        Returns
        -------
        :class:`.Result`
            Result loaded from this ``DataFrame``.
        """

        attrs = df.attrs

        # get dims either from kwarg or from attrs, error if dims not provided
        if dims is None:
            if "dims" in attrs:
                dims = attrs["dims"]
            else:
                raise ValueError(
                    "'dims' neither supplied or found in the 'DataFrame.attrs'. "
                    "Can't process 'DataFrame' into 'sweep.Results'."
                )

        # grab the columns of the data, sorted into coords and values
        coords_list = []
        values_list = []
        for key in df.keys():
            data_column = df[key]
            if key in dims:
                coords_list.append(data_column)
            else:
                values_list.append(data_column)

        # transpose the data and store in Result along with the other attrs (if present)
        coords = np.array(coords_list).T.tolist()
        values = np.array(values_list).T.tolist()
        return cls(
            dims=dims,
            coords=coords,
            values=values,
            output_names=attrs.get("output_names"),
            task_ids=attrs.get("task_ids"),
            fn_source=attrs.get("fn_source"),
        )

    def combine(self, other: Result) -> Result:
        """Combine data from two results into a new result (also works with '+').

        Parameters
        ----------
        other : :class:`.Result`
            Results to combine with this object.

        Returns
        -------
        :class:`.Result`
            Combined :class:`.Result`
        """

        if self.fn_source != other.fn_source:
            raise ValueError("Can't combine results, function sources don't match.")

        if self.output_names != other.output_names:
            raise ValueError("Can't combine results, output names don't match.")

        if self.dims != other.dims:
            raise ValueError("Can't combine results, dimensions don't match.")

        def combine_tuples(tuple1: tuple, tuple2: tuple):
            """Combine two tuples together if not None."""
            if tuple1 is None and tuple2 is None:
                return None
            if (tuple1 is None) != (tuple2 is None):
                raise ValueError("Can't combine data where one only one field is `None`.")
            return list(tuple1) + list(tuple2)

        task_ids = combine_tuples(self.task_ids, other.task_ids)
        coords = combine_tuples(self.coords, other.coords)
        values = combine_tuples(self.values, other.values)

        return Result(
            dims=self.dims,
            coords=coords,
            values=values,
            output_names=self.output_names,
            fn_source=self.fn_source,
            task_ids=task_ids,
        )

    def __add__(self, other):
        """Special syntax for design_result1 + design_result2."""
        return self.combine(other)

    def get_index(self, fn_args: Dict[str, float]) -> int:
        """Get index into the data for a specific set of arguments."""

        key_list = list(self.coords)
        arg_key = tuple(fn_args[dim] for dim in self.dims)
        return key_list.index(arg_key)

    def delete(self, fn_args: Dict[str, float]) -> Result:
        """Delete a specific set of arguments from the result.

        Parameters
        ----------
        fn_args : Dict[str, float]
            ``dict`` containing the function arguments one wishes to delete.

        Returns
        -------
        :class:`.Result`
            Copy of the result with that element removed.
        """

        index = self.get_index(fn_args)

        new_coords = list(self.coords)
        new_values = list(self.values)

        new_coords.pop(index)
        new_values.pop(index)

        return self.updated_copy(values=new_values, coords=new_coords)

    def add(self, fn_args: Dict[str, float], value: Any) -> Result:
        """Add a specific argument and value the result.

        Parameters
        ----------
        fn_args : Dict[str, float]
            ``dict`` containing the function arguments one wishes to add.
        value : Any
            Data point value corresponding to these arguments.

        Returns
        -------
        :class:`.Result`
            Copy of the result with that element added.
        """

        new_values = list(self.values) + [value]
        new_coords = list(self.coords) + [tuple(fn_args[dim] for dim in self.dims)]

        return self.updated_copy(values=new_values, coords=new_coords)
