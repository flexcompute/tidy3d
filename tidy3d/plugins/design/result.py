"""Defines parameter sweeping utilities for tidy3d."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas
import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel, cached_property

# NOTE: Coords are args_dict from method and design. This may be changed in future to unify naming


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

    task_names: list = pd.Field(
        None,
        title="Task Names",
        description="Task name of every simulation run during ``DesignSpace.run``. Only available if "
        "the parameter sweep function is split into pre and post processing, otherwise is ``None``. "
        "Stored in the same format as the output of fn_pre i.e. if pre outputs a dict, this output is a dict with the keys preserved.",
    )

    task_paths: list = pd.Field(
        None,
        title="Task Paths",
        description="Task paths of every simulation run during ``DesignSpace.run``. Useful for loading download ``SimulationData`` hdf5 files."
        "Only available if the parameter sweep function is split into pre and post processing, otherwise is ``None``. "
        "Stored in the same format as the output of fn_pre i.e. if pre outputs a dict, this output is a dict with the keys preserved.",
    )

    aux_values: Tuple[Any, ...] = pd.Field(
        None,
        title="Auxiliary values output from the user function",
        description="The auxiliary return values from the design problem function. This is the collection of objects returned "
        "alongside the float value used for the optimization. These weren't used to inform the optimizer, if one was used.",
    )

    optimizer: Any = pd.Field(
        None,
        title="Optimizer object",
        description="The optimizer returned at the end of an optimizer run. Can be used to analyze and plot how the optimization progressed. "
        "Attributes depend on the optimizer used; a full explaination of the optimizer can be found on associated library doc pages. Will be ``None`` for sampling based methods.",
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
                    f"doesn't match the number of 'dims' ({num_dims})."
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
            return tuple(f"output_{i}" for i in range(len(value)))

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

    def to_dataframe(self, include_aux: bool = False) -> pandas.DataFrame:
        """Data as a ``pandas.DataFrame``.

        Output a ``pandas.DataFrame`` of the ``Result``. Can include auxiliary data if ``include_aux`` is ``True``
        and auxiliary data is found in the ``Result``. If auxiliary data is in a dictionary the keys will be used
        as column names, otherwise they will be labeled ``aux_key_X`` for X auxiliary columns.

        Parameters
        ----------
        include_aux: bool = False
            Toggle to include auxiliary values in the dataframe. Requires auxiliary values in the ``Result``.

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

        if include_aux:
            if self.aux_values is not None:
                # Can use [0] for aux keys as the function is assumed producing the same structure of output each run
                if all(isinstance(auxs, dict) for auxs in self.aux_values):
                    expanded_data = [
                        data_row + list(auxs.values())
                        for data_row, auxs in zip(data, self.aux_values)
                    ]
                    aux_keys = list(self.aux_values[0].keys())
                else:
                    expanded_data = [
                        data_row + aux_row for data_row, aux_row in zip(data, self.aux_values)
                    ]
                    aux_keys = [f"aux_key_{val}" for val in range(len(self.aux_values[0]))]

                columns = columns + aux_keys
                data = expanded_data

            else:
                raise ValueError(
                    "``include_aux`` is True but no ``aux_values`` were found in the ``Results``."
                )

        df = pandas.DataFrame(data=data, columns=columns)

        attrs = dict(
            task_names=self.task_names,
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
            task_names=attrs.get("task_names"),
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

        task_names = combine_tuples(self.task_names, other.task_names)
        coords = combine_tuples(self.coords, other.coords)
        values = combine_tuples(self.values, other.values)

        return Result(
            dims=self.dims,
            coords=coords,
            values=values,
            output_names=self.output_names,
            fn_source=self.fn_source,
            task_names=task_names,
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

        # ParticleSwarm optimizer doesn't work with updated_copy
        # Creating new result with updated values and coords instead
        if self.optimizer is not None:
            new_result = Result(
                dims=self.dims,
                values=new_values,
                coords=new_coords,
                output_names=self.output_names,
                fn_source=self.fn_source,
                task_names=self.task_names,
                aux_values=self.aux_values,
                optimizer=self.optimizer,
            )
            return new_result

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

        # ParticleSwarm optimizer doesn't work with updated_copy
        # Creating new result with updated values and coords instead
        if self.optimizer is not None:
            new_result = Result(
                dims=self.dims,
                values=new_values,
                coords=new_coords,
                output_names=self.output_names,
                fn_source=self.fn_source,
                task_names=self.task_names,
                aux_values=self.aux_values,
                optimizer=self.optimizer,
            )
            return new_result

        return self.updated_copy(values=new_values, coords=new_coords)

    def __len__(self):
        """Implement len function to return the number of items in the result."""

        return len(self.coords)

    def __getitem__(self, data_index):
        """Implement the accessor function to index into the coordinates and values of the result."""

        features = self.coords[data_index]
        labels = self.values[data_index]

        return np.array(features), np.array(labels)
