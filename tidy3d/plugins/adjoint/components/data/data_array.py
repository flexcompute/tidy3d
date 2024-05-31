"""Defines jax-compatible DataArrays."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pydantic.v1 as pd
import xarray as xr
from jax.tree_util import register_pytree_node_class

from .....components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from .....exceptions import AdjointError, DataError, Tidy3dKeyError

# condition setting when to set value in DataArray to zero:
# if abs(val) <= VALUE_FILTER_THRESHOLD * max(abs(val))
VALUE_FILTER_THRESHOLD = 1e-6

# JaxDataArrays are written to json as JAX_DATA_ARRAY_TAG
JAX_DATA_ARRAY_TAG = "<<JaxDataArray>>"


@register_pytree_node_class
class JaxDataArray(Tidy3dBaseModel):
    """A :class:`.DataArray`-like class that only wraps xarray for jax compatibility."""

    values: Any = pd.Field(
        ...,
        title="Values",
        description="Nested list containing the raw values, which can be tracked by jax.",
        jax_field=True,
    )

    coords: Dict[str, list] = pd.Field(
        ...,
        title="Coords",
        description="Dictionary storing the coordinates, namely ``(direction, f, mode_index)``.",
    )

    def to_tidy3d(self: JaxDataArray) -> xr.DataArray:
        """Convert :class:`.JaxDataArray` instance to ``xr.DataArray`` instance."""
        coords = {k: np.array(v).tolist() for k, v in self.coords.items()}
        return xr.DataArray(np.array(self.values), coords=coords, dims=self.coords.keys())

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: xr.DataArray) -> JaxDataArray:
        """Convert ``xr.DataArray`` instance to :class:`.JaxDataArray`."""
        coords = {k: np.array(v).tolist() for k, v in tidy3d_obj.coords.items()}
        return cls(values=tidy3d_obj.data, coords=coords)

    @pd.validator("values", always=True)
    def _convert_values_to_np(cls, val):
        """Convert supplied values to numpy if they are list (from file)."""
        if isinstance(val, list):
            return np.array(val)
        return val

    @pd.validator("coords", always=True)
    @skip_if_fields_missing(["values"])
    def _coords_match_values(cls, val, values):
        """Make sure the coordinate dimensions and shapes match the values data."""

        _values = values.get("values")

        # get the shape, handling both regular and jax objects
        try:
            values_shape = np.array(_values).shape
        except TypeError:
            values_shape = jnp.array(_values).shape

        for (key, coord_val), size_dim in zip(val.items(), values_shape):
            if len(coord_val) != size_dim:
                raise ValueError(
                    f"JaxDataArray coord {key} has {len(coord_val)} elements, "
                    "which doesn't match the values array "
                    f"with size {size_dim} along that dimension."
                )

        return val

    @pd.validator("coords", always=True)
    def _convert_coords_to_list(cls, val):
        """Convert supplied coordinates to Dict[str, list]."""
        return {coord_name: list(coord_list) for coord_name, coord_list in val.items()}

    def __eq__(self, other) -> bool:
        """Check if two ``JaxDataArray`` instances are equal."""
        return jnp.array_equal(self.values, other.values)

    def to_hdf5(self, fname: str, group_path: str) -> None:
        """Save an xr.DataArray to the hdf5 file with a given path to the group."""
        sub_group = fname.create_group(group_path)
        sub_group["values"] = self.values
        dims = []
        for key, val in self.coords.items():
            # sub_group[key] = val
            dims.append(key)
            val = np.array(val)
            if val.dtype == "<U1":
                sub_group[key] = val.tolist()
            else:
                sub_group[key] = val
        sub_group["dims"] = dims

    @classmethod
    def from_hdf5(cls, fname: str, group_path: str) -> JaxDataArray:
        """Load an DataArray from an hdf5 file with a given path to the group."""
        with h5py.File(fname, "r") as f:
            sub_group = f[group_path]
            values = np.array(sub_group["values"])
            dims = sub_group["dims"]
            coords = {dim: np.array(sub_group[dim]) for dim in dims}
            for key, val in coords.items():
                val = np.array(val)
                if val.dtype == "O":
                    coords[key] = [byte_string.decode() for byte_string in val.tolist()]

            coords = {
                key: val.tolist() if isinstance(val, np.ndarray) else val
                for key, val in coords.items()
            }
            return cls(values=values, coords=coords)

    @cached_property
    def as_ndarray(self) -> np.ndarray:
        """``self.values`` as a numpy array."""
        if not isinstance(self.values, np.ndarray):
            return np.array(self.values)
        return self.values

    @cached_property
    def as_jnp_array(self) -> jnp.ndarray:
        """``self.values`` as a jax array."""
        if not isinstance(self.values, jnp.ndarray):
            return jnp.array(self.values)
        return self.values

    @cached_property
    def shape(self) -> tuple:
        """Shape of self.values."""
        return self.as_jnp_array.shape

    @cached_property
    def as_list(self) -> list:
        """``self.values`` as a numpy array converted to a list."""
        return self.as_ndarray.tolist()

    @cached_property
    def real(self) -> np.ndarray:
        """Real part of self."""
        new_values = jnp.real(self.as_jnp_array)
        return self.copy(update=dict(values=new_values))

    @cached_property
    def imag(self) -> np.ndarray:
        """Imaginary part of self."""
        new_values = jnp.imag(self.as_jnp_array)
        return self.copy(update=dict(values=new_values))

    def conj(self) -> JaxDataArray:
        """Complex conjugate of self."""
        new_values = jnp.conj(self.as_jnp_array)
        return self.copy(update=dict(values=new_values))

    def __abs__(self) -> JaxDataArray:
        """Absolute value of self's values."""
        new_values = jnp.abs(self.as_jnp_array)
        return self.updated_copy(values=new_values)

    def __pow__(self, power: int) -> JaxDataArray:
        """Values raised to a power."""
        new_values = self.as_jnp_array**power
        return self.updated_copy(values=new_values)

    def __add__(self, other: JaxDataArray) -> JaxDataArray:
        """Sum self with something else."""
        if isinstance(other, JaxDataArray):
            new_values = self.as_jnp_array + other.as_jnp_array
        else:
            new_values = self.as_jnp_array + other
        return self.updated_copy(values=new_values)

    def __neg__(self) -> JaxDataArray:
        """Negative of self."""
        new_values = -self.as_jnp_array
        return self.updated_copy(values=new_values)

    def __sub__(self, other) -> JaxDataArray:
        """Subtraction"""
        return self + (-other)

    def __radd__(self, other) -> JaxDataArray:
        """Sum self with something else."""
        return self + other

    def __mul__(self, other: JaxDataArray) -> JaxDataArray:
        """Multiply self with something else."""
        if isinstance(other, JaxDataArray):
            new_values = self.as_jnp_array * other.as_jnp_array
        elif isinstance(other, xr.DataArray):
            # handle case where other is missing dims present in self
            new_shape = list(self.shape)
            for dim_index, dim in enumerate(self.coords.keys()):
                if dim not in other.dims:
                    other = other.expand_dims(dim=dim)
                    new_shape[dim_index] = 1

            other_values = other.values.reshape(new_shape)
            new_values = self.as_jnp_array * other_values
        else:
            new_values = self.as_jnp_array * other

        return self.updated_copy(values=new_values)

    def __rmul__(self, other) -> JaxDataArray:
        """Multiply self with something else."""
        return self * other

    def sum(self, dim: str = None):
        """Sum (optionally along a single or multiple dimensions)."""

        if dim is None:
            return jnp.sum(self.as_jnp_array)

        # dim is supplied
        if isinstance(dim, str):
            axis = list(self.coords.keys()).index(dim)
            new_values = jnp.sum(self.values, axis=axis)
            new_coords = self.coords.copy()
            new_coords.pop(dim)
            return self.updated_copy(values=new_values, coords=new_coords)

        # dim is iterative, recursively call sum with single dim
        ret = self.copy()
        for dim_i in dim:
            ret = ret.sum(dim=dim_i)
        return ret

    def squeeze(self, dim: str = None, drop: bool = True) -> JaxDataArray:
        """Remove any non-zero dims."""

        if dim is None:
            new_values = jnp.squeeze(self.as_jnp_array)

            new_coords = {}
            for (key, val), dim_size in zip(self.coords.items(), self.values.shape):
                if dim_size > 1:
                    new_coords.update({key: val})

        else:
            axis = list(self.coords.keys()).index(dim)
            new_values = jnp.array(jnp.squeeze(self.as_jnp_array, axis=axis))
            new_coords = self.coords.copy()
            new_coords.pop(dim)
        return self.updated_copy(values=new_values, coords=new_coords)

    def get_coord_list(self, coord_name: str) -> list:
        """Get a coordinate list by name."""

        if coord_name not in self.coords:
            raise Tidy3dKeyError(f"Could not select '{coord_name}', not found in coords dict.")
        return self.coords.get(coord_name)

    def isel_single(self, coord_name: str, coord_index: int) -> JaxDataArray:
        """Select a value corresponding to a single coordinate from the :class:`.JaxDataArray`."""

        # select out the proper values and coordinates
        coord_axis = list(self.coords.keys()).index(coord_name)
        values = self.as_jnp_array
        new_values = jnp.take(values, indices=coord_index, axis=coord_axis)
        new_coords = self.coords.copy()

        # if the coord index has more than one item, keep that coordinate
        coord_index = np.array(coord_index)
        if len(coord_index.shape) >= 1:
            coord_indices = coord_index.tolist()
            new_coord_vals = [self.coords[coord_name][coord_index] for coord_index in coord_indices]
            new_coords[coord_name] = new_coord_vals
        else:
            new_coords.pop(coord_name)

        # return just the values if no coordinate remain
        if not new_coords:
            if new_values.shape:
                raise AdjointError(
                    "All coordinates selected out, but raw data values are still multi-dimensional."
                    " If you encountered this error, please raise an issue on the Tidy3D "
                    "front end github repository so we can look into the source of the bug."
                )

            return new_values

        # otherwise, return another JaxDataArray with the values and coords selected out
        return self.copy(update=dict(values=new_values, coords=new_coords))

    def isel(self, **isel_kwargs) -> JaxDataArray:
        """Select a value from the :class:`.JaxDataArray` by indexing into coordinates by index."""

        self_sel = self.copy()
        for coord_name, coord_index in isel_kwargs.items():
            coord_index = np.array(coord_index)
            coord_list = self_sel.get_coord_list(coord_name)
            if np.any(coord_index < 0) or np.any(coord_index >= len(coord_list)):
                raise DataError(
                    f"'isel' kwarg '{coord_name}={coord_index}' is out of range "
                    f"for the coordinate '{coord_name}' with {len(coord_list)} values."
                )
            self_sel = self_sel.isel_single(coord_name=coord_name, coord_index=coord_index)

        return self_sel

    def sel(
        self, indexers: dict = None, method: Literal[None, "nearest"] = None, **sel_kwargs
    ) -> JaxDataArray:
        """Select a value from the :class:`.JaxDataArray` by indexing into coordinates by value.

        Parameters
        ----------
        sel_kwargs : dict
            Keyword arguments with names matching the coordinates of :class:`.JaxDataArray` and values
            given by scalars or lists, e.g. `da.sel(x=0.1, y=[0.2, 0.3])`.
        method : Literal[None, "nearest"] = None
            Method to use for matching coordinate values:

            - None (default): only exact matches
            - nearest: use nearest valid index value

        Returns
        -------
        JaxDataArray
            JaxDataArray with extracted values.
        """
        if method not in [None, "nearest"]:
            raise NotImplementedError(f"Unkown selection method: {method}.")

        isel_kwargs = {}
        for coord_name, vals in sel_kwargs.items():
            coord_list = self.get_coord_list(coord_name)

            try:  # handle non-numeric types (e.g. str)
                coord_list = jnp.asarray(coord_list)
            except TypeError:
                isel_kwargs[coord_name] = self._indices_literal(coord_list, vals)
                continue

            vals_ary = jnp.atleast_1d(vals)
            dist = jnp.abs(coord_list[None] - vals_ary[:, None])

            if method is None:
                indices = jnp.where(jnp.isclose(dist, 0))[1]
            elif method == "nearest":
                indices = jnp.argmin(dist, axis=1)

            if indices.size == 0:
                raise DataError(
                    f"Could not select '{coord_name}={vals_ary}', some values were not found."
                )

            if np.isscalar(vals):
                indices = jnp.squeeze(indices)

            isel_kwargs[coord_name] = indices

        return self.isel(**isel_kwargs)

    def _indices_literal(self, coord_list: list, values: Union[Any, Sequence[Any]]) -> np.ndarray:
        """Find indices of non-numeric `values` in `coord_list`.

        Parameters
        ----------
        coord_list : list
            List of all entries for a specific coordinate.
        values : Union[Any, Sequence[Any]]
            Single value or values of which to find the index of.

        Returns
        -------
        numpy.ndarray
            Indices of `values` in `coord_list`.
        """
        indices = []
        for v in np.atleast_1d(values):
            if v not in coord_list:
                raise DataError(f"Could not select '{v}' from '{coord_list}', value not found.")
            indices.append(coord_list.index(v))

        if np.isscalar(values):
            indices = np.squeeze(indices)

        return indices

    def assign_coords(self, coords: dict = None, **coords_kwargs) -> JaxDataArray:
        """Assign new coordinates to this object."""

        update_kwargs = self.coords.copy()

        for key, val in coords_kwargs.items():
            update_kwargs[key] = val

        if coords:
            for key, val in coords.items():
                update_kwargs[key] = val

        update_kwargs = {key: np.array(value).tolist() for key, value in update_kwargs.items()}
        return self.updated_copy(coords=update_kwargs)

    def multiply_at(self, value: complex, coord_name: str, indices: List[int]) -> JaxDataArray:
        """Multiply self by value at indices into ."""
        axis = list(self.coords.keys()).index(coord_name)
        scalar_data_arr = self.as_jnp_array
        scalar_data_arr = jnp.moveaxis(scalar_data_arr, axis, 0)
        scalar_data_arr = scalar_data_arr.at[indices].multiply(value)
        scalar_data_arr = jnp.moveaxis(scalar_data_arr, 0, axis)
        return self.updated_copy(values=scalar_data_arr)

    def interp_single(self, key: str, val: float) -> JaxDataArray:
        """Interpolate into a single dimension of self.

        Note: this interpolation works by finding the index of the value into the coords list.
        Instead of an integer value, we use interpolation to get a floating point index.
        The floor() of this value is the 'minus' index and the ceil() gives the 'plus' index.
        We then apply coefficients linearly based on how close to `plus` or minus we are.
        This is a workaround to `jnp.interp` not allowing multi-dimensional interpolation.
        """

        val = jax.lax.stop_gradient(val)

        # get the coordinates associated with this key.
        if key not in self.coords:
            raise Tidy3dKeyError(f"Key '{key}' not found in JaxDataArray coords.")
        coords_1d = jnp.array(self.coords[key])
        axis = list(self.coords.keys()).index(key)

        # get floating point index of the value into these coordinates
        coord_indices = jnp.arange(len(coords_1d))
        index_interp = jnp.interp(x=val, xp=coords_1d, fp=coord_indices)

        # strip out the linear interpolation coefficients from the float index
        index_minus = np.array(index_interp).astype(int)
        index_plus = index_minus + 1
        coeff_plus = index_interp - index_minus

        # if any plus_index is out of range, set it in range (coeff will be 0 anyway)
        if index_plus.shape:
            index_plus[index_plus >= len(coord_indices)] = len(coord_indices) - 1
        else:
            if index_plus > len(coord_indices):
                index_plus = index_minus
                coeff_plus = 0.0

        coeff_minus = 1 - coeff_plus

        def get_values_at_index(key: str, index: int) -> jnp.array:
            """grab values array at index into coordinate key."""
            values_sel = self.isel(**{key: index})
            if isinstance(values_sel, JaxDataArray):
                return values_sel.values
            return values_sel

        # return weighted average of this object along these dimensions
        values_minus = get_values_at_index(key=key, index=index_minus)

        if np.any(coeff_plus > 0):
            values_plus = get_values_at_index(key=key, index=index_plus)

            if coeff_minus.shape:
                coeff_shape = np.ones(len(values_minus.shape), dtype=int)
                coeff_shape[axis] = len(coeff_minus)
                coeff_minus = coeff_minus.reshape(coeff_shape)
                coeff_plus = coeff_plus.reshape(coeff_shape)

            values_interp = coeff_minus * values_minus + coeff_plus * values_plus
        else:
            values_interp = values_minus

        # construct a new JaxDataArray to return
        coords_interp = self.coords.copy()
        if jnp.array(index_interp).size <= 1:
            coords_interp.pop(key)
        else:
            coords_interp[key] = np.atleast_1d(val).tolist()

        if coords_interp:
            return JaxDataArray(values=values_interp, coords=coords_interp)
        return values_interp

    def interp(self, kwargs=None, assume_sorted=None, **interp_kwargs) -> JaxDataArray:
        """Linearly interpolate into the :class:`.JaxDataArray` at values into coordinates."""

        # note: kwargs does nothing, only used for making this subclass compatible with super

        ret_value = self.copy()
        for key, val in interp_kwargs.items():
            ret_value = ret_value.interp_single(key=key, val=val)
        return ret_value

    @cached_property
    def nonzero_val_coords(self) -> Tuple[List[complex], Dict[str, Any]]:
        """The value and coordinate associated with the only non-zero element of ``self.values``."""

        values = np.nan_to_num(self.as_ndarray)

        # filter out values that are very small relative to maximum
        values_filtered = values.copy()
        max_value = np.max(np.abs(values_filtered))
        val_cutoff = VALUE_FILTER_THRESHOLD * max_value
        values_filtered[np.abs(values_filtered) <= val_cutoff] = 0.0

        nonzero_inds = np.nonzero(values_filtered)
        nonzero_values = values_filtered[nonzero_inds].tolist()

        nonzero_coords = {}
        for nz_inds, (coord_name, coord_list) in zip(nonzero_inds, self.coords.items()):
            coord_array = np.array(coord_list)
            nonzero_coords[coord_name] = coord_array[nz_inds].tolist()

        return nonzero_values, nonzero_coords

    def tree_flatten(self) -> Tuple[list, dict]:
        """Jax works on the values, stash the coords for reconstruction."""

        return self.values, self.coords

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxDataArray:
        """How to unflatten the values and coords."""

        return cls(values=children, coords=aux_data)
