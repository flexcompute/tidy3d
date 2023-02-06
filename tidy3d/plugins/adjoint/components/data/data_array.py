"""Defines jax-compatible DataArrays."""
from __future__ import annotations

from typing import Tuple, Any, Dict, List

import pydantic as pd
import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from .....components.base import Tidy3dBaseModel, cached_property
from .....log import DataError, Tidy3dKeyError
from ...log import AdjointError


@register_pytree_node_class
class JaxDataArray(Tidy3dBaseModel):
    """A :class:`.DataArray`-like class that only wraps xarray for jax compability."""

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

    @pd.validator("coords", always=True)
    def _convert_coords_to_list(cls, val):
        """Convert supplied coordinates to Dict[str, list]."""
        return {coord_name: list(coord_list) for coord_name, coord_list in val.items()}

    # removed because it was slowing things down.
    # @pd.validator("coords", always=True)
    # def _coords_match_values(cls, val, values):
    #     """Make sure the coordinate dimensions and shapes match the values data."""

    #     values = values.get("values")

    #     # if values did not pass validation, just skip this validator
    #     if values is None:
    #         return None

    #     # compute the shape, otherwise exit.
    #     try:
    #         shape = jnp.array(values).shape
    #     except TypeError:
    #         return val

    #     if len(shape) != len(val):
    #         raise AdjointError(f"'values' has '{len(shape)}' dims, but given '{len(val)}'.")

    #     # make sure each coordinate list has same length as values along that axis
    #     for len_dim, (coord_name, coord_list) in zip(shape, val.items()):
    #         if len_dim != len(coord_list):
    #             raise AdjointError(
    #                 f"coordinate '{coord_name}' has '{len(coord_list)}' elements, "
    #                 f"expected '{len_dim}' to match number of 'values' along this dimension."
    #             )

    #     return val

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
        return self.as_ndarray.shape

    @cached_property
    def as_list(self) -> list:
        """``self.values`` as a numpy array converted to a list."""
        return self.as_ndarray.tolist()

    @cached_property
    def real(self) -> np.ndarray:
        """Real part of self."""
        new_values = np.real(self.as_ndarray)
        return self.copy(update=dict(values=new_values))

    @cached_property
    def imag(self) -> np.ndarray:
        """Imaginary part of self."""
        new_values = np.imag(self.as_ndarray)
        return self.copy(update=dict(values=new_values))

    def get_coord_list(self, coord_name: str) -> list:
        """Get a coordinate list by name."""

        if coord_name not in self.coords:
            raise Tidy3dKeyError(f"Could not select '{coord_name}', not found in coords dict.")
        return self.coords.get(coord_name)

    def isel_single(self, coord_name: str, coord_index: int) -> JaxDataArray:
        """Select a value cooresponding to a single coordinate from the :class:`.JaxDataArray`."""

        # select out the proper values and coordinates
        coord_axis = list(self.coords.keys()).index(coord_name)
        values = self.as_jnp_array
        new_values = jnp.take(values, indices=coord_index, axis=coord_axis)
        new_coords = self.coords.copy()
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
            coord_list = self_sel.get_coord_list(coord_name)
            if coord_index < 0 or coord_index >= len(coord_list):
                raise DataError(
                    f"'isel' kwarg '{coord_name}={coord_index}' is out of range "
                    f"for the coordinate '{coord_name}' with {len(coord_list)} values."
                )
            self_sel = self_sel.isel_single(coord_name=coord_name, coord_index=coord_index)

        return self_sel

    def sel(self, **sel_kwargs) -> JaxDataArray:
        """Select a value from the :class:`.JaxDataArray` by indexing into coordinate values."""
        isel_kwargs = {}
        for coord_name, sel_kwarg in sel_kwargs.items():
            coord_list = self.get_coord_list(coord_name)
            if sel_kwarg not in coord_list:
                raise DataError(f"Could not select '{coord_name}={sel_kwarg}', value not found.")
            coord_index = coord_list.index(sel_kwarg)
            isel_kwargs[coord_name] = coord_index
        return self.isel(**isel_kwargs)

    def interp(self, **interp_kwargs):
        """Interpolate into the :class:`.JaxDataArray`. Not yet supported."""

        raise NotImplementedError("Interpolation is not currently supported in the 'output_data'.")

    @cached_property
    def nonzero_val_coords(self) -> Tuple[List[complex], Dict[str, Any]]:
        """The value and coordinate associated with the only non-zero element of ``self.values``."""

        values = np.nan_to_num(self.as_ndarray)
        nonzero_inds = np.nonzero(values)
        nonzero_values = values[nonzero_inds].tolist()

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
