"""Autograd-compatible selection, interpolation, and update methods bound onto ``DataArray``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import autograd.numpy as anp
import numpy as np
import xarray as xr
from xarray.core import missing
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import _outer_to_numpy_indexer
from xarray.core.utils import OrderedSet, either_dict_or_kwargs
from xarray.core.variable import as_variable

from tidy3d.compat import alignment
from tidy3d.components.autograd import interpn
from tidy3d.exceptions import format_chained_exception_message

if TYPE_CHECKING:
    from collections.abc import Mapping

    from xarray.core.types import InterpOptions

    from tidy3d.components.autograd import InterpolationType

    from .base import DataArray

# Stands for the class a method is bound to, as 'Self' did when these were methods of 'DataArray'.
_DataArrayT = TypeVar("_DataArrayT", bound="DataArray")


def _ag_multiply_at(
    self: _DataArrayT, value: complex, coord_name: str, indices: list[int]
) -> _DataArrayT:
    """Autograd multiply_at override when tracing."""
    key = {coord_name: indices}
    _, index_tuple, _ = self.variable._broadcast_indexes(key)
    idx = _outer_to_numpy_indexer(index_tuple, self.data.shape)
    mask = np.zeros(self.data.shape, dtype="?")
    mask[idx] = True
    return self.copy(deep=False, data=anp.where(mask, self.data * value, self.data))


def _ag_interp(
    self: _DataArrayT,
    coords: Mapping[Any, Any] | None = None,
    method: InterpOptions = "linear",
    assume_sorted: bool = False,
    kwargs: Mapping[str, Any] | None = None,
    **coords_kwargs: Any,
) -> _DataArrayT:
    """Autograd interp override when tracing over self.data.

    This implementation closely follows the interp implementation of xarray
    to match its behavior as closely as possible while supporting autograd.

    See:
    - https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html
    - https://docs.xarray.dev/en/latest/generated/xarray.Dataset.interp.html
    """
    if kwargs is None:
        kwargs = {}

    ds = self._to_temp_dataset()

    coords = either_dict_or_kwargs(coords, coords_kwargs, "interp")
    indexers = dict(ds._validate_interp_indexers(coords))

    if coords:
        # Find shared dimensions between the dataset and the indexers
        sdims = (
            set(ds.dims)
            .intersection(*[set(nx.dims) for nx in indexers.values()])
            .difference(coords.keys())
        )
        indexers.update({d: ds.variables[d] for d in sdims})

    obj = ds if assume_sorted else ds.sortby(list(coords))

    # workaround to get a variable for a dimension without a coordinate
    validated_indexers = {
        k: (obj._variables.get(k, as_variable((k, range(obj.sizes[k])))), v)
        for k, v in indexers.items()
    }

    for k, v in validated_indexers.items():
        obj, newidx = missing._localize(obj, {k: v})
        validated_indexers[k] = newidx[k]

    variables = {}
    reindex = False
    for name, var in obj._variables.items():
        if name in indexers:
            continue
        dtype_kind = var.dtype.kind
        if dtype_kind in "uifc":
            # Interpolation for numeric types
            var_indexers = {k: v for k, v in validated_indexers.items() if k in var.dims}
            variables[name] = self._ag_interp_func(var, var_indexers, method, **kwargs)
        elif dtype_kind in "ObU" and (validated_indexers.keys() & var.dims):
            # Stepwise interpolation for non-numeric types
            reindex = True
        elif all(d not in indexers for d in var.dims):
            # Keep variables not dependent on interpolated coords
            variables[name] = var

    if reindex:
        # Reindex for non-numeric types
        reindex_indexers = {k: v for k, (_, v) in validated_indexers.items() if v.dims == (k,)}
        reindexed = alignment.reindex(
            obj,
            indexers=reindex_indexers,
            method="nearest",
            exclude_vars=variables.keys(),
        )
        indexes = dict(reindexed._indexes)
        variables.update(reindexed.variables)
    else:
        # Get the indexes that are not being interpolated along
        indexes = {k: v for k, v in obj._indexes.items() if k not in indexers}

    # Get the coords that also exist in the variables
    coord_names = obj._coord_names & variables.keys()
    selected = ds._replace_with_new_dims(variables.copy(), coord_names, indexes=indexes)

    # Attach indexer as coordinate
    for k, v in indexers.items():
        if v.dims == (k,):
            index = PandasIndex(v, k, coord_dtype=v.dtype)
            index_vars = index.create_variables({k: v})
            indexes[k] = index
            variables.update(index_vars)
        else:
            variables[k] = v

    # Extract coordinates from indexers
    coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(coords)
    variables.update(coord_vars)
    indexes.update(new_indexes)

    coord_names = obj._coord_names & variables.keys() | coord_vars.keys()
    ds = ds._replace_with_new_dims(variables, coord_names, indexes=indexes)
    return self._from_temp_dataset(ds)


def _ag_interp_func(
    var: xr.Variable,
    indexes_coords: dict[str, tuple[xr.Variable, xr.Variable]],
    method: InterpolationType,
    **kwargs: Any,
) -> xr.Variable:
    """
    Interpolate the variable `var` along the coordinates specified in `indexes_coords` using the given `method`.

    The implementation follows xarray's interp implementation in xarray.core.missing,
    but replaces some of the pre-processing as well as the actual interpolation
    function with an autograd-compatible approach.


    Parameters
    ----------
    var : xr.Variable
        The variable to be interpolated.
    indexes_coords : dict
        A dictionary mapping dimension names to coordinate values for interpolation.
    method : Literal["nearest", "linear"]
        The interpolation method to use.
    **kwargs : dict
        Additional keyword arguments to pass to the interpolation function.

    Returns
    -------
    xr.Variable
        The interpolated variable.
    """
    if not indexes_coords:
        return var.copy()
    result = var
    for indep_indexes_coords in missing.decompose_interp(indexes_coords):
        var = result

        # target dimensions
        dims = list(indep_indexes_coords)
        x, new_x = zip(*[indep_indexes_coords[d] for d in dims])
        destination = missing.broadcast_variables(*new_x)

        broadcast_dims = [d for d in var.dims if d not in dims]
        original_dims = broadcast_dims + dims
        new_dims = broadcast_dims + list(destination[0].dims)

        x, new_x = missing._floatize_x(x, new_x)

        permutation = [var.dims.index(dim) for dim in original_dims]
        combined_permutation = permutation[-len(x) :] + permutation[: -len(x)]
        data = anp.transpose(var.data, combined_permutation)
        xi = anp.stack([anp.ravel(new_xi.data) for new_xi in new_x], axis=-1)

        result = interpn([xn.data for xn in x], data, xi, method=method, **kwargs)

        result = anp.moveaxis(result, 0, -1)
        result = anp.reshape(result, result.shape[:-1] + new_x[0].shape)

        result = xr.Variable(new_dims, result, attrs=var.attrs, fastpath=True)

        out_dims: OrderedSet = OrderedSet()
        for d in var.dims:
            if d in dims:
                out_dims.update(indep_indexes_coords[d][1].dims)
            else:
                out_dims.add(d)
        if len(out_dims) > 1:
            result = result.transpose(*out_dims)
    return result


def _with_updated_data(self: _DataArrayT, data: np.ndarray, coords: dict[str, Any]) -> DataArray:
    """Make copy of ``DataArray`` with ``data`` at specified ``coords``, autograd compatible

    Constraints / Edge cases:
        - `coords` must map to a specific value eg {x: '1'}, does not broadcast to arrays
        - `data` will be reshaped to try to match `self.shape` except where `coords` present
    """

    # make mask
    mask = xr.zeros_like(self, dtype=bool)
    mask.loc[coords] = True

    # reshape `data` to line up with `self.dims`, with shape of 1 along the selected axis
    old_data = self.data
    new_shape = list(old_data.shape)
    for i, dim in enumerate(self.dims):
        if dim in coords:
            new_shape[i] = 1
    try:
        new_data = data.reshape(new_shape)
    except ValueError as e:
        raise ValueError(
            format_chained_exception_message(
                "Couldn't reshape the supplied 'data' to update 'DataArray'. The provided "
                f"data was of shape {data.shape} and tried to reshape to {new_shape}. If "
                "you encounter this error please raise an issue on the tidy3d github "
                "repository with this context.",
                e,
            )
        ) from e

    # broadcast data to repeat data along the selected dimensions to match mask
    new_data = new_data + np.zeros_like(old_data)

    new_data = np.where(mask, new_data, old_data)

    return self.copy(deep=True, data=new_data)
