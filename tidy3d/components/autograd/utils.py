# utilities for working with autograd

import typing

import autograd.numpy as anp
import numpy as np
import xarray as xr
from autograd.tracer import getval, isbox

from .constants import AUTOGRAD_KEY


def _is_traced_objary(x: typing.Any) -> bool:
    return isinstance(x, np.ndarray) and x.size > 0 and isbox(x.flat[0])


def get_data(x: typing.Any) -> typing.Any:
    if isinstance(x, xr.DataArray) or isinstance(x, xr.Variable):
        return x.data
    return x


def is_traced(x: typing.Any) -> bool:
    """Check whether x is traced (contains an autograd Box)"""
    data = get_data(x)
    return isbox(data) or _is_traced_objary(data)


def get_static(x: typing.Any) -> typing.Any:
    """Get the 'static' (untraced) version of some value."""
    data = getval(get_data(x))

    if isinstance(x, xr.DataArray) or isinstance(x, xr.Variable):
        return x.copy(deep=False, data=data)

    if _is_traced_objary(data) or (
        isinstance(data, np.ndarray) and data.dtype == np.dtype("object")
    ):
        return np.array(data.tolist())

    return data


def get_box(x: typing.Any) -> typing.Any:
    """Get the 'boxed' (traced) version of some value."""
    if not is_traced(x):
        return None

    data = get_data(x)

    if not _is_traced_objary(data):
        return data

    if hasattr(x, "attrs") and isinstance(x.attrs, dict) and AUTOGRAD_KEY in x.attrs:
        return x.attrs[AUTOGRAD_KEY]

    return anp.array(data.tolist())


def split_list(x: list[typing.Any], index: int) -> (list[typing.Any], list[typing.Any]):
    """Split a list at a given index."""
    x = list(x)
    return x[:index], x[index:]


__all__ = [
    "get_static",
    "get_data",
    "get_box",
    "is_traced",
    "split_list",
]
