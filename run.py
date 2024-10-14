#!/usr/bin/env -S poetry run python
# ruff: noqa: F401

import warnings
from time import perf_counter

import autograd
import autograd.numpy as anp
import numpy as np
import xarray as xr
from autograd import value_and_grad
from autograd.extend import VSpace
from autograd.numpy.numpy_vspaces import ArrayVSpace, ComplexArrayVSpace
from autograd.test_util import check_grads
from autograd.tracer import getval, isbox

from tidy3d.components.autograd.boxes import TidyArrayBox


class DataArray(xr.DataArray):
    __slots__ = ()

    def __new__(cls, data, *args, **kwargs):
        if isbox(data):

            def values(self):
                warnings.warn(
                    "'DataArray.values' is deprecated, please use 'DataArray.data' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return self.data

            cls.values = property(values)
        return super().__new__(cls)

    def __init__(self, data, *args, **kwargs):
        if isbox(data):
            data = TidyArrayBox(data._value, data._trace, data._node)
        super().__init__(data, *args, **kwargs)

    @property
    def abs(self):
        return abs(self)


def f_data(x):
    coords = {chr(65 + i): range(x.shape[i]) for i in range(x.ndim)}
    da = DataArray(x, coords=coords)
    da = da.sel(A=[1, 5], method="nearest")
    da = da.isel(B=[1, 3, 6, 10])
    return anp.linalg.norm(da.data)


def main():
    rng = np.random.default_rng(1232523)
    x = rng.uniform(-10, 10, (100, 100))

    t0 = perf_counter()
    for _ in range(10):
        f_data(x)
    print(f"fwd:  {perf_counter() - t0:.3e}s")

    t0 = perf_counter()
    for _ in range(10):
        autograd.grad(f_data)(x)
    print(f"bwd:  {perf_counter() - t0:.3e}s")

    check_grads(f_data, modes=["fwd", "rev"], order=2)(x)


if __name__ == "__main__":
    main()
