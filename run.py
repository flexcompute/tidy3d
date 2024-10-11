#!/usr/bin/env -S poetry run python
# ruff: noqa: F401

from time import perf_counter

import autograd
import autograd.numpy as anp
import numpy as np
import xarray as xr
from autograd import value_and_grad
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.test_util import check_grads
from autograd.tracer import getval, isbox


class DataArray(xr.DataArray):
    __slots__ = ()

    def __new__(cls, data, *args, **kwargs):
        if isbox(data):
            cls.__array_ufunc__ = cls._array_ufunc
            cls.__array_function__ = cls._array_function
        return super().__new__(cls)

    def __init__(self, data, *args, **kwargs):
        if isbox(data):
            ArrayBox.__array_namespace__ = lambda self, *, api_version=None: anp
            data = ArrayBox(data._value, data._trace, data._node)
        super().__init__(data, *args, **kwargs)

    @staticmethod
    def _extract_data(args, kwargs):
        args = (arg.data if isinstance(arg, DataArray) else arg for arg in args)
        kwargs = {k: (arg.data if isinstance(arg, DataArray) else arg) for k, arg in kwargs.items()}
        return args, kwargs

    @staticmethod
    def _get_anp_function(name):
        modules = (anp.linalg, anp.fft, anp)
        for module in modules:
            func = getattr(module, name, None)
            if func is not None:
                return func
        raise NotImplementedError(f"The function '{name}' is not implemented in autograd.numpy")

    def _array_ufunc(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplementedError(f"ufunc method {method} is not implemented")
        args, kwargs = self._extract_data(inputs, kwargs)
        f = self._get_anp_function(ufunc.__name__)
        return f(*args, **kwargs)

    def _array_function(self, func, types, args, kwargs):
        args, kwargs = self._extract_data(args, kwargs)
        f = self._get_anp_function(func.__name__)
        return f(*args, **kwargs)


def f_data(x):
    da = DataArray(x, dims=range(x.ndim))
    return anp.mean(anp.real(da))
    # return anp.mean(anp.multiply(da, da))


def main():
    rng = np.random.default_rng(1232523)
    x = rng.uniform(1, 10, (100, 100))
    # x = np.arange(5).astype(float)

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
