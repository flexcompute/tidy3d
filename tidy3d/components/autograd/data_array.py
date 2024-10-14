import autograd.numpy as anp
import xarray as xr
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.tracer import isbox


class DataArrayAutogradMixin:
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
        args = (arg.data if isinstance(arg, xr.DataArray) else arg for arg in args)
        kwargs = {
            k: (arg.data if isinstance(arg, xr.DataArray) else arg) for k, arg in kwargs.items()
        }
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
