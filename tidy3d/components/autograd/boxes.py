import autograd.numpy as anp
from autograd.numpy.numpy_boxes import ArrayBox

TidyArrayBox = ArrayBox  # alias for clarity, technically not necessary


def __array_function__(self, func, types, args, kwargs):
    f = getattr(anp, func.__name__, None)
    if f is None:
        raise NotImplementedError(
            f"The function '{func.__name__}' is not implemented in autograd.numpy"
        )
    if f.__name__ == "nanmean":
        f = anp.mean  # somehow xarray dispatches mean to nanmean
        kwargs.pop("dtype")  # autograd mean vjp doesn't support dtype
    return f(*args, **kwargs)


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if method != "__call__":
        raise NotImplementedError(f"ufunc method {method} is not implemented")
    f = getattr(anp, ufunc.__name__, None)
    if f is None:
        raise NotImplementedError(
            f"The ufunc '{ufunc.__name__}' is not implemented in autograd.numpy"
        )
    return f(*inputs, **kwargs)


def item(self):
    if self.size != 1:
        raise ValueError("Can only convert an array of size 1 to a scalar")
    return anp.ravel(self)[0]


TidyArrayBox.__array_namespace__ = lambda self, *, api_version=None: anp
TidyArrayBox.__array_ufunc__ = __array_ufunc__
TidyArrayBox.__array_function__ = __array_function__
TidyArrayBox.__repr__ = lambda self: str(self)
TidyArrayBox.real = property(anp.real)
TidyArrayBox.imag = property(anp.imag)
TidyArrayBox.conj = anp.conj
TidyArrayBox.item = item
