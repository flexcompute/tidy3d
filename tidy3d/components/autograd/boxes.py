# Adds some functionality to the autograd arraybox
# NOTE: this is not a subclass of ArrayBox since that would break autograd's internal checks

import importlib
from typing import Any, Callable, Dict, List, Tuple

import autograd.numpy as anp
from autograd.extend import defjvp
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.numpy.numpy_wrapper import _astype

TidyArrayBox = ArrayBox  # NOT a subclass

_autograd_module_cache = {}  # cache for imported autograd modules

defjvp(
    _astype,
    lambda g, ans, A, dtype, order="K", casting="unsafe", subok=True, copy=True: _astype(g, dtype),
)

anp.astype = _astype


@classmethod
def from_arraybox(cls, box: ArrayBox) -> TidyArrayBox:
    """Construct a TidyArrayBox from an ArrayBox."""
    return cls(box._value, box._trace, box._node)


def __array_function__(
    self: Any,
    func: Callable,
    types: List[Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """
    Handle the dispatch of NumPy functions to autograd's numpy implementation.

    Parameters
    ----------
    self : Any
        The instance of the class.
    func : Callable
        The NumPy function being called.
    types : List[Any]
        The types of the arguments that implement __array_function__.
    args : Tuple[Any, ...]
        The positional arguments to the function.
    kwargs : Dict[str, Any]
        The keyword arguments to the function.

    Returns
    -------
    Any
        The result of the function call, or NotImplemented.

    Raises
    ------
    NotImplementedError
        If the function is not implemented in autograd.numpy.

    See Also
    --------
    https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
    """
    if not all(t in TidyArrayBox.type_mappings for t in types):
        return NotImplemented

    module_name = func.__module__

    if module_name.startswith("numpy"):
        anp_module_name = "autograd." + module_name
    else:
        return NotImplemented

    # Use the cached module if available
    anp_module = _autograd_module_cache.get(anp_module_name)
    if anp_module is None:
        try:
            anp_module = importlib.import_module(anp_module_name)
            _autograd_module_cache[anp_module_name] = anp_module
        except ImportError:
            return NotImplemented

    f = getattr(anp_module, func.__name__, None)
    if f is None:
        return NotImplemented

    if f.__name__ == "nanmean":  # somehow xarray always dispatches to nanmean
        f = anp.mean
        kwargs.pop("dtype", None)  # autograd mean vjp doesn't support dtype

    return f(*args, **kwargs)


def __array_ufunc__(
    self: Any,
    ufunc: Callable,
    method: str,
    *inputs: Any,
    **kwargs: Dict[str, Any],
) -> Any:
    """
    Handle the dispatch of NumPy ufuncs to autograd's numpy implementation.

    Parameters
    ----------
    self : Any
        The instance of the class.
    ufunc : Callable
        The universal function being called.
    method : str
        The method of the ufunc being called.
    inputs : Any
        The input arguments to the ufunc.
    kwargs : Dict[str, Any]
        The keyword arguments to the ufunc.

    Returns
    -------
    Any
        The result of the ufunc call, or NotImplemented.

    See Also
    --------
    https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    if method != "__call__":
        return NotImplemented

    ufunc_name = ufunc.__name__

    anp_ufunc = getattr(anp, ufunc_name, None)
    if anp_ufunc is not None:
        return anp_ufunc(*inputs, **kwargs)

    return NotImplemented


def item(self):
    if self.size != 1:
        raise ValueError("Can only convert an array of size 1 to a scalar")
    return anp.ravel(self)[0]


TidyArrayBox._tidy = True
TidyArrayBox.from_arraybox = from_arraybox
TidyArrayBox.__array_namespace__ = lambda self, *, api_version=None: anp
TidyArrayBox.__array_ufunc__ = __array_ufunc__
TidyArrayBox.__array_function__ = __array_function__
TidyArrayBox.__repr__ = str
TidyArrayBox.real = property(anp.real)
TidyArrayBox.imag = property(anp.imag)
TidyArrayBox.conj = anp.conj
TidyArrayBox.item = item
