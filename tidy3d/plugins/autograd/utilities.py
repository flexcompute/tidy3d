from __future__ import annotations

from collections.abc import Iterable
from functools import reduce, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

import autograd.numpy as anp
import numpy as np
import xarray as xr

from tidy3d.exceptions import Tidy3dError, format_chained_exception_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .types import KernelType

P = ParamSpec("P")
R = TypeVar("R")


def _kernel_circular(size: Iterable[int]) -> NDArray:
    """Create a circular kernel in n dimensions.

    Parameters
    ----------
    size : Iterable[int]
        The size of the circular kernel in pixels for each dimension.

    Returns
    -------
    NDArray
        An n-dimensional array representing the circular kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    squared_distances = sum(grid**2 for grid in grids)
    kernel = np.array(squared_distances <= 1, dtype=np.float64)
    return kernel


def _kernel_conic(size: Iterable[int]) -> NDArray:
    """Create a conic kernel in n dimensions.

    Parameters
    ----------
    size : Iterable[int]
        The size of the conic kernel in pixels for each dimension.

    Returns
    -------
    NDArray
        An n-dimensional array representing the conic kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    dists = sum(grid**2 for grid in grids)
    kernel = np.maximum(0, 1 - np.sqrt(dists))
    return kernel


def make_kernel(kernel_type: KernelType, size: Iterable[int], normalize: bool = True) -> NDArray:
    """Create a kernel based on the specified type in n dimensions.

    Parameters
    ----------
    kernel_type : KernelType
        The type of kernel to create ('circular' or 'conic').
    size : Iterable[int]
        The size of the kernel in pixels for each dimension.
    normalize : bool = True
        Whether to normalize the kernel so that it sums to 1.

    Returns
    -------
    NDArray
        An n-dimensional array representing the specified type of kernel.
    """
    if not all(np.issubdtype(type(dim), int) and dim > 0 for dim in size):
        raise ValueError("'size' must be an iterable of positive integers.")

    if kernel_type == "circular":
        kernel = _kernel_circular(size)
    elif kernel_type == "conic":
        kernel = _kernel_conic(size)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    if normalize:
        kernel /= np.sum(kernel)

    return kernel


def get_kernel_size_px(
    radius: float | Iterable[float] | None = None,
    dl: float | Iterable[float] | None = None,
) -> int | list[int]:
    """Calculate the kernel size in pixels based on the provided radius and grid spacing.

    Parameters
    ----------
    radius : Union[float, Iterable[float]] = None
        The radius of the kernel. Can be a scalar or an iterable of floats.
    dl : Union[float, Iterable[float]] = None
        The grid spacing. Can be a scalar or an iterable of floats.

    Returns
    -------
    Union[int, list[int]]
        The size of the kernel in pixels for each dimension. Returns an integer if the radius is scalar, otherwise a list of integers.

    Raises
    ------
    ValueError
        If either 'radius' or 'dl' is not provided.
    """
    if radius is None or dl is None:
        raise ValueError("Either 'size_px' or both 'radius' and 'dl' must be provided.")

    if np.isscalar(radius):
        radius = [radius] * len(dl) if isinstance(dl, Iterable) else [radius]  # type: ignore[list-item]
    if np.isscalar(dl):
        dl = [dl] * len(radius)  # type: ignore[list-item]

    radius_px = [np.ceil(r / g) for r, g in zip(radius, dl)]
    return (
        [int(2 * r_px + 1) for r_px in radius_px]
        if len(radius_px) > 1
        else int(2 * radius_px[0] + 1)
    )


def chain(*funcs: Callable | Iterable[Callable]) -> Callable[[NDArray], NDArray]:
    """Chain multiple functions together to apply them sequentially to an array.

    Parameters
    ----------
    funcs : Union[Callable, Iterable[Callable]]
        A variable number of functions or a single iterable of functions to be chained together.

    Returns
    -------
    Callable
        A function that takes an array and applies the chained functions to it sequentially.

    Examples
    --------
    >>> import numpy as np
    >>> from tidy3d.plugins.autograd.utilities import chain
    >>> def add_one(x):
    ...     return x + 1
    >>> def square(x):
    ...     return x ** 2
    >>> chained_func = chain(add_one, square)
    >>> array = np.array([1, 2, 3])
    >>> chained_func(array)
    array([ 4,  9, 16])

    >>> # Using a list of functions
    >>> funcs = [add_one, square]
    >>> chained_func = chain(funcs)
    >>> chained_func(array)
    array([ 4,  9, 16])
    """
    if len(funcs) == 1 and isinstance(funcs[0], Iterable):
        funcs = funcs[0]

    if not all(callable(f) for f in funcs):
        raise TypeError("All elements in funcs must be callable.")

    def chained(array: NDArray) -> NDArray:
        return reduce(lambda x, y: y(x), funcs, array)

    return chained


@overload
def scalar_objective(
    func: None = None, *, has_aux: bool = False
) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...


@overload
def scalar_objective(func: Callable[P, Any], *, has_aux: bool = False) -> Callable[P, Any]: ...


def scalar_objective(
    func: Callable[P, Any] | None = None, *, has_aux: bool = False
) -> Callable[..., Any]:
    """Decorator to ensure the objective function returns a real scalar value.

    This decorator wraps an objective function to ensure that its return value is a real scalar.
    If the function returns auxiliary data, it expects the return value to be a tuple of the form
    (result, aux_data).

    Parameters
    ----------
    func : Callable, optional
        The objective function to be decorated. If not provided, the decorator should be used with
        arguments.
    has_aux : bool = False
        If True, expects the function to return a tuple (result, aux_data).

    Returns
    -------
    Callable
        The wrapped function that ensures a real scalar return value. If `has_aux` is True, the
        wrapped function returns a tuple (result, aux_data).

    Raises
    ------
    Tidy3dError
        If the return value is not a real scalar, or if `has_aux` is True and the function does not return a tuple of length 2.
    """

    def decorator(f: Callable[P, Any]) -> Callable[P, Any]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            result = f(*args, **kwargs)
            aux_data = None

            if has_aux:
                if not isinstance(result, tuple) or len(result) != 2:
                    raise Tidy3dError(
                        "If 'has_aux' is True, the objective function must return a tuple of length 2."
                    )
                result, aux_data = result

            if isinstance(result, xr.DataArray):
                result = result.data

            result = anp.squeeze(result)

            try:
                result = result.item()
            except AttributeError:
                if not isinstance(result, (float, int)):
                    raise Tidy3dError(
                        "An objective function's return value must be a scalar, "
                        "a Python float/int, or an array containing a single element."
                    ) from None
            except ValueError as e:
                raise Tidy3dError(
                    format_chained_exception_message(
                        "An objective function's return value must be a scalar but got an "
                        f"array with shape {getattr(result, 'shape', 'N/A')}",
                        e,
                    )
                ) from e

            if not anp.isreal(result):
                raise Tidy3dError("An objective function's return value must be real.")

            return (result, aux_data) if aux_data is not None else result

        return wrapper

    return decorator(func) if func is not None else decorator
