from functools import reduce
from typing import Tuple, Callable, Iterable, Union

import numpy as np

from tidy3d.plugins.autograd.types import KernelType


def _kernel_circular(size: Tuple[int, ...]) -> np.ndarray:
    """Create a circular kernel in n dimensions.

    Parameters
    ----------
    size : Tuple[int, ...]
        The size of the circular kernel in pixels for each dimension.

    Returns
    -------
    np.ndarray
        An n-dimensional array representing the circular kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    squared_distances = sum(grid**2 for grid in grids)
    kernel = squared_distances <= 1
    return kernel


def _kernel_conic(size: Tuple[int, ...]) -> np.ndarray:
    """Create a conic kernel in n dimensions.

    Parameters
    ----------
    size : Tuple[int, ...]
        The size of the conic kernel in pixels for each dimension.

    Returns
    -------
    np.ndarray
        An n-dimensional array representing the conic kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    dists = sum(grid**2 for grid in grids)
    kernel = np.maximum(0, 1 - np.sqrt(dists))
    return kernel


def make_kernel(
    kernel_type: KernelType, size: Tuple[int, ...], normalize: bool = True
) -> np.ndarray:
    """Create a kernel based on the specified type in n dimensions.

    Parameters
    ----------
    kernel_type : KernelType
        The type of kernel to create ('circular' or 'conic').
    size : Tuple[int, ...]
        The size of the kernel in pixels for each dimension.
    normalize : bool, optional
        Whether to normalize the kernel so that it sums to 1. Default is True.

    Returns
    -------
    np.ndarray
        An n-dimensional array representing the specified type of kernel.
    """
    if kernel_type == "circular":
        kernel = _kernel_circular(size)
    elif kernel_type == "conic":
        kernel = _kernel_conic(size)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    if normalize:
        kernel /= np.sum(kernel)

    return kernel


def chain(*funcs: Union[Callable, Iterable[Callable]]):
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

    def chained(array: np.ndarray):
        return reduce(lambda x, y: y(x), funcs, array)

    return chained
