import numpy as np


from typing import Tuple


def make_kernel_circular(size: Tuple[int, ...], normalize: bool = True) -> np.ndarray:
    """Create a circular kernel in n dimensions.

    Parameters
    ----------
    size : Tuple[int, ...]
        The size of the circular kernel in pixels for each dimension.
    normalize : bool, optional
        Whether to normalize the kernel so that its sum is 1. Default is True.

    Returns
    -------
    np.ndarray
        An n-dimensional array representing the circular kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    squared_distances = sum(grid**2 for grid in grids)
    kernel = squared_distances <= 1
    if normalize:
        kernel /= np.sum(kernel)
    return kernel


def make_kernel_conic(size: Tuple[int, ...], normalize: bool = True) -> np.ndarray:
    """Create a conic kernel in n dimensions.

    Parameters
    ----------
    size : Tuple[int, ...]
        The size of the conic kernel in pixels for each dimension.
    normalize : bool, optional
        Whether to normalize the kernel so that its sum is 1. Default is True.

    Returns
    -------
    np.ndarray
        An n-dimensional array representing the conic kernel.
    """
    grids = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in size)]
    dists = sum(grid**2 for grid in grids)
    kernel = np.maximum(0, 1 - np.sqrt(dists))
    if normalize:
        kernel /= np.sum(kernel)
    return kernel
