import autograd.numpy as np
from numpy.typing import NDArray


def grey_indicator(array: NDArray) -> float:
    """Calculate the grey indicator for a given array.

    The grey indicator returns 1 for completely grey arrays (all 0.5) and 0 for
    perfectly binary arrays. It is calculated based on:
    Sigmund, Ole. "Morphology-based black and white filters for topology optimization."
    Structural and Multidisciplinary Optimization 33 (2007): 401-424.

    Parameters
    ----------
    array : NDArray
        The input array for which the grey indicator is to be calculated.

    Returns
    -------
    float
        The calculated grey indicator.
    """
    return np.mean(4 * array * (1 - array))


def get_kernel_size_px(radius: float, dl: float) -> int:
    """Calculate the size of the kernel in pixels based on the given radius and pixel size.

    Parameters
    ----------
    radius : float
        The radius of the kernel in micrometers.
    dl : float
        The size of each pixel in micrometers.

    Returns
    -------
    float
        The size of the kernel in pixels.
    """
    radius_px = np.ceil(radius / dl)
    return int(2 * radius_px + 1)
