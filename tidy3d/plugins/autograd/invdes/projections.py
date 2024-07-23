import autograd.numpy as np
from numpy.typing import NDArray


def ramp_projection(array: NDArray, width: float = 0.1, center: float = 0.5) -> NDArray:
    """Apply a piecewise linear ramp projection to an array.

    This function performs a ramp projection on the input array, modifying its values
    based on the specified width and center. Values within the range
    [center - width/2, center + width/2] are linearly transformed, while values
    outside this range are projected to 0 or 1. The input and output is assumed to be
    within the range [0, 1].

    Parameters
    ----------
    array : NDArray
        The input array to be projected.
    width : float, optional
        The width of the ramp. Default is 0.1.
    center : float, optional
        The center of the ramp. Default is 0.5.

    Returns
    -------
    NDArray
        The array after applying the ramp projection.
    """
    ll = array <= (center - width / 2)
    cc = (array > (center - width / 2)) & (array < (center + width / 2))
    rr = array >= (center + width / 2)

    return np.concatenate(
        [
            np.zeros(array[ll].size),
            (array[cc] - (center - width / 2)) / width,
            np.ones(array[rr].size),
        ]
    )


def tanh_projection(array: NDArray, beta: float = 1.0, eta: float = 0.5) -> NDArray:
    """Apply a tanh-based soft-thresholding projection to an array.

    This function performs a tanh projection on the input array, which is a common
    soft-thresholding scheme used in topology optimization. The projection modifies
    the values of the array based on the specified `beta` and `eta` parameters.

    Parameters
    ----------
    array : NDArray
        The input array to be projected.
    beta : float, optional
        The steepness of the projection. Higher values result in a sharper transition.
        Default is 1.0.
    eta : float, optional
        The midpoint of the projection. Default is 0.5.

    Returns
    -------
    NDArray
        The array after applying the tanh projection.
    """
    if beta == 0:
        return array
    num = np.tanh(beta * eta) + np.tanh(beta * (array - eta))
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / denom
