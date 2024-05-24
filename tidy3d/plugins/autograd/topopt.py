from typing import Union
import autograd.numpy as np


def ramp_projection(array: np.ndarray, width: float = 0.1, center: float = 0.5):
    """Apply a piecewise linear ramp projection to an array.

    This function performs a ramp projection on the input array, modifying its values
    based on the specified width and center. Values within the range
    [center - width/2, center + width/2] are linearly transformed, while values
    outside this range are projected to 0 or 1. The input array is assumed to be
    within the range [0, 1].

    Parameters
    ----------
    array : np.ndarray
        The input array to be projected.
    width : float, optional
        The width of the ramp. Default is 0.1.
    center : float, optional
        The center of the ramp. Default is 0.5.

    Returns
    -------
    np.ndarray
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


def tanh_projection(array: np.ndarray, beta: float = 1.0, eta: float = 0.5) -> np.ndarray:
    """Apply a tanh-based soft-thresholding projection to an array.

    This function performs a tanh projection on the input array, which is a common
    soft-thresholding scheme used in topology optimization. The projection modifies
    the values of the array based on the specified `beta` and `eta` parameters.

    Parameters
    ----------
    array : np.ndarray
        The input array to be projected.
    beta : float, optional
        The steepness of the projection. Higher values result in a sharper transition.
        Default is 1.0.
    eta : float, optional
        The midpoint of the projection. Default is 0.5.

    Returns
    -------
    np.ndarray
        The array after applying the tanh projection.
    """
    if beta == 0:
        return array
    num = np.tanh(beta * eta) + np.tanh(beta * (array - eta))
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / denom


def rescale(
    array: np.ndarray, out_min: float, out_max: float, in_min: float = 0.0, in_max: float = 1.0
) -> np.ndarray:
    """
    Rescale an array from an arbitrary input range to an arbitrary output range.

    Parameters
    ----------
    array : np.ndarray
        The input array to be rescaled.
    out_min : float
        The minimum value of the output range.
    out_max : float
        The maximum value of the output range.
    in_min : float, optional
        The minimum value of the input range. Default is 0.0.
    in_max : float, optional
        The maximum value of the input range. Default is 1.0.

    Returns
    -------
    np.ndarray
        The rescaled array.
    """
    scaled = (array - in_min) / (in_max - in_min)
    return scaled * (out_max - out_min) + out_min


def threshold(
    array: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, level: Union[float, None] = None
) -> np.ndarray:
    """Apply a threshold to an array, setting values below the threshold to `vmin` and values above to `vmax`.

    Parameters
    ----------
    array : np.ndarray
        The input array to be thresholded.
    vmin : float, optional
        The value to assign to elements below the threshold. Default is 0.0.
    vmax : float, optional
        The value to assign to elements above the threshold. Default is 1.0.
    level : Union[float, None], optional
        The threshold level. If None, the threshold is set to the midpoint between `vmin` and `vmax`. Default is None.

    Returns
    -------
    np.ndarray
        The thresholded array.
    """
    if level is None:
        level = (vmin + vmax) / 2
    return np.where(array < level, vmin, vmax)
