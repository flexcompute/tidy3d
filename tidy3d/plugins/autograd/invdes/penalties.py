from typing import Tuple

import autograd.numpy as np

from .parametrizations import make_filter_and_project


def make_erosion_dilation_penalty(
    filter_size: Tuple[int, ...],
    beta: float = 100.0,
    eta: float = 0.5,
    delta_eta: float = 0.01,
):
    """Computes a penalty for erosion/dilation of a parameter map not being unity.

    Accepts a parameter array normalized between 0 and 1. Uses filtering and projection methods
    to erode and dilate the features within this array. Measures the change in the array after
    eroding and dilating (and also dilating and eroding). Returns a penalty proportional to the
    magnitude of this change. The amount of change under dilation and erosion is minimized if
    the structure has large feature sizes and large radius of curvature relative to the length scale.

    Parameters
    ----------
    filter_size : Tuple[int, ...]
        The size of the filter to be used for erosion and dilation.
    beta : float, optional
        Strength of the tanh projection. Default is 100.0.
    eta : float, optional
        Midpoint of the tanh projection. Default is 0.5.
    delta_eta : float, optional
        The binarization threshold for erosion and dilation operations. Default is 0.01.

    Returns
    -------
    Callable
        A function that computes the erosion/dilation penalty for a given array.
    """
    filtproj = make_filter_and_project(filter_size, beta, eta)
    eta_dilate = 0.0 + delta_eta
    eta_eroded = 1.0 - delta_eta

    def _dilate(array: np.ndarray, beta: float):
        return filtproj(array, beta=beta, eta=eta_dilate)

    def _erode(array: np.ndarray, beta: float):
        return filtproj(array, beta=beta, eta=eta_eroded)

    def _open(array: np.ndarray, beta: float):
        return _dilate(_erode(array, beta=beta), beta=beta)

    def _close(array: np.ndarray, beta: float):
        return _erode(_dilate(array, beta=beta), beta=beta)

    def _erosion_dilation_penalty(array: np.ndarray, beta: float = beta):
        diff = _close(array, beta) - _open(array, beta)

        if not np.any(diff):
            return 0.0

        return np.linalg.norm(diff) / np.sqrt(diff.size)

    return _erosion_dilation_penalty
