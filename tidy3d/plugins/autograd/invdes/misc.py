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
    array : np.ndarray
        The input array for which the grey indicator is to be calculated.

    Returns
    -------
    float
        The calculated grey indicator.
    """
    return np.mean(4 * array * (1 - array))
