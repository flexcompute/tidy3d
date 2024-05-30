from typing import Tuple

import autograd.numpy as np

from .filters import make_filter
from .projections import tanh_projection
from ..types import KernelType, PaddingType


def make_filter_and_project(
    filter_size: Tuple[int, ...],
    beta: float = 1.0,
    eta: float = 0.5,
    filter_type: KernelType = "conic",
    padding: PaddingType = "reflect",
):
    """Create a function that filters and projects an array.

    This is the standard filter-and-project scheme used in topology optimization.

    Parameters
    ----------
    filter_size : Tuple[int, ...]
        The size of the filter kernel in pixels.
    beta : float, optional
        The beta parameter for the tanh projection, by default 1.0.
    eta : float, optional
        The eta parameter for the tanh projection, by default 0.5.
    filter_type : KernelType, optional
        The type of filter kernel to use, by default "conic".
    padding : PaddingType, optional
        The padding type to use for the filter, by default "reflect".

    Returns
    -------
    function
        A function that takes an array and applies the filter and projection.
    """
    _filter = make_filter(filter_type, filter_size, padding=padding)

    def _filter_and_project(array: np.ndarray, beta: float = beta, eta: float = eta):
        array = _filter(array)
        array = tanh_projection(array, beta, eta)
        return array

    return _filter_and_project
