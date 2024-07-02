from typing import Tuple, Union

import autograd.numpy as np

from ..types import KernelType, PaddingType
from .filters import make_filter
from .projections import tanh_projection


def make_filter_and_project(
    radius: Union[float, Tuple[float, ...]] = None,
    dl: Union[float, Tuple[float, ...]] = None,
    *,
    size_px: Union[int, Tuple[int, ...]] = None,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_type: KernelType = "conic",
    padding: PaddingType = "reflect",
):
    """Create a function that filters and projects an array.

    This is the standard filter-and-project scheme used in topology optimization.

    Parameters
    ----------
    radius : Union[float, Tuple[float, ...]], optional
        The radius of the kernel. Can be a scalar or a tuple. Default is None.
    dl : Union[float, Tuple[float, ...]], optional
        The grid spacing. Can be a scalar or a tuple. Default is None.
    size_px : Union[int, Tuple[int, ...]], optional
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple. Default is None.
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
    _filter = make_filter(radius, dl, size_px=size_px, filter_type=filter_type, padding=padding)

    def _filter_and_project(array: np.ndarray, beta: float = beta, eta: float = eta):
        array = _filter(array)
        array = tanh_projection(array, beta, eta)
        return array

    return _filter_and_project
