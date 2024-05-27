from typing import Tuple
from functools import partial

from ..functions import convolve
from ..types import KernelType, PaddingType
from ..utilities import make_kernel


def make_filter(
    filter_type: KernelType,
    size: Tuple[int, ...],
    *,
    normalize: bool = True,
    padding: PaddingType = "reflect",
):
    """Create a filter function based on the specified kernel type and size.

    Parameters
    ----------
    filter_type : KernelType
        The type of kernel to create (`circular` or `conic`).
    size : Tuple[int, ...]
        The size of the kernel in pixels for each dimension.
    normalize : bool, optional
        Whether to normalize the kernel so that it sums to 1. Default is True.
    padding : PadMode, optional
        The padding mode to use. Default is "reflect".

    Returns
    -------
    function
        A function that applies the created filter to an input array.
    """
    kernel = make_kernel(kernel_type=filter_type, size=size, normalize=normalize)

    def _filter(array):
        return convolve(array, kernel, padding=padding)

    return _filter


conic_filter = partial(make_filter, filter_type="conic")
conic_filter.__doc__ = """make_filter() with a default filter_type value of `conic`.

See Also
--------
make_filter : Function to create a filter based on the specified kernel type and size.
"""

circular_filter = partial(make_filter, filter_type="circular")
circular_filter.__doc__ = """make_filter() with a default filter_type value of `circular`.

See Also
--------
make_filter : Function to create a filter based on the specified kernel type and size.
"""
