from functools import partial
from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..functions import convolve
from ..types import KernelType, PaddingType
from ..utilities import make_kernel


def make_filter(
    filter_type: KernelType,
    size: Union[int, Tuple[int, ...]],
    *,
    normalize: bool = True,
    padding: PaddingType = "reflect",
) -> Callable:
    """Create a filter function based on the specified kernel type and size.

    Parameters
    ----------
    filter_type : KernelType
        The type of kernel to create (`circular` or `conic`).
    size : Union[int, Tuple[int, ...]]
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple.
    normalize : bool, optional
        Whether to normalize the kernel so that it sums to 1. Default is True.
    padding : PadMode, optional
        The padding mode to use. Default is "reflect".

    Returns
    -------
    function
        A function that applies the created filter to an input array.
    """
    _kernel = {}

    def _filter(array: NDArray) -> NDArray:
        original_shape = array.shape
        squeezed_array = np.squeeze(array)

        if squeezed_array.ndim not in _kernel:
            if np.isscalar(size):
                kernel_size = (size,) * squeezed_array.ndim
            else:
                kernel_size = size
            _kernel[squeezed_array.ndim] = make_kernel(
                kernel_type=filter_type, size=kernel_size, normalize=normalize
            )

        convolved_array = convolve(squeezed_array, _kernel[squeezed_array.ndim], padding=padding)
        return np.reshape(convolved_array, original_shape)

    return _filter


make_conic_filter = partial(make_filter, filter_type="conic")
make_conic_filter.__doc__ = """make_filter() with a default filter_type value of `conic`.

See Also
--------
make_filter : Function to create a filter based on the specified kernel type and size.
"""

make_circular_filter = partial(make_filter, filter_type="circular")
make_circular_filter.__doc__ = """make_filter() with a default filter_type value of `circular`.

See Also
--------
make_filter : Function to create a filter based on the specified kernel type and size.
"""
