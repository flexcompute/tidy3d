from functools import partial
from typing import Tuple, Union

import numpy as np

from ..functions import convolve
from ..types import KernelType, PaddingType
from ..utilities import get_kernel_size_px, make_kernel


def _get_kernel_size(
    radius: Union[float, Tuple[float, ...]],
    dl: Union[float, Tuple[float, ...]],
    size_px: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    """Determine the kernel size based on the provided radius, grid spacing, or size in pixels.

    Parameters
    ----------
    radius : Union[float, Tuple[float, ...]]
        The radius of the kernel. Can be a scalar or a tuple.
    dl : Union[float, Tuple[float, ...]]
        The grid spacing. Can be a scalar or a tuple.
    size_px : Union[int, Tuple[int, ...]]
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple.

    Returns
    -------
    Tuple[int, ...]
        The size of the kernel in pixels for each dimension.

    Raises
    ------
    ValueError
        If neither 'size_px' nor both 'radius' and 'dl' are provided.
    """
    if size_px is not None:
        return (size_px,) if np.isscalar(size_px) else tuple(size_px)
    elif radius is not None and dl is not None:
        kernel_size = get_kernel_size_px(radius=radius, dl=dl)
        return (kernel_size,) if np.isscalar(kernel_size) else tuple(kernel_size)
    else:
        raise ValueError("Either 'size_px' or both 'radius' and 'dl' must be provided.")


def make_filter(
    radius: Union[float, Tuple[float, ...]] = None,
    dl: Union[float, Tuple[float, ...]] = None,
    *,
    size_px: Union[int, Tuple[int, ...]] = None,
    normalize: bool = True,
    padding: PaddingType = "reflect",
    filter_type: KernelType,
):
    """Create a filter function based on the specified kernel type and size.

    Parameters
    ----------
    radius : Union[float, Tuple[float, ...]], optional
        The radius of the kernel. Can be a scalar or a tuple. Default is None.
    dl : Union[float, Tuple[float, ...]], optional
        The grid spacing. Can be a scalar or a tuple. Default is None.
    size_px : Union[int, Tuple[int, ...]], optional
        The size of the kernel in pixels for each dimension. Can be a scalar or a tuple. Default is None.
    normalize : bool, optional
        Whether to normalize the kernel so that it sums to 1. Default is True.
    padding : PaddingType, optional
        The padding mode to use. Default is "reflect".
    filter_type : KernelType
        The type of kernel to create (`circular` or `conic`).

    Returns
    -------
    function
        A function that applies the created filter to an input array.
    """
    _kernel = {}

    kernel_size = _get_kernel_size(radius, dl, size_px)

    def _filter(array):
        original_shape = array.shape
        squeezed_array = np.squeeze(array)

        if squeezed_array.ndim not in _kernel:
            ks = kernel_size
            if len(ks) != squeezed_array.ndim:
                ks *= squeezed_array.ndim
            _kernel[squeezed_array.ndim] = make_kernel(
                kernel_type=filter_type, size=ks, normalize=normalize
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
