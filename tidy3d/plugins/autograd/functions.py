from typing import Iterable, List, Literal, Tuple, Union

import autograd.numpy as np
from autograd.scipy.signal import convolve as convolve_ag
from numpy.typing import NDArray

from tidy3d.components.autograd.functions import add_at, interpn, trapz

from .types import PaddingType

__all__ = [
    "interpn",
    "trapz",
    "add_at",
    "pad",
    "convolve",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "grey_closing",
    "morphological_gradient",
    "morphological_gradient_internal",
    "morphological_gradient_external",
    "rescale",
    "threshold",
]


def _pad_indices(n: int, pad_width: Tuple[int, int], mode: PaddingType) -> NDArray:
    """Compute the indices to pad an array along a single axis based on the padding mode.

    Parameters
    ----------
    n : int
        The size of the axis to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of the axis.
    mode : PaddingType
        The padding mode to use.

    Returns
    -------
    NDArray
        The indices for padding along the axis.
    """
    total_pad = pad_width[0] + pad_width[1]
    if n == 0:
        # If the axis length is zero, return zeros for indices
        return np.zeros(total_pad, dtype=int)

    idx = np.arange(-pad_width[0], n + pad_width[1])
    if mode == "constant":
        # For constant mode, indices outside the array bounds are ignored
        pass
    elif mode == "edge":
        idx = np.clip(idx, 0, n - 1)
    elif mode == "reflect":
        period = 2 * n - 2 if n > 1 else 1
        idx = np.mod(idx, period)
        idx = np.where(idx >= n, period - idx, idx)
    elif mode == "symmetric":
        period = 2 * n if n > 1 else 1
        idx = np.mod(idx, period)
        idx = np.where(idx >= n, period - idx - 1, idx)
    elif mode == "wrap":
        idx = np.mod(idx, n)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    return idx


def _pad(
    array: NDArray,
    pad_width: Tuple[int, int],
    axis: int,
    *,
    mode: PaddingType = "constant",
    constant_value: float = 0.0,
) -> NDArray:
    """Pad an array along a specified axis with a given mode and pad width.

    Parameters
    ----------
    array : NDArray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of the axis.
    axis : int
        The axis along which to pad.
    mode : PaddingType = "constant"
        The padding mode to use.
    constant_value : float = 0.0
        The constant value to pad with when mode is 'constant'.

    Returns
    -------
    NDArray
        The padded array.
    """
    n = array.shape[axis]
    if mode == "constant":
        p = [(0, 0)] * array.ndim
        p[axis] = pad_width
        return np.pad(array, p, mode="constant", constant_values=constant_value)
    else:
        idx = _pad_indices(n, pad_width, mode)
        indexer = [slice(None)] * array.ndim
        indexer[axis] = idx
        return array[tuple(indexer)]


def pad(
    array: NDArray,
    pad_width: Union[int, Tuple[int, int]],
    *,
    mode: PaddingType = "constant",
    axis: Union[int, Iterable[int], None] = None,
    constant_value: float = 0.0,
) -> NDArray:
    """Pad an array along specified axes with a given mode and padding width.

    Parameters
    ----------
    array : NDArray
        The input array to pad.
    pad_width : Union[int, Tuple[int, int]]
        The number of values padded to the edges of each axis. If an integer is provided,
        it is used for both the left and right sides. If a tuple is provided, it specifies
        the padding for the left and right sides respectively.
    mode : PaddingType = "constant"
        The padding mode to use.
    axis : Union[int, Iterable[int], None] = None
        The axis or axes along which to pad. If None, padding is applied to all axes.
    constant_value : float = 0.0
        The value to set the padded values for "constant" mode.

    Returns
    -------
    NDArray
        The padded array.

    Raises
    ------
    ValueError
        If the padding width has more than two elements or if padding is negative.
    IndexError
        If an axis is out of range for the array dimensions.
    """
    pad_width = np.atleast_1d(pad_width)
    if pad_width.size == 1:
        pad_width = (pad_width[0], pad_width[0])
    elif pad_width.size == 2:
        pad_width = tuple(pad_width)
    else:
        raise ValueError(f"Padding width must have one or two elements, got {pad_width.size}.")

    if any(p < 0 for p in pad_width):
        raise ValueError("Padding must be non-negative.")
    if all(p == 0 for p in pad_width):
        return array

    if axis is None:
        axes = range(array.ndim)
    else:
        axes = [axis] if isinstance(axis, int) else axis

    for ax in axes:
        if ax < 0:
            ax += array.ndim
        if ax < 0 or ax >= array.ndim:
            raise IndexError(f"Axis {ax} out of range for array with {array.ndim} dimensions.")

        array = _pad(
            array,
            pad_width,
            axis=ax,
            mode=mode,
            constant_value=constant_value,
        )
    return array


def convolve(
    array: NDArray,
    kernel: NDArray,
    *,
    padding: PaddingType = "constant",
    axes: Union[Tuple[List[int], List[int]], None] = None,
    mode: Literal["full", "valid", "same"] = "same",
) -> NDArray:
    """Convolve an array with a given kernel.

    Parameters
    ----------
    array : NDArray
        The input array to be convolved.
    kernel : NDArray
        The kernel to convolve with the input array. All dimensions of the kernel must be odd.
    padding : PaddingType = "constant"
        The padding mode to use.
    axes : Union[Tuple[List[int], List[int]], None] = None
        The axes along which to perform the convolution.
    mode : Literal["full", "valid", "same"] = "same"
        The convolution mode.

    Returns
    -------
    NDArray
        The result of the convolution.

    Raises
    ------
    ValueError
        If any dimension of the kernel is even.
        If the dimensions of the kernel do not match the dimensions of the array.
    """
    if any(k % 2 == 0 for k in kernel.shape):
        raise ValueError(f"All kernel dimensions must be odd, got {kernel.shape}.")

    if kernel.ndim != array.ndim and axes is None:
        raise ValueError(
            f"Kernel dimensions must match array dimensions, got kernel {kernel.shape} and array {array.shape}."
        )

    if mode in ("same", "full"):
        kernel_dims = kernel.shape if axes is None else [kernel.shape[d] for d in axes[1]]
        pad_widths = [(ks // 2, ks // 2) for ks in kernel_dims]
        for axis, pad_width in enumerate(pad_widths):
            array = pad(array, pad_width, mode=padding, axis=axis)
        mode = "valid" if mode == "same" else mode

    return convolve_ag(array, kernel, axes=axes, mode=mode)


def grey_dilation(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey dilation on an array.

    Parameters
    ----------
    array : NDArray
        The input array to perform grey dilation on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The result of the grey dilation operation.

    Raises
    ------
    ValueError
        If both `size` and `structure` are None.
    """
    if size is None and structure is None:
        raise ValueError("Either size or structure must be provided.")

    if size is not None:
        size = np.atleast_1d(size)
        shape = (size[0], size[-1])
        nb = np.zeros(shape)
    elif np.all(structure == 0):
        nb = np.zeros_like(structure)
    else:
        nb = np.copy(structure)
        nb[structure == 0] = -maxval

    h, w = nb.shape
    bias = np.reshape(nb, (-1, 1, 1))
    kernel = np.reshape(np.eye(h * w), (h * w, h, w))

    array = convolve(array, kernel, axes=((0, 1), (1, 2)), padding=mode) + bias
    return np.max(array, axis=0)


def grey_erosion(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey erosion on an array.

    Parameters
    ----------
    array : NDArray
        The input array to perform grey dilation on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The result of the grey dilation operation.

    Raises
    ------
    ValueError
        If both `size` and `structure` are None.
    """
    if size is None and structure is None:
        raise ValueError("Either size or structure must be provided.")

    if size is not None:
        size = np.atleast_1d(size)
        shape = (size[0], size[-1])
        nb = np.zeros(shape)
    elif np.all(structure == 0):
        nb = np.zeros_like(structure)
    else:
        nb = np.copy(structure)
        nb[structure == 0] = -maxval

    h, w = nb.shape
    bias = np.reshape(nb, (-1, 1, 1))
    kernel = np.reshape(np.eye(h * w), (h * w, h, w))

    array = convolve(array, kernel, axes=((0, 1), (1, 2)), padding=mode) - bias
    return np.min(array, axis=0)


def grey_opening(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey opening on an array.

    Parameters
    ----------
    array : NDArray
        The input array to perform grey opening on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The result of the grey opening operation.
    """
    array = grey_erosion(array, size, structure, mode=mode, maxval=maxval)
    array = grey_dilation(array, size, structure, mode=mode, maxval=maxval)
    return array


def grey_closing(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey closing on an array.

    Parameters
    ----------
    array : NDArray
        The input array to perform grey closing on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The result of the grey closing operation.
    """
    array = grey_dilation(array, size, structure, mode=mode, maxval=maxval)
    array = grey_erosion(array, size, structure, mode=mode, maxval=maxval)
    return array


def morphological_gradient(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the morphological gradient of an array.

    Parameters
    ----------
    array : NDArray
        The input array to compute the morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The morphological gradient of the input array.
    """
    return grey_dilation(array, size, structure, mode=mode, maxval=maxval) - grey_erosion(
        array, size, structure, mode=mode, maxval=maxval
    )


def morphological_gradient_internal(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the internal morphological gradient of an array.

    Parameters
    ----------
    array : NDArray
        The input array to compute the internal morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The internal morphological gradient of the input array.
    """
    return array - grey_erosion(array, size, structure, mode=mode, maxval=maxval)


def morphological_gradient_external(
    array: NDArray,
    size: Union[Union[int, Tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the external morphological gradient of an array.

    Parameters
    ----------
    array : NDArray
        The input array to compute the external morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[NDArray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    NDArray
        The external morphological gradient of the input array.
    """
    return grey_dilation(array, size, structure, mode=mode, maxval=maxval) - array


def rescale(
    array: NDArray, out_min: float, out_max: float, in_min: float = 0.0, in_max: float = 1.0
) -> NDArray:
    """
    Rescale an array from an arbitrary input range to an arbitrary output range.

    Parameters
    ----------
    array : NDArray
        The input array to be rescaled.
    out_min : float
        The minimum value of the output range.
    out_max : float
        The maximum value of the output range.
    in_min : float = 0.0
        The minimum value of the input range.
    in_max : float = 1.0
        The maximum value of the input range.

    Returns
    -------
    NDArray
        The rescaled array.
    """

    if in_min == in_max:
        raise ValueError(
            f"'in_min' ({in_min}) must not be equal to 'in_max' ({in_max}) "
            "to avoid division by zero."
        )
    if out_min >= out_max:
        raise ValueError(f"'out_min' ({out_min}) must be less than 'out_max' ({out_max}).")
    if in_min >= in_max:
        raise ValueError(f"'in_min' ({in_min}) must be less than 'in_max' ({in_max}).")

    scaled = (array - in_min) / (in_max - in_min)
    return scaled * (out_max - out_min) + out_min


def threshold(
    array: NDArray, vmin: float = 0.0, vmax: float = 1.0, level: Union[float, None] = None
) -> NDArray:
    """Apply a threshold to an array, setting values below the threshold to `vmin` and values above to `vmax`.

    Parameters
    ----------
    array : NDArray
        The input array to be thresholded.
    vmin : float = 0.0
        The value to assign to elements below the threshold.
    vmax : float = 1.0
        The value to assign to elements above the threshold.
    level : Union[float, None] = None
        The threshold level. If None, the threshold is set to the midpoint between `vmin` and `vmax`.

    Returns
    -------
    NDArray
        The thresholded array.
    """
    if vmin >= vmax:
        raise ValueError(
            f"Invalid threshold range: 'vmin' ({vmin}) must be smaller than 'vmax' ({vmax})."
        )

    if level is None:
        level = (vmin + vmax) / 2
    elif not (vmin <= level <= vmax):
        raise ValueError(
            f"Invalid threshold level: 'level' ({level}) must be "
            f"between 'vmin' ({vmin}) and 'vmax' ({vmax})."
        )

    return np.where(array < level, vmin, vmax)
