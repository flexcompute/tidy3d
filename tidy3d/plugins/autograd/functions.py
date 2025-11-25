from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Literal, SupportsInt, Union

import autograd.numpy as np
import numpy as onp
from autograd import jacobian
from autograd.extend import defvjp, primitive
from autograd.numpy.fft import fftn, ifftn
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from numpy.fft import irfftn, rfftn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.fft import next_fast_len

from tidy3d.components.autograd.functions import add_at, interpn, trapz

from .types import PaddingType

__all__ = [
    "add_at",
    "convolve",
    "grey_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "interpn",
    "morphological_gradient",
    "morphological_gradient_external",
    "morphological_gradient_internal",
    "pad",
    "rescale",
    "smooth_max",
    "smooth_min",
    "threshold",
    "trapz",
]


def _normalize_axes(
    ndim_array: int,
    ndim_kernel: int,
    axes: Union[tuple[Iterable[SupportsInt], Iterable[SupportsInt]], None],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalize the axes specification for convolution."""

    def _normalize_single_axis(ax: SupportsInt, ndim: int, kind: str) -> int:
        if not isinstance(ax, int):
            try:
                ax = int(ax)
            except Exception as e:
                raise TypeError(f"Axis {ax!r} could not be converted to an integer.") from e

        if not -ndim <= ax < ndim:
            raise ValueError(f"Invalid axis {ax} for {kind} with ndim {ndim}.")
        return ax + ndim if ax < 0 else ax

    if axes is None:
        if ndim_array != ndim_kernel:
            raise ValueError(
                "Kernel dimensions must match array dimensions when 'axes' is not provided, "
                f"got array ndim {ndim_array} and kernel ndim {ndim_kernel}."
            )
        axes_array = tuple(range(ndim_array))
        axes_kernel = tuple(range(ndim_kernel))
        return axes_array, axes_kernel

    if len(axes) != 2:
        raise ValueError("'axes' must be a tuple of two iterable collections of axis indices.")

    axes_array_raw, axes_kernel_raw = axes

    axes_array = tuple(_normalize_single_axis(ax, ndim_array, "array") for ax in axes_array_raw)
    axes_kernel = tuple(_normalize_single_axis(ax, ndim_kernel, "kernel") for ax in axes_kernel_raw)

    if len(axes_array) != len(axes_kernel):
        raise ValueError(
            "The number of convolution axes for the array and kernel must be the same, "
            f"got {len(axes_array)} and {len(axes_kernel)}."
        )

    if len(set(axes_array)) != len(axes_array) or len(set(axes_kernel)) != len(axes_kernel):
        raise ValueError("Convolution axes must be unique for both the array and the kernel.")

    return axes_array, axes_kernel


def _fft_convolve_general(
    array: NDArray,
    kernel: NDArray,
    axes_array: tuple[int, ...],
    axes_kernel: tuple[int, ...],
    mode: Literal["full", "valid"],
) -> NDArray:
    """Perform convolution using FFT along the specified axes."""

    num_conv_axes = len(axes_array)

    if num_conv_axes == 0:
        array_shape = array.shape
        kernel_shape = kernel.shape
        result = np.multiply(
            array.reshape(array_shape + (1,) * kernel.ndim),
            kernel.reshape((1,) * array.ndim + kernel_shape),
        )
        return result.reshape(array_shape + kernel_shape)

    ignore_axes_array = tuple(ax for ax in range(array.ndim) if ax not in axes_array)
    ignore_axes_kernel = tuple(ax for ax in range(kernel.ndim) if ax not in axes_kernel)

    new_order_array = ignore_axes_array + axes_array
    new_order_kernel = ignore_axes_kernel + axes_kernel

    array_reordered = np.transpose(array, new_order_array) if array.ndim else array
    kernel_reordered = np.transpose(kernel, new_order_kernel) if kernel.ndim else kernel

    num_batch_array = len(ignore_axes_array)
    num_batch_kernel = len(ignore_axes_kernel)

    array_conv_shape = array_reordered.shape[num_batch_array:]
    kernel_conv_shape = kernel_reordered.shape[num_batch_kernel:]

    if any(d <= 0 for d in array_conv_shape + kernel_conv_shape):
        raise ValueError("Convolution dimensions must be positive; got zero-length axis.")

    fft_axes = tuple(range(-num_conv_axes, 0))
    fft_shape = [next_fast_len(n + k - 1) for n, k in zip(array_conv_shape, kernel_conv_shape)]
    use_real_fft = fft_shape[-1] % 2 == 0  # only applicable in this case

    fft_fun = rfftn if use_real_fft else fftn
    array_fft = fft_fun(array_reordered, fft_shape, axes=fft_axes)
    kernel_fft = fft_fun(kernel_reordered, fft_shape, axes=fft_axes)

    if num_batch_kernel:
        array_batch_shape = array_fft.shape[:num_batch_array]
        conv_shape = array_fft.shape[num_batch_array:]
        array_fft = np.reshape(
            array_fft,
            array_batch_shape + (1,) * num_batch_kernel + conv_shape,
        )

    if num_batch_array:
        kernel_batch_shape = kernel_fft.shape[:num_batch_kernel]
        conv_shape = kernel_fft.shape[num_batch_kernel:]
        kernel_fft = np.reshape(
            kernel_fft,
            (1,) * num_batch_array + kernel_batch_shape + conv_shape,
        )
    use_real_fft = fft_shape[-1] % 2 == 0

    product = array_fft * kernel_fft

    ifft_fun = irfftn if use_real_fft else ifftn
    full_result = ifft_fun(product, fft_shape, axes=fft_axes)

    if mode == "full":
        result = full_result
    elif mode == "valid":
        valid_slices = [slice(None)] * full_result.ndim
        for axis_offset, (array_dim, kernel_dim) in enumerate(
            zip(array_conv_shape, kernel_conv_shape)
        ):
            start = int(min(array_dim, kernel_dim) - 1)
            length = int(abs(array_dim - kernel_dim) + 1)
            axis = full_result.ndim - num_conv_axes + axis_offset
            valid_slices[axis] = slice(start, start + length)
        result = full_result[tuple(valid_slices)]
    else:
        raise ValueError(f"Unsupported convolution mode '{mode}'.")

    return np.real(result)


def _get_pad_indices(
    n: int,
    pad_width: tuple[int, int],
    *,
    mode: PaddingType,
    numpy_module,
) -> NDArray:
    """Compute the indices to pad an array along a single axis based on the padding mode.

    Parameters
    ----------
    n : int
        The size of the axis to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of the axis.
    mode : PaddingType
        The padding mode to use.
    numpy_module : module
        The numpy module to use (either `numpy` or `autograd.numpy`).

    Returns
    -------
    np.ndarray
        The indices for padding along the axis.
    """
    total_pad = sum(pad_width)
    if n == 0:
        return numpy_module.zeros(total_pad, dtype=int)

    pad_left, pad_right = pad_width
    if mode == "constant":
        return numpy_module.arange(-pad_left, n + pad_right)

    try:
        indices = onp.pad(onp.arange(n), (pad_left, pad_right), mode=mode)
    except ValueError as error:
        raise ValueError(f"Unsupported padding mode: {mode}") from error
    return numpy_module.asarray(indices, dtype=int)


def pad(
    array: NDArray,
    pad_width: Union[int, tuple[int, int]],
    *,
    mode: PaddingType = "constant",
    axis: Union[int, Iterable[int], None] = None,
    constant_value: float = 0.0,
) -> NDArray:
    """Pad an array along specified axes with a given mode and padding width.

    Parameters
    ----------
    array : np.ndarray
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
    np.ndarray
        The padded array.

    Raises
    ------
    ValueError
        If the padding width has more than two elements or if padding is negative.
    IndexError
        If an axis is out of range for the array dimensions.
    """
    pad_width = np.atleast_1d(pad_width)
    if pad_width.size > 2:
        raise ValueError(f"Padding width must have one or two elements, got {pad_width.size}.")
    pad_tuple = (pad_width[0], pad_width[0]) if pad_width.size == 1 else tuple(pad_width)

    if any(p < 0 for p in pad_tuple):
        raise ValueError("Padding must be non-negative.")
    if all(p == 0 for p in pad_tuple):
        return array

    axes = range(array.ndim) if axis is None else [axis] if isinstance(axis, int) else axis
    axes = [ax + array.ndim if ax < 0 else ax for ax in axes]
    if any(ax < 0 or ax >= array.ndim for ax in axes):
        raise IndexError(f"Axis out of range for array with {array.ndim} dimensions.")

    result = array
    for ax in axes:
        if mode == "constant":
            padding = [(0, 0)] * result.ndim
            padding[ax] = pad_tuple
            result = np.pad(result, padding, mode="constant", constant_values=constant_value)
        else:
            idx = _get_pad_indices(result.shape[ax], pad_tuple, mode=mode, numpy_module=np)
            indexer = [slice(None)] * result.ndim
            indexer[ax] = idx
            result = result[tuple(indexer)]
    return result


def convolve(
    array: NDArray,
    kernel: NDArray,
    *,
    padding: PaddingType = "constant",
    axes: Union[tuple[list[SupportsInt], list[SupportsInt]], None] = None,
    mode: Literal["full", "valid", "same"] = "same",
) -> NDArray:
    """Convolve an array with a given kernel.

    Parameters
    ----------
    array : np.ndarray
        The input array to be convolved.
    kernel : np.ndarray
        The kernel to convolve with the input array. All dimensions of the kernel must be odd.
    padding : PaddingType = "constant"
        The padding mode to use.
    axes : Union[Tuple[List[int], List[int]], None] = None
        The axes along which to perform the convolution.
    mode : Literal["full", "valid", "same"] = "same"
        The convolution mode.

    Returns
    -------
    np.ndarray
        The result of the convolution.

    Raises
    ------
    ValueError
        If any dimension of the kernel is even.
        If the dimensions of the kernel do not match the dimensions of the array.
    """
    if any(k % 2 == 0 for k in kernel.shape):
        raise ValueError(f"All kernel dimensions must be odd, got {kernel.shape}.")

    axes_array, axes_kernel = _normalize_axes(array.ndim, kernel.ndim, axes)

    working_array = array
    effective_mode = mode

    if mode in ["same", "full"]:
        for ax_array, ax_kernel in zip(axes_array, axes_kernel):
            pad_width = (
                kernel.shape[ax_kernel] // 2 if mode == "same" else kernel.shape[ax_kernel] - 1
            )
            if pad_width > 0:
                working_array = pad(
                    working_array, (pad_width, pad_width), mode=padding, axis=ax_array
                )
        effective_mode = "valid"

    return _fft_convolve_general(working_array, kernel, axes_array, axes_kernel, effective_mode)


def _get_footprint(size, structure, maxval):
    """Helper to generate the morphological footprint from size or structure."""
    if size is None and structure is None:
        raise ValueError("Either size or structure must be provided.")
    if size is not None and structure is not None:
        raise ValueError("Cannot specify both size and structure.")
    if structure is None:
        size_np = onp.atleast_1d(size)
        shape = (size_np[0], size_np[-1]) if size_np.size > 1 else (size_np[0], size_np[0])
        nb = onp.zeros(shape)
    else:
        structure_np = getval(structure)
        nb = onp.copy(structure_np)
        nb[structure_np == 0] = -maxval
    if nb.shape[0] % 2 == 0 or nb.shape[1] % 2 == 0:
        raise ValueError(f"Structuring element dimensions must be odd, got {nb.shape}.")
    return nb


@primitive
def grey_dilation(
    array: NDArray,
    size: Union[int, tuple[int, int], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey dilation on an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to perform grey dilation on.
    size : Union[Union[int, tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
        If a single integer is provided, a square structuring element is created.
        For 1D arrays, use a tuple (size, 1) or (1, size) for horizontal or vertical operations.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
        For 1D operations on 2D arrays, use a 2D structure with one dimension being 1.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The result of the grey dilation operation.

    Raises
    ------
    ValueError
        If both `size` and `structure` are None, or if the structuring element has even dimensions.
    """
    nb = _get_footprint(size, structure, maxval)
    h, w = nb.shape

    padded_array = pad(array, (h // 2, h // 2), mode=mode, axis=0)
    padded_array = pad(padded_array, (w // 2, w // 2), mode=mode, axis=1)

    padded_array_np = getval(padded_array)

    windows = sliding_window_view(padded_array_np, window_shape=(h, w))
    dilated_windows = windows + nb
    return onp.max(dilated_windows, axis=(-2, -1))


def _vjp_maker_dilation(ans, array, size=None, structure=None, *, mode="reflect", maxval=1e4):
    """VJP for the custom grey_dilation primitive."""
    nb = _get_footprint(size, structure, maxval)
    h, w = nb.shape

    padded_array = pad(array, (h // 2, h // 2), mode=mode, axis=0)
    padded_array = pad(padded_array, (w // 2, w // 2), mode=mode, axis=1)

    padded_array_np = getval(padded_array)
    in_h, in_w = getval(array).shape

    windows = sliding_window_view(padded_array_np, window_shape=(h, w))
    dilated_windows = windows + nb

    output_reshaped = ans[..., None, None]
    is_max_mask = (dilated_windows == output_reshaped).astype(onp.float64)

    # normalize the gradient for cases where multiple elements are the maximum.
    # When multiple elements in a window equal the maximum value, the gradient
    # is distributed equally among them. This ensures gradient conservation.
    # Note: Values can never exceed maxval in the output since we add structure
    # values (capped at maxval) to the input array values.
    multiplicity = onp.sum(is_max_mask, axis=(-2, -1), keepdims=True)
    is_max_mask /= onp.maximum(multiplicity, 1)

    def vjp(g):
        g_reshaped = g[..., None, None]
        grad_windows = g_reshaped * is_max_mask

        grad_padded = onp.zeros_like(padded_array_np)

        # create broadcastable indices for the scatter-add operation
        i = onp.arange(in_h)[:, None, None, None]
        j = onp.arange(in_w)[None, :, None, None]
        u = onp.arange(h)[None, None, :, None]
        v = onp.arange(w)[None, None, None, :]

        onp.add.at(grad_padded, (i + u, j + v), grad_windows)

        pad_h, pad_w = h // 2, w // 2

        # for constant padding, we can just slice the gradient
        if mode == "constant":
            return grad_padded[pad_h : pad_h + in_h, pad_w : pad_w + in_w]

        # for other modes, we need to sum gradients from padded regions by unpadding each axis
        grad_unpadded_w = onp.zeros((in_h + 2 * pad_h, in_w))
        padded_indices_w = _get_pad_indices(in_w, (pad_w, pad_w), mode=mode, numpy_module=onp)
        row_indices_w = onp.arange(in_h + 2 * pad_h)[:, None]
        onp.add.at(grad_unpadded_w, (row_indices_w, padded_indices_w), grad_padded)

        grad_unpadded_hw = onp.zeros((in_h, in_w))
        padded_indices_h = _get_pad_indices(in_h, (pad_h, pad_h), mode=mode, numpy_module=onp)[
            :, None
        ]
        col_indices_h = onp.arange(in_w)[None, :]
        onp.add.at(grad_unpadded_hw, (padded_indices_h, col_indices_h), grad_unpadded_w)

        return grad_unpadded_hw

    return vjp


defvjp(grey_dilation, _vjp_maker_dilation, argnums=[0])


def grey_erosion(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey erosion on an array.

    This function is implemented via duality, calling `grey_dilation` internally.

    Parameters
    ----------
    array : np.ndarray
        The input array to perform grey erosion on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The result of the grey erosion operation.
    """
    if structure is not None:
        structure = structure[::-1, ::-1]

    return -grey_dilation(
        -array,
        size=size,
        structure=structure,
        mode=mode,
        maxval=maxval,
    )


def grey_opening(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey opening on an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to perform grey opening on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The result of the grey opening operation.
    """
    array = grey_erosion(array, size, structure, mode=mode, maxval=maxval)
    array = grey_dilation(array, size, structure, mode=mode, maxval=maxval)
    return array


def grey_closing(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Perform grey closing on an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to perform grey closing on.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The result of the grey closing operation.
    """
    array = grey_dilation(array, size, structure, mode=mode, maxval=maxval)
    array = grey_erosion(array, size, structure, mode=mode, maxval=maxval)
    return array


def morphological_gradient(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the morphological gradient of an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to compute the morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The morphological gradient of the input array.
    """
    return grey_dilation(array, size, structure, mode=mode, maxval=maxval) - grey_erosion(
        array, size, structure, mode=mode, maxval=maxval
    )


def morphological_gradient_internal(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the internal morphological gradient of an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to compute the internal morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
        The internal morphological gradient of the input array.
    """
    return array - grey_erosion(array, size, structure, mode=mode, maxval=maxval)


def morphological_gradient_external(
    array: NDArray,
    size: Union[Union[int, tuple[int, int]], None] = None,
    structure: Union[NDArray, None] = None,
    *,
    mode: PaddingType = "reflect",
    maxval: float = 1e4,
) -> NDArray:
    """Compute the external morphological gradient of an array.

    Parameters
    ----------
    array : np.ndarray
        The input array to compute the external morphological gradient of.
    size : Union[Union[int, Tuple[int, int]], None] = None
        The size of the structuring element. If None, `structure` must be provided.
    structure : Union[np.ndarray, None] = None
        The structuring element. If None, `size` must be provided.
    mode : PaddingType = "reflect"
        The padding mode to use.
    maxval : float = 1e4
        Value to assume for infinite elements in the kernel.

    Returns
    -------
    np.ndarray
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
    array : np.ndarray
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
    np.ndarray
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
    result = scaled * (out_max - out_min) + out_min

    return np.clip(result, out_min, out_max)


def threshold(
    array: NDArray, vmin: float = 0.0, vmax: float = 1.0, level: Union[float, None] = None
) -> NDArray:
    """Apply a threshold to an array, setting values below the threshold to `vmin` and values above to `vmax`.

    Parameters
    ----------
    array : np.ndarray
        The input array to be thresholded.
    vmin : float = 0.0
        The value to assign to elements below the threshold.
    vmax : float = 1.0
        The value to assign to elements above the threshold.
    level : Union[float, None] = None
        The threshold level. If None, the threshold is set to the midpoint between `vmin` and `vmax`.

    Returns
    -------
    np.ndarray
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


def smooth_max(
    x: NDArray, tau: float = 1.0, axis: Union[int, tuple[int, ...], None] = None
) -> float:
    """Compute the smooth maximum of an array using temperature parameter tau.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    tau : float = 1.0
        Temperature parameter controlling smoothness. Larger values make the maximum smoother.
    axis : Union[int, Tuple[int, ...], None] = None
        Axis or axes over which the smooth maximum is computed. By default, the smooth maximum is computed over the entire array.

    Returns
    -------
    np.ndarray
        The smooth maximum of the input array.
    """
    return tau * logsumexp(x / tau, axis=axis)


def smooth_min(
    x: NDArray, tau: float = 1.0, axis: Union[int, tuple[int, ...], None] = None
) -> float:
    """Compute the smooth minimum of an array using temperature parameter tau.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    tau : float = 1.0
        Temperature parameter controlling smoothness. Larger values make the minimum smoother.
    axis : Union[int, Tuple[int, ...], None] = None
        Axis or axes over which the smooth minimum is computed. By default, the smooth minimum is computed over the entire array.

    Returns
    -------
    np.ndarray
        The smooth minimum of the input array.
    """
    return -smooth_max(-x, tau, axis=axis)


def least_squares(
    func: Callable[[NDArray, float], NDArray],
    x: NDArray,
    y: NDArray,
    initial_guess: tuple[float, ...],
    max_iterations: int = 100,
    tol: float = 1e-6,
) -> NDArray:
    """Perform least squares fitting to find the best-fit parameters for a model function.

    Parameters
    ----------
    func : Callable[[np.ndarray, float], np.ndarray]
        The model function to fit. It should accept the independent variable `x` and a tuple of parameters,
        and return the predicted dependent variable values.
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    initial_guess : Tuple[float, ...]
        Initial guess for the parameters to be optimized.
    max_iterations : int = 100
        Maximum number of iterations for the optimization process.
    tol : float = 1e-6
        Tolerance for convergence. The optimization stops when the change in parameters is below this threshold.

    Returns
    -------
    np.ndarray
        The optimized parameters that best fit the model to the data.

    Raises
    ------
    np.linalg.LinAlgError
        If the optimization does not converge within the specified number of iterations.

    Example
    -------
    >>> import numpy as np
    >>> def linear_model(x, a, b):
    ...     return a * x + b
    >>> x_data = np.linspace(0, 10, 50)
    >>> y_data = 2.0 * x_data - 3.0
    >>> initial_guess = (0.0, 0.0)
    >>> params = least_squares(linear_model, x_data, y_data, initial_guess)
    >>> print(params)
    [ 2. -3.]
    """
    params = np.array(initial_guess, dtype="f8")
    jac = jacobian(lambda params: func(x, *params))

    for _ in range(max_iterations):
        residuals = y - func(x, *params)
        jacvec = jac(params)
        pseudo_inv = np.linalg.pinv(jacvec)
        delta = np.dot(pseudo_inv, residuals)
        params = params + delta
        if np.linalg.norm(delta) < tol:
            break

    return params
