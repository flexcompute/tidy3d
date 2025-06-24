from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Literal, Union

import autograd.numpy as np
import numpy as onp
from autograd import jacobian
from autograd.extend import defvjp, primitive
from autograd.scipy.signal import convolve as convolve_ag
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

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

    idx = numpy_module.arange(-pad_width[0], n + pad_width[1])

    if mode == "constant":
        return idx
    if mode == "edge":
        return numpy_module.clip(idx, 0, n - 1)
    if mode == "reflect":
        period = 2 * n - 2 if n > 1 else 1
        idx = numpy_module.mod(idx, period)
        return numpy_module.where(idx >= n, period - idx, idx)
    if mode == "symmetric":
        period = 2 * n if n > 1 else 1
        idx = numpy_module.mod(idx, period)
        return numpy_module.where(idx >= n, period - idx - 1, idx)
    if mode == "wrap":
        return numpy_module.mod(idx, n)

    raise ValueError(f"Unsupported padding mode: {mode}")


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
    axes: Union[tuple[list[int], list[int]], None] = None,
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
    return scaled * (out_max - out_min) + out_min


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
