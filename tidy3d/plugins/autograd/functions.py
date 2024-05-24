from typing import Iterable, Union, Tuple, Literal

import autograd.numpy as np
from autograd.scipy.signal import convolve as convolve_ag
from .types import _pad_modes

_mode_to_scipy = {
    "constant": "constant",
    "edge": "nearest",
    "reflect": "mirror",
    "symmetric": "reflect",
    "wrap": "wrap",
}


def _make_slices(rule: Union[int, slice], ndim: int, axis: int) -> Tuple[slice, ...]:
    """Create a tuple of slices for indexing an array.

    Parameters
    ----------
    rule : Union[int, slice]
        The rule to apply on the specified axis.
    ndim : int
        The number of dimensions of the array.
    axis : int
        The axis to which the rule should be applied.

    Returns
    -------
    Tuple[slice, ...]
        A tuple of slices for indexing.
    """
    return tuple(slice(None) if d != axis else rule for d in range(ndim))


def _constant_pad(
    array: np.ndarray, pad_width: Tuple[int, int], axis: int, *, constant_value: float = 0.0
) -> np.ndarray:
    """Pad an array with a constant value along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of each axis.
    axis : int
        The axis along which to pad.
    constant_value : float, optional
        The constant value to pad with. Default is 0.0.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    p = [(pad_width[0], pad_width[1]) if ax == axis else (0, 0) for ax in range(array.ndim)]
    return np.pad(array, p, mode="constant", constant_values=constant_value)


def _edge_pad(array: np.ndarray, pad_width: Tuple[int, int], axis: int) -> np.ndarray:
    """Pad an array using the `edge` mode along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of each axis.
    axis : int
        The axis along which to pad.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    left, right = (_make_slices(rule, array.ndim, axis) for rule in (0, -1))

    cat_arys = []
    if pad_width[0] > 0:
        cat_arys.append(np.stack(pad_width[0] * [array[left]], axis=axis))
    cat_arys.append(array)
    if pad_width[1] > 0:
        cat_arys.append(np.stack(pad_width[1] * [array[right]], axis=axis))

    return np.concatenate(cat_arys, axis=axis)


def _reflect_pad(array: np.ndarray, pad_width: Tuple[int, int], axis: int) -> np.ndarray:
    """Pad an array using the `reflect` mode along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of each axis.
    axis : int
        The axis along which to pad.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    left, right = (
        _make_slices(rule, array.ndim, axis)
        for rule in (slice(pad_width[0], 0, -1), slice(-2, -pad_width[1] - 2, -1))
    )
    return np.concatenate([array[left], array, array[right]], axis=axis)


def _symmetric_pad(array: np.ndarray, pad_width: Tuple[int, int], axis: int) -> np.ndarray:
    """Pad an array using the `symmetric` mode along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of each axis.
    axis : int
        The axis along which to pad.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    left, right = (
        _make_slices(rule, array.ndim, axis)
        for rule in (
            slice(pad_width[0] - 1, None, -1) if pad_width[0] > 0 else slice(0, 0),
            slice(-1, -pad_width[1] - 1, -1),
        )
    )
    return np.concatenate([array[left], array, array[right]], axis=axis)


def _wrap_pad(array: np.ndarray, pad_width: Tuple[int, int], axis: int) -> np.ndarray:
    """Pad an array using the `wrap` mode along a specified axis.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Tuple[int, int]
        The number of values padded to the edges of each axis.
    axis : int
        The axis along which to pad.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    left, right = (
        _make_slices(rule, array.ndim, axis)
        for rule in (
            slice(-pad_width[0], None) if pad_width[0] > 0 else slice(0, 0),
            slice(0, pad_width[1]) if pad_width[1] > 0 else slice(0, 0),
        )
    )
    return np.concatenate([array[left], array, array[right]], axis=axis)


_mode_map = {
    "constant": _constant_pad,
    "edge": _edge_pad,
    "reflect": _reflect_pad,
    "symmetric": _symmetric_pad,
    "wrap": _wrap_pad,
}


def pad(
    array: np.ndarray,
    pad_width: Union[int, Tuple[int, int]],
    *,
    mode: _pad_modes = "constant",
    axis: Union[int, Iterable[int], None] = None,
    constant_value: float = 0.0,
) -> np.ndarray:
    """Pad an array along a specified axis with a given mode and padding width.

    Parameters
    ----------
    array : np.ndarray
        The input array to pad.
    pad_width : Union[int, Tuple[int, int]]
        The number of values padded to the edges of each axis. If an integer is provided,
        it is used for both the left and right sides. If a tuple is provided, it specifies
        the padding for the left and right sides respectively.
    mode : _pad_modes, optional
        The padding mode to use. Default is "constant".
    axis : Union[int, Iterable[int], None], optional
        The axis or axes along which to pad. If None, padding is applied to all axes. Default is None.
    constant_value : float, optional
        The value to set the padded values for "constant" mode. Default is 0.0.

    Returns
    -------
    np.ndarray
        The padded array.

    Raises
    ------
    ValueError
        If the padding width has more than two elements or if padding is negative.
    NotImplementedError
        If padding larger than the input size is requested.
    KeyError
        If an unsupported padding mode is specified.
    IndexError
        If an axis is out of range for the array dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from tidy3d.plugins.autograd.functions import pad
    >>> array = np.array([[1, 2], [3, 4]])
    >>> pad(array, (1, 1), mode="constant", constant_value=0)
    array([[0, 0, 0, 0],
           [0, 1, 2, 0],
           [0, 3, 4, 0],
           [0, 0, 0, 0]])
    >>> pad(array, 1, mode="reflect")
    array([[4, 3, 4, 3],
           [2, 1, 2, 1],
           [4, 3, 4, 3],
           [2, 1, 2, 1]])
    """
    pad_width = np.atleast_1d(pad_width)

    if pad_width.size == 1:
        pad_width = np.array([pad_width[0], pad_width[0]])
    elif pad_width.size != 2:
        raise ValueError("Padding width must have one or two elements, got {pad_width.size}.")

    if any(any(p >= s for p in pad_width) for s in array.shape):
        raise NotImplementedError("Padding larger than the input size is not supported.")
    if any(p < 0 for p in pad_width):
        raise ValueError("Padding must be positive.")
    if all(p == 0 for p in pad_width):
        return array

    if axis is None:
        axis = range(array.ndim)

    try:
        pad_fun = _mode_map[mode]
    except KeyError as e:
        raise KeyError(f"Unsupported padding mode: {mode}.") from e

    for ax in np.atleast_1d(axis):
        if ax < 0:
            ax += array.ndim
        if ax < 0 or ax >= array.ndim:
            raise IndexError(f"Axis out of range for array with {array.ndim} dimensions.")

        array = pad_fun(array, pad_width, axis=ax)

    return array


def convolve(
    array: np.ndarray,
    kernel: np.ndarray,
    *,
    padding: _pad_modes = "constant",
    mode: Literal["full", "valid", "same"] = "same",
) -> np.ndarray:
    if any(k % 2 == 0 for k in kernel.shape):
        raise ValueError(f"All kernel dimensions must be odd, got {kernel.shape}.")

    if len(set(kernel.shape)) != 1:
        raise ValueError(f"All kernel dimensions must be the same, got {kernel.shape}.")

    if kernel.ndim != array.ndim:
        raise ValueError(
            f"Kernel dimensions must match array dimensions, got kernel {kernel.shape} and array {array.shape}."
        )

    if mode in ("same", "full"):
        array = pad(array, kernel.shape[0] // 2, mode=padding)
        mode = "valid" if mode == "same" else "full"

    return convolve_ag(array, kernel, mode=mode)


def grey_dilation(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    h, w = k.shape
    bias = np.reshape(np.where(k == 0, -1, 0), (-1, 1, 1))
    k = np.reshape(np.eye(h * w), (h * w, h, w))

    x = convolve(x, k, axes=([0, 1], [1, 2]), mode=mode) + bias
    x = np.max(x, axis=0) + 1

    return x


def grey_erosion(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    return -grey_dilation(-x, k, mode)


def grey_opening(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    x = grey_erosion(x, k, mode)
    x = grey_dilation(x, k, mode)
    return x


def grey_closing(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    x = grey_dilation(x, k, mode)
    x = grey_erosion(x, k, mode)
    return x


def morphological_gradient(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    return grey_dilation(x, k, mode) - grey_erosion(x, k, mode)


def morphological_gradient_internal(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    return x - grey_erosion(x, k, mode)


def morphological_gradient_external(x: np.ndarray, k: np.ndarray, mode="reflect") -> np.ndarray:
    return grey_dilation(x, k, mode) - x
