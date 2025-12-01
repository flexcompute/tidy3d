from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import NDArray


def symmetrize_mirror(array: NDArray, axis: int | tuple[int, int]) -> NDArray:
    """
    Symmetrizes the parameter array by averaging the mirrored parts of the array. The axis argument specifies the
    symmetry axis.

    Parameters
    ----------
    array : NDArray
        The input array to be symmetrized.
    axis: int | tuple[int, int]
        The symmetry axis. This can either be a single axis (x=0 or y=1) or a tuple of both axes.

    Returns
    -------
    NDArray
        The array after applying the mirror symmetrization.

    Example
    -------
        >>> import autograd.numpy as np
        >>> from tidy3d.plugins.autograd.invdes.symmetries import symmetrize_mirror
        >>> arr = np.asarray([
        ...     [1, 2],
        ...     [3, 4]
        ... ])
        >>> res_x = symmetrize_mirror(arr, axis=0)
        >>> res_y = symmetrize_mirror(arr, axis=1)
        >>> assert np.all(np.isclose(res_x, np.asarray([
        ...     [2, 3],
        ...     [2, 3]
        ... ])))
        >>> assert np.all(np.isclose(res_y, np.asarray([
        ...     [1.5, 1.5],
        ...     [3.5, 3.5]
        ... ])))
    """
    if isinstance(axis, int) and (axis != 0 and axis != 1):
        raise ValueError(f"Invalid axis: expected 0, 1, or Sequence thereof, but got {axis}")
    if array.ndim != 2:
        raise ValueError(f"Invalid array shape: {array.shape}. Need 2d array.")
    if isinstance(axis, Sequence) and (
        len(axis) != 2 or axis[0] not in [0, 1] or axis[1] not in [0, 1] or axis[0] == axis[1]
    ):
        raise ValueError(f"Invalid axis: expected 0, 1, or Sequence thereof, but got {axis}")

    # Helper function to flip along a specific axis using slicing
    # Autograd supports slicing (e.g. ::-1) but lacks VJP for np.flip
    def flip_axis(arr, ax):
        if ax == 0:
            return arr[::-1, :]
        elif ax == 1:
            return arr[:, ::-1]
        return arr

    # Case 1: Symmetrize along both axes
    if isinstance(axis, Sequence):
        # Symmetrize along axis 0
        array = (array + flip_axis(array, 0)) / 2.0
        # Symmetrize along axis 1
        array = (array + flip_axis(array, 1)) / 2.0
        return array

    # Case 2: Symmetrize along a single axis
    return (array + flip_axis(array, axis)) / 2.0


def symmetrize_rotation(array: NDArray) -> NDArray:
    """
    Symmetrizes the parameter array by averaging over all four 90-degree rotations.
    The input array must be square.

    Parameters
    ----------
    array : NDArray
        The input array to be symmetrized.

    Returns
    -------
    NDArray
        The array after applying rotational symmetrization.

    Example
    -------
        >>> import autograd.numpy as np
        >>> from tidy3d.plugins.autograd.invdes.symmetries import symmetrize_rotation
        >>> arr = np.asarray([
        ...     [0, 0, 0],
        ...     [1, 5, 2],
        ...     [8, 1, 8]
        ... ])
        >>> res = symmetrize_rotation(arr)
        >>> assert np.all(np.isclose(res, np.asarray([
        ...     [4, 1, 4],
        ...     [1, 5, 1],
        ...     [4, 1, 4]
        ... ])))
    """
    if array.ndim != 2:
        raise ValueError(f"Invalid array shape: {array.shape}. Need 2d array.")

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"Invalid array shape: {array.shape}. Array must be square for rotational symmetry."
        )

    # Average the array with its 90, 180, and 270 degree rotations
    # We manually implement rotations using Transpose (.T) and Slicing ([::-1])
    # to ensure full compatibility with autograd.

    # 0 degrees: array
    # 90 degrees CCW: Transpose -> Flip Rows (equivalent to array.T[::-1, :])
    rot90 = array.T[::-1, :]

    # 180 degrees: Flip Rows -> Flip Cols (equivalent to array[::-1, ::-1])
    rot180 = array[::-1, ::-1]

    # 270 degrees CCW: Transpose -> Flip Cols (equivalent to array.T[:, ::-1])
    rot270 = array.T[:, ::-1]

    return (array + rot90 + rot180 + rot270) / 4.0


def symmetrize_diagonal(array: NDArray, anti: bool = False) -> NDArray:
    """
    Symmetrizes the parameter array by averaging it with its transpose.
    Can symmetrize along the main diagonal or the anti-diagonal.

    Parameters
    ----------
    array : NDArray
        The input array to be symmetrized. Must be square.
    anti : bool, optional
        If False (default), symmetrizes along the main diagonal (top-left to bottom-right).
        If True, symmetrizes along the anti-diagonal (top-right to bottom-left).

    Returns
    -------
    NDArray
        The array after applying diagonal symmetrization.

    Example
    -------
        >>> import autograd.numpy as np
        >>> from tidy3d.plugins.autograd.invdes.symmetries import symmetrize_diagonal
        >>> arr = np.asarray([
        ...     [1, 2],
        ...     [4, 7],
        ... ])
        >>> res = symmetrize_diagonal(arr)
        >>> res_anti = symmetrize_diagonal(arr, anti=True)
        >>> assert np.all(np.isclose(res, np.asarray([
        ...     [1, 3],
        ...     [3, 7]
        ... ])))
        >>> assert np.all(np.isclose(res_anti, np.asarray([
        ...     [4, 2],
        ...     [4, 4]
        ... ])))
    """
    if array.ndim != 2:
        raise ValueError(f"Invalid array shape: {array.shape}. Need 2d array.")

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"Invalid array shape: {array.shape}. Array must be square for diagonal symmetry."
        )

    if anti:
        # Anti-diagonal symmetrization.
        # Mathematically equivalent to averaging A with the flip of A transposed.
        # np.flip(array.T) is equivalent to array.T[::-1, ::-1]
        anti_transpose = array.T[::-1, ::-1]
        return (array + anti_transpose) / 2.0

    # Standard main diagonal symmetry: (A + A_transpose) / 2
    return (array + array.T) / 2.0
