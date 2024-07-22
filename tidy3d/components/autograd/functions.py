import itertools

import autograd.numpy as anp
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from .types import InterpolationType


def _evaluate_nearest(
    indices: NDArray[np.int64], norm_distances: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Perform nearest neighbor interpolation in an n-dimensional space.

    This function determines the nearest neighbor in a grid for a given point
    and returns the corresponding value from the input array.

    Parameters
    ----------
    indices : NDArray[np.int64]
        Indices of the lower bounds of the grid cell containing the interpolation point.
    norm_distances : NDArray[np.float64]
        Normalized distances from the lower bounds of the grid cell to the
        interpolation point, for each dimension.
    values : NDArray[np.float64]
        The n-dimensional array of values to interpolate from.

    Returns
    -------
    NDArray[np.float64]
        The value of the nearest neighbor to the interpolation point.
    """
    idx_res = tuple(anp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances))
    return values[idx_res]


def _evaluate_linear(
    indices: NDArray[np.int64], norm_distances: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Perform linear interpolation in an n-dimensional space.

    This function calculates the linearly interpolated value at a point in an
    n-dimensional grid, given the indices of the surrounding grid points and
    the normalized distances to these points.
    The multi-linear interpolation is implemented by computing a weighted
    average of the values at the vertices of the hypercube surrounding the
    interpolation point.

    Parameters
    ----------
    indices : NDArray[np.int64]
        Indices of the lower bounds of the grid cell containing the interpolation point.
    norm_distances : NDArray[np.float64]
        Normalized distances from the lower bounds of the grid cell to the
        interpolation point, for each dimension.
    values : NDArray[np.float64]
        The n-dimensional array of values to interpolate from.

    Returns
    -------
    NDArray[np.float64]
        The interpolated value at the desired point.
    """
    # Create a slice object for broadcasting over trailing dimensions
    _slice = (slice(None),) + (None,) * (values.ndim - len(indices))

    # Prepare iterables for lower and upper bounds of the hypercube
    ix = zip(indices, (1 - yi for yi in norm_distances))
    iy = zip((i + 1 for i in indices), norm_distances)

    # Initialize the result
    value = anp.zeros(1)

    # Iterate over all vertices of the hypercube
    for h in itertools.product(*zip(ix, iy)):
        edge_indices, weights = zip(*h)

        # Compute the weight for this vertex
        weight = anp.ones(1)
        for w in weights:
            weight = weight * w

        # Compute the contribution of this vertex and add it to the result
        term = values[edge_indices] * weight[_slice]
        value = value + term

    return value


def interpn(
    points: tuple[NDArray[np.float64], ...],
    values: NDArray[np.float64],
    xi: tuple[NDArray[np.float64], ...],
    *,
    method: InterpolationType = "linear",
) -> NDArray[np.float64]:
    """Interpolate over a rectilinear grid in arbitrary dimensions.

    This function mirrors the interface of `scipy.interpolate.interpn` but is differentiable with autograd.

    Parameters
    ----------
    points : tuple[NDArray[np.float64], ...]
        The points defining the rectilinear grid in n dimensions.
    values : NDArray[np.float64]
        The data values on the rectilinear grid.
    xi : tuple[NDArray[np.float64], ...]
        The coordinates to sample the gridded data at.
    method : InterpolationType = "linear"
        The method of interpolation to perform. Supported are "linear" and "nearest".

    Returns
    -------
    NDArray[np.float64]
        The interpolated values.

    Raises
    ------
    ValueError
        If the interpolation method is not supported.

    See Also
    --------
    `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html>`_
    """
    if method == "nearest":
        interp_fn = _evaluate_nearest
    elif method == "linear":
        interp_fn = _evaluate_linear
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    itrp = RegularGridInterpolator(points, values, method=method)
    grid = anp.meshgrid(*xi, indexing="ij")

    # Prepare the grid for interpolation
    # This step reshapes the grid, checks for NaNs and out-of-bounds values
    # It returns:
    #   - reshaped grid
    #   - original shape
    #   - number of dimensions
    #   - boolean array indicating NaN positions
    #   - (discarded) boolean array for out-of-bounds values
    grid, shape, ndim, nans, _ = itrp._prepare_xi(tuple(grid))

    # Find the indices of the grid cells containing the interpolation points
    # and calculate the normalized distances (ranging from 0 at lower grid point to 1
    # at upper grid point) within these cells
    indices, norm_distances = itrp._find_indices(grid.T)

    result = interp_fn(indices, norm_distances, values)
    nans = anp.reshape(nans, (-1,) + (1,) * (result.ndim - 1))
    result = anp.where(nans, np.nan, result)
    return anp.reshape(result, shape[:-1] + values.shape[ndim:])


__all__ = [
    "interpn",
]
