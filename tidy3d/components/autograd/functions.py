import itertools

import autograd.numpy as anp
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from .types import InterpolationType


def _evaluate_nearest(
    indices: NDArray[np.int64], norm_distances: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    idx_res = tuple(anp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances))
    return values[idx_res]


def _evaluate_linear(
    indices: NDArray[np.int64], norm_distances: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    _slice = (slice(None),) + (None,) * (values.ndim - len(indices))

    ix = zip(indices, (1 - yi for yi in norm_distances))
    iy = zip((i + 1 for i in indices), norm_distances)

    value = anp.zeros(1)
    for h in itertools.product(*zip(ix, iy)):
        edge_indices, weights = zip(*h)
        weight = anp.ones(1)
        for w in weights:
            weight = weight * w
        # term = anp.array(values[edge_indices]) * weight[_slice]
        # term = np.array(values[edge_indices]) * weight[_slice]
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
    method : InterpolationType, optional
        The method of interpolation to perform. Supported are "linear" and "nearest". Default is "linear".

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
    scipy.interpolate.interpn : Interpolation on a rectilinear grid in arbitrary dimensions.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html
    """
    if method == "nearest":
        interp_fn = _evaluate_nearest
    elif method == "linear":
        interp_fn = _evaluate_linear
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    itrp = RegularGridInterpolator(points, values, method=method)
    grid = anp.meshgrid(*xi, indexing="ij")
    grid, shape, ndim, nans, _ = itrp._prepare_xi(tuple(grid))
    indices, norm_distances = itrp._find_indices(grid.T)

    result = interp_fn(indices, norm_distances, values)
    # nans = anp.reshape(nans, (-1,) + (1,) * (result.ndim - 1))
    # result = anp.where(nans, np.nan, result)
    return anp.reshape(result, shape[:-1] + values.shape[ndim:])
