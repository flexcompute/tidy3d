import itertools

import autograd.numpy as anp
import scipy.ndimage
from autograd.extend import defvjp, primitive
from scipy.interpolate import RegularGridInterpolator

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)


def interpn(points, values, xi, *, method="linear"):
    if method == "nearest":
        interp_fn = _evaluate_nearest
    elif method == "linear":
        interp_fn = _evaluate_linear
    else:
        raise ValueError
    interpolator = RegularGridInterpolator(points, values, method=method)
    grid = anp.meshgrid(*xi, indexing="ij")
    grid, shape, ndim, nans, out_of_bounds = interpolator._prepare_xi(tuple(grid))
    indices, norm_distances = interpolator._find_indices(grid.T)
    return interp_fn(indices, norm_distances, values, shape, ndim)


def _evaluate_nearest(indices, norm_distances, values, shape, ndim):
    idx_res = tuple(anp.where(yi <= 0.5, i, i + 1) for i, yi in zip(indices, norm_distances))
    return values[idx_res]


def _evaluate_linear(indices, norm_distances, values, shape, ndim):
    # slice for broadcasting over trailing dimensions in self.values
    vslice = (slice(None),) + (None,) * (values.ndim - len(indices))

    # Compute shifting up front before zipping everything together
    shift_norm_distances = [1 - yi for yi in norm_distances]
    shift_indices = [i + 1 for i in indices]

    # The formula for linear interpolation in 2d takes the form:
    # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
    #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
    #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
    #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
    # We pair i with 1 - yi (zipped1) and i + 1 with yi (zipped2)
    zipped1 = zip(indices, shift_norm_distances)
    zipped2 = zip(shift_indices, norm_distances)

    # Take all products of zipped1 and zipped2 and iterate over them
    # to get the terms in the above formula. This corresponds to iterating
    # over the vertices of a hypercube.
    hypercube = itertools.product(*zip(zipped1, zipped2))
    value = anp.array([0.0])
    for h in hypercube:
        edge_indices, weights = zip(*h)
        weight = anp.array([1.0])
        for w in weights:
            weight = weight * w
        term = anp.array(values[edge_indices]) * weight[vslice]
        value = value + term  # cannot use += because broadcasting
    return anp.reshape(value, shape[:-1] + values.shape[ndim:])
