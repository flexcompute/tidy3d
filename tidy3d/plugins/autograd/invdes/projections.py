from __future__ import annotations

import autograd.numpy as np
from numpy.typing import NDArray

from tidy3d.plugins.autograd.constants import BETA_DEFAULT, ETA_DEFAULT


def ramp_projection(array: NDArray, width: float = 0.1, center: float = 0.5) -> NDArray:
    """Apply a piecewise linear ramp projection to an array.

    This function performs a ramp projection on the input array, modifying its values
    based on the specified width and center. Values within the range
    [center - width/2, center + width/2] are linearly transformed, while values
    outside this range are projected to 0 or 1. The input and output is assumed to be
    within the range [0, 1].

    Parameters
    ----------
    array : np.ndarray
        The input array to be projected.
    width : float = 0.1
        The width of the ramp.
    center : float 0.5
        The center of the ramp.

    Returns
    -------
    np.ndarray
        The array after applying the ramp projection.
    """
    ll = array <= (center - width / 2)
    cc = (array > (center - width / 2)) & (array < (center + width / 2))
    rr = array >= (center + width / 2)

    return np.concatenate(
        [
            np.zeros(array[ll].size),
            (array[cc] - (center - width / 2)) / width,
            np.ones(array[rr].size),
        ]
    )


def tanh_projection(
    array: NDArray, beta: float = BETA_DEFAULT, eta: float = ETA_DEFAULT
) -> NDArray:
    """Apply a tanh-based soft-thresholding projection to an array.

    This function performs a tanh projection on the input array, which is a common
    soft-thresholding scheme used in topology optimization. The projection modifies
    the values of the array based on the specified `beta` and `eta` parameters.

    Parameters
    ----------
    array : np.ndarray
        The input array to be projected.
    beta : float = BETA_DEFAULT
        The steepness of the projection. Higher values result in a sharper transition.
    eta : float = ETA_DEFAULT
        The midpoint of the projection.

    Returns
    -------
    np.ndarray
        The array after applying the tanh projection.
    """
    if beta == 0:
        return array
    if beta == np.inf:
        return np.where(array > eta, 1.0, 0.0)
    num = np.tanh(beta * eta) + np.tanh(beta * (array - eta))
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / denom


def smoothed_projection(
    array: NDArray,
    beta: float = BETA_DEFAULT,
    eta: float = ETA_DEFAULT,
    scaling_factor=1.0,
) -> NDArray:
    """
    Apply a subpixel-smoothed projection method.
    The subpixel-smoothed projection method is discussed in [1]_ as follows:

    This projection method eliminates discontinuities by applying first-order
    smoothing at material boundaries through analytical fill factors. Unlike
    traditional quadrature approaches, it works with maximum projection strength
    (:math:`\\beta = \\infty`) and derives closed-form expressions for interfacial regions.

    Prerequisites: input fields must be pre-filtered for continuity (for example
    using a conic filter).

    The algorithm detects whether boundaries intersect grid cells. When interfaces
    are absent, standard projection is applied. For cells containing boundaries,
    analytical fill ratios are computed to maintain gradient continuity as interfaces
    move through cells and traverse pixel centers. This enables arbitrarily large
    :math:`\\beta` values while preserving differentiability throughout the transition
    process.

    .. warning::
        This function assumes that the device is placed on a uniform grid. When using
        ```GridSpec.auto``` in the simulation, make sure to place a ``MeshOverrideStructure`` at
        the position of the optimized geometry.

    Parameters
    ----------
    array : np.ndarray
        The input array to be projected.
    beta : float = BETA_DEFAULT
        The steepness of the projection. Higher values result in a sharper transition.
    eta : float = ETA_DEFAULT
        The midpoint of the projection.
    scaling_factor: float = 1.0
        Optional scaling factor to adjust dx and dy to different resolutions.

    Example
    -------
        >>> import autograd.numpy as np
        >>> from tidy3d.plugins.autograd.invdes.filters import ConicFilter
        >>> arr = np.random.uniform(size=(50, 50))
        >>> filter = ConicFilter(kernel_size=5)
        >>> arr_filtered = filter(arr)
        >>> eta = 0.5  # center of projection
        >>> smoothed = smoothed_projection(arr_filtered, beta=np.inf, eta=eta)

    .. [1] A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and
       S. G. Johnson, "Unifying and accelerating level-set and density-based topology
       optimization by subpixel-smoothed projection," arXiv:2503.20189v3 [physics.optics]
       (2025).
    """
    # sanity checks
    if array.ndim != 2:
        raise ValueError(f"Smoothed projection expects a 2d-array, but got shape {array.shape=}")

    # smoothing kernel is circle (or ellipse for non-uniform grid)
    # we choose smoothing kernel with unit area, which is r~=0.56, a bit larger than (arbitrary) default r=0.55 in paper
    dx = dy = scaling_factor
    smooth_radius = np.sqrt(1 / np.pi) * scaling_factor

    original_projected = tanh_projection(array, beta=beta, eta=eta)

    # finite-difference spatial gradients
    rho_filtered_grad = np.gradient(array)
    rho_filtered_grad_helper = (rho_filtered_grad[0] / dx) ** 2 + (rho_filtered_grad[1] / dy) ** 2

    nonzero_norm = np.abs(rho_filtered_grad_helper) > 1e-10

    filtered_grad_norm = np.sqrt(np.where(nonzero_norm, rho_filtered_grad_helper, 1))
    filtered_grad_norm_eff = np.where(nonzero_norm, filtered_grad_norm, 1)

    # distance of pixel center to nearest interface
    distance = (eta - array) / filtered_grad_norm_eff

    needs_smoothing = nonzero_norm & (np.abs(distance) < smooth_radius)

    # double where trick
    d_rel = distance / smooth_radius
    polynom = np.where(
        needs_smoothing, 0.5 - 15 / 16 * d_rel + 5 / 8 * d_rel**3 - 3 / 16 * d_rel**5, 1.0
    )
    # F(-d)
    polynom_neg = np.where(
        needs_smoothing, 0.5 + 15 / 16 * d_rel - 5 / 8 * d_rel**3 + 3 / 16 * d_rel**5, 1.0
    )

    # two projections, one for lower and one for upper bound
    rho_filtered_minus = array - smooth_radius * filtered_grad_norm_eff * polynom
    rho_filtered_plus = array + smooth_radius * filtered_grad_norm_eff * polynom_neg
    rho_minus_eff_projected = tanh_projection(rho_filtered_minus, beta=beta, eta=eta)
    rho_plus_eff_projected = tanh_projection(rho_filtered_plus, beta=beta, eta=eta)

    # Smoothing is only applied to projections
    projected_smoothed = (1 - polynom) * rho_minus_eff_projected + polynom * rho_plus_eff_projected
    smoothed = np.where(
        needs_smoothing,
        projected_smoothed,
        original_projected,
    )
    return smoothed
