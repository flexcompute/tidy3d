from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import projections as projection_numerics

from tidy3d.plugins.autograd.constants import BETA_DEFAULT, ETA_DEFAULT

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


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
    return projection_numerics.tanh_projection(array, beta=beta, eta=eta)


def smoothed_projection(
    array: NDArray,
    beta: float = BETA_DEFAULT,
    eta: float = ETA_DEFAULT,
    scaling_factor: float = 1.0,
    *,
    order: Literal[1, 2] = 1,
    periodic: bool = False,
) -> NDArray:
    """Apply a subpixel-smoothed projection method.
    The subpixel-smoothed projection method is discussed in [1]_ as follows:

    This projection method eliminates discontinuities by applying subpixel
    smoothing at material boundaries through analytical fill factors. Unlike
    traditional quadrature approaches, it works with maximum projection strength
    (:math:`\\beta = \\infty`) and derives closed-form expressions for interfacial regions.

    Prerequisites: input fields must be pre-filtered for continuity (for example
    using a conic filter).

    The algorithm detects whether boundaries intersect grid cells. When interfaces
    are absent, standard projection is applied. For cells containing boundaries,
    analytical fill ratios are computed to maintain gradient continuity as interfaces
    move through cells and traverse pixel centers. This enables arbitrarily large
    :math:`\\beta` values while preserving differentiability as isolated interfaces move.
    SSP1 can lose smoothness when interfaces merge.

    With ``order=2``, SSP2 [2]_ regularizes the boundary distance using the
    Frobenius norm of the filtered field's Hessian. This smooths topology
    transitions, such as merging interfaces, where the spatial gradient vanishes
    but the Hessian remains nonzero. Where both vanish, ordinary projection is used.
    Spatial derivatives are evaluated using finite differences, as in SSP1,
    rather than the bicubic interpolation used in [2]_.

    .. warning::
        This function assumes equal grid spacing on both axes. When using
        ```GridSpec.auto``` in the simulation, make sure to place a ``MeshOverrideStructure`` at
        the position of the optimized geometry.

    Parameters
    ----------
    array : np.ndarray
        The filtered 2D input array. Each axis must contain at least two samples
        for SSP1 and three samples for SSP2. Periodic differences require at
        least three samples along each axis for both orders.
    beta : float = BETA_DEFAULT
        The steepness of the projection. Higher values result in a sharper transition.
    eta : float = ETA_DEFAULT
        The midpoint of the projection.
    scaling_factor: float = 1.0
        Positive, finite grid spacing used for both axes and the smoothing radius.
        The common scale cancels analytically for SSP2. SSP1 retains its gradient
        cutoff of ``1e-5`` in these spatial units, so scaling can affect SSP1 near
        that cutoff.
    order : Literal[1, 2] = 1
        Use first-order SSP (1) or Hessian-regularized SSP2 (2).
    periodic : bool = False
        Use centered periodic spatial differences on both axes. Otherwise use
        centered interior differences and one-sided differences at the edges.

    Returns
    -------
    np.ndarray
        The projected density, with the same shape as the input.

    Example
    -------
        >>> import autograd.numpy as np
        >>> from tidy3d.plugins.autograd.invdes.filters import ConicFilter
        >>> arr = np.random.uniform(size=(50, 50))
        >>> filter = ConicFilter(kernel_size=5)
        >>> arr_filtered = filter(arr)
        >>> eta = 0.5  # center of projection
        >>> smoothed = smoothed_projection(arr_filtered, beta=np.inf, eta=eta)
        >>> smoothed2 = smoothed_projection(arr_filtered, beta=np.inf, eta=eta, order=2)

    .. [1] A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and
       S. G. Johnson, "Unifying and accelerating level-set and density-based topology
       optimization by subpixel-smoothed projection," arXiv:2503.20189v3 [physics.optics]
       (2025).

    .. [2] G. Romano, R. Arrieta, and S. G. Johnson, "Differentiating through
       binarized topology changes: Second-order subpixel-smoothed projection,"
       arXiv:2601.10737 (2026). https://arxiv.org/abs/2601.10737
    """
    return projection_numerics.smoothed_projection(
        array, beta=beta, eta=eta, scaling_factor=scaling_factor, order=order, periodic=periodic
    )


def _validate_smoothed_projection_shape(
    shape: tuple[int, ...], order: int, *, periodic: bool = False
) -> None:
    """Validate the stencil size before projection or a combined filter operation."""
    projection_numerics.validate_smoothed_projection_shape(shape, order, periodic=periodic)
