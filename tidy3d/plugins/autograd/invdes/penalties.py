from typing import Callable, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd
from numpy.typing import NDArray

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat2D

from ..types import PaddingType
from .parametrizations import FilterAndProject


class ErosionDilationPenalty(Tidy3dBaseModel):
    """A class that computes a penalty for erosion/dilation of a parameter map not being unity."""

    radius: Union[float, Tuple[float, ...]] = pd.Field(
        ..., title="Radius", description="The radius of the kernel."
    )
    dl: Union[float, Tuple[float, ...]] = pd.Field(
        ..., title="Grid Spacing", description="The grid spacing."
    )
    size_px: Union[int, Tuple[int, ...]] = pd.Field(
        None, title="Size in Pixels", description="The size of the kernel in pixels."
    )
    beta: pd.NonNegativeFloat = pd.Field(
        20.0, title="Beta", description="The beta parameter for the tanh projection."
    )
    eta: pd.NonNegativeFloat = pd.Field(
        0.5, title="Eta", description="The eta parameter for the tanh projection."
    )
    filter_type: str = pd.Field(
        "conic", title="Filter Type", description="The type of filter to create."
    )
    padding: PaddingType = pd.Field(
        "reflect", title="Padding", description="The padding mode to use."
    )
    delta_eta: float = pd.Field(
        0.01,
        title="Delta Eta",
        description="The binarization threshold for erosion and dilation operations.",
    )

    def __call__(self, array: NDArray) -> float:
        """Compute the erosion/dilation penalty for a given array.

        Parameters
        ----------
        array : np.ndarray
            The input array to compute the penalty for.

        Returns
        -------
        float
            The computed erosion/dilation penalty.
        """
        filtproj = FilterAndProject(
            radius=self.radius,
            dl=self.dl,
            size_px=self.size_px,
            beta=self.beta,
            eta=self.eta,
            filter_type=self.filter_type,
            padding=self.padding,
        )

        eta_dilate = 0.0 + self.delta_eta
        eta_eroded = 1.0 - self.delta_eta

        def _dilate(arr: NDArray):
            return filtproj(arr, eta=eta_dilate)

        def _erode(arr: NDArray):
            return filtproj(arr, eta=eta_eroded)

        def _open(arr: NDArray):
            return _dilate(_erode(arr))

        def _close(arr: NDArray):
            return _erode(_dilate(arr))

        diff = _close(array) - _open(array)

        if not np.any(diff):
            return 0.0

        return np.linalg.norm(diff) / np.sqrt(diff.size)


def make_erosion_dilation_penalty(
    radius: Union[float, Tuple[float, ...]],
    dl: Union[float, Tuple[float, ...]],
    *,
    size_px: Union[int, Tuple[int, ...]] = None,
    beta: float = 20.0,
    eta: float = 0.5,
    delta_eta: float = 0.01,
    padding: PaddingType = "reflect",
) -> Callable:
    """Computes a penalty for erosion/dilation of a parameter map not being unity.

    See Also
    --------
    :func:`~penalties.ErosionDilationPenalty`.
    """
    return ErosionDilationPenalty(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=beta,
        eta=eta,
        delta_eta=delta_eta,
        padding=padding,
    )


def curvature(dp: NDArray, ddp: NDArray) -> NDArray:
    """Calculate the curvature at a point given the first and second derivatives.

    Parameters
    ----------
    dp : np.ndarray
        The first derivative at the point, with (x, y) entries in the first dimension.
    ddp : np.ndarray
        The second derivative at the point, with (x, y) entries in the first dimension.

    Returns
    -------
    np.ndarray
        The curvature at the given point.

    Notes
    -----
    The curvature can be positive or negative, indicating the direction of curvature.
    The radius of curvature is defined as 1 / |κ|, where κ is the curvature.
    """
    num = dp[0] * ddp[1] - dp[1] * ddp[0]
    den = np.power(dp[0] ** 2 + dp[1] ** 2, 1.5)
    return num / den


def bezier_with_grads(
    t: float, p0: NDArray, pc: NDArray, p2: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the Bezier curve and its first and second derivatives at a given point.

    Parameters
    ----------
    t : float
        The parameter at which to evaluate the Bezier curve.
    p0 : np.ndarray
        The first control point of the Bezier curve.
    pc : np.ndarray
        The central control point of the Bezier curve.
    p2 : np.ndarray
        The last control point of the Bezier curve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the Bezier curve value, its first derivative, and its second derivative at the given point.
    """
    p1 = 2 * pc - p0 / 2 - p2 / 2
    b = (1 - t) ** 2 * (p0 - p1) + p1 + t**2 * (p2 - p1)
    dbdt = 2 * ((1 - t) * (p1 - p0) + t * (p2 - p1))
    dbd2t = 2 * (p0 - 2 * p1 + p2)
    return b, dbdt, dbd2t


def bezier_curvature(x: NDArray, y: NDArray, t: Union[NDArray, float] = 0.5) -> NDArray:
    """
    Calculate the curvature of a Bezier curve at a given parameter t.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the control points.
    y : np.ndarray
        The y-coordinates of the control points.
    t : Union[np.ndarray, float] = 0.5
        The parameter at which to evaluate the curvature.

    Returns
    -------
    np.ndarray
        The curvature of the Bezier curve at the given parameter t.
    """
    p = np.stack((x, y), axis=1)
    _, dbdt, dbd2t = bezier_with_grads(t, p[:-2], p[1:-1], p[2:])
    return curvature(dbdt.T, dbd2t.T)


def make_curvature_penalty(
    min_radius: float, alpha: float = 1.0, kappa: float = 10.0, *, eps: float = 1e-6
) -> Callable:
    """Create a penalty function based on the curvature of a set of points.

    Parameters
    ----------
    min_radius : float
        The minimum radius of curvature.
    alpha : float = 1.0
        Scaling factor for the penalty.
    kappa : float = 10.0
        Exponential factor for the penalty.
    eps : float = 1e-6
        A small value to avoid division by zero.

    Returns
    -------
    Callable
        A function that computes the curvature penalty for a given set of points.

    Notes
    -----
    The penalty function is defined as:

    .. math::

        p(r) = \\frac{\\mathrm{exp}(-\\kappa(r - r_{min}))}{1 + \\mathrm{exp}(-\\kappa(r - r_{min}))}

    This formula was described by A. Micheals et al.
    "Leveraging continuous material averaging for inverse electromagnetic design",
    Optics Express (2018).
    """

    def _curvature_penalty(points: ArrayFloat2D) -> float:
        xs, ys = np.array(points).T
        crv = bezier_curvature(xs, ys)
        curvature_radius = 1 / (np.abs(crv) + eps)
        arg = kappa * (curvature_radius - min_radius)
        exp_arg = np.exp(-arg)
        penalty = alpha * (exp_arg / (1 + exp_arg))
        return np.mean(penalty)

    return _curvature_penalty
