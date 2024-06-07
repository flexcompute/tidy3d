from typing import Tuple, Union

import autograd.numpy as np

from tidy3d.components.types import ArrayFloat2D

from ..types import PaddingType
from .parametrizations import make_filter_and_project


def make_erosion_dilation_penalty(
    filter_size: Tuple[int, ...],
    beta: float = 100.0,
    eta: float = 0.5,
    delta_eta: float = 0.01,
    padding: PaddingType = "reflect",
):
    """Computes a penalty for erosion/dilation of a parameter map not being unity.

    Accepts a parameter array normalized between 0 and 1. Uses filtering and projection methods
    to erode and dilate the features within this array. Measures the change in the array after
    eroding and dilating (and also dilating and eroding). Returns a penalty proportional to the
    magnitude of this change. The amount of change under dilation and erosion is minimized if
    the structure has large feature sizes and large radius of curvature relative to the length scale.

    Parameters
    ----------
    filter_size : Tuple[int, ...]
        The size of the filter to be used for erosion and dilation.
    beta : float, optional
        Strength of the tanh projection. Default is 100.0.
    eta : float, optional
        Midpoint of the tanh projection. Default is 0.5.
    delta_eta : float, optional
        The binarization threshold for erosion and dilation operations. Default is 0.01.
    padding : PaddingType, optional
        The padding type to use for the filter. Default is "reflect".

    Returns
    -------
    Callable
        A function that computes the erosion/dilation penalty for a given array.
    """
    filtproj = make_filter_and_project(filter_size, beta, eta, padding=padding)
    eta_dilate = 0.0 + delta_eta
    eta_eroded = 1.0 - delta_eta

    def _dilate(array: np.ndarray, beta: float):
        return filtproj(array, beta=beta, eta=eta_dilate)

    def _erode(array: np.ndarray, beta: float):
        return filtproj(array, beta=beta, eta=eta_eroded)

    def _open(array: np.ndarray, beta: float):
        return _dilate(_erode(array, beta=beta), beta=beta)

    def _close(array: np.ndarray, beta: float):
        return _erode(_dilate(array, beta=beta), beta=beta)

    def _erosion_dilation_penalty(array: np.ndarray, beta: float = beta):
        diff = _close(array, beta) - _open(array, beta)

        if not np.any(diff):
            return 0.0

        return np.linalg.norm(diff) / np.sqrt(diff.size)

    return _erosion_dilation_penalty


def curvature(dp: np.ndarray, ddp: np.ndarray) -> np.ndarray:
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
    t: float, p0: np.ndarray, pc: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def bezier_curvature(x: np.ndarray, y: np.ndarray, t: Union[np.ndarray, float] = 0.5) -> np.ndarray:
    """
    Calculate the curvature of a Bezier curve at a given parameter t.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the control points.
    y : np.ndarray
        The y-coordinates of the control points.
    t : Union[np.ndarray, float], optional
        The parameter at which to evaluate the curvature, by default 0.5.

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
):
    """Create a penalty function based on the curvature of a set of points.

    Parameters
    ----------
    min_radius : float
        The minimum radius of curvature.
    alpha : float, optional
        Scaling factor for the penalty, by default 1.0.
    kappa : float, optional
        Exponential factor for the penalty, by default 10.0.
    eps : float, optional
        A small value to avoid division by zero, by default 1e-6.

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

    def _curvature_penalty(points: ArrayFloat2D):
        xs, ys = np.array(points).T
        crv = bezier_curvature(xs, ys)
        curvature_radius = 1 / (np.abs(crv) + eps)
        arg = kappa * (curvature_radius - min_radius)
        exp_arg = np.exp(-arg)
        penalty = alpha * (exp_arg / (1 + exp_arg))
        return np.mean(penalty)

    return _curvature_penalty
