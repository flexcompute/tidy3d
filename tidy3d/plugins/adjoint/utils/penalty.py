"""Penalty Functions for adjoint plugin."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import pydantic.v1 as pd

from ....components.base import Tidy3dBaseModel
from ....components.types import ArrayFloat2D
from ....constants import MICROMETER
from ....log import log
from .filter import BinaryProjector, ConicFilter

# Radius of Curvature Calculation


def is_jax_object(arr) -> bool:
    """Test whether an object is a `jnp.ndarray` or an iterable containing them."""
    if isinstance(arr, jnp.ndarray):
        return True
    if isinstance(arr, (list, tuple)):
        return is_jax_object(arr[0])
    return False


class Penalty(Tidy3dBaseModel, ABC):
    """Abstract penalty class. Initializes with parameters and .evaluate() on a design."""

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""


class RadiusPenalty(Penalty):
    """Computes a penalty for small radius of curvature determined by a fit of points in a 2D plane.

    Note
    ----
    .. math::

        p(r) = \\frac{\\mathrm{exp}(-\\kappa(r - r_{min}))}{1 + \\mathrm{exp}(-\\kappa(r - r_{min}))}

    Note
    ----
    This formula was described by A. Micheals et al.
    "Leveraging continuous material averaging for inverse electromagnetic design",
    Optics Express (2018).

    """

    min_radius: float = pd.Field(
        0.150,
        title="Minimum Radius",
        description="Radius of curvature value below which the penalty ramps to its maximum value.",
        units=MICROMETER,
    )

    alpha: float = pd.Field(
        1.0,
        title="Alpha",
        description="Parameter controlling the strength of the penalty.",
    )

    kappa: float = pd.Field(
        10.0,
        title="Kappa",
        description="Parameter controlling the steepness of the penalty evaluation.",
        units="1/" + MICROMETER,
    )

    wrap: bool = pd.Field(
        False,
        title="Wrap",
        description="Whether to consider the first set of points as connected to the last.",
    )

    def evaluate(self, points: ArrayFloat2D) -> float:
        """Get the average penalty as a function of supplied (x, y) points by
        fitting a spline to the curve and evaluating local radius of curvature compared to a
        desired minimum value. If ``wrap``, it is assumed that the points wrap around to form a
        closed geometry instead of an isolated line segment."""

        if not is_jax_object(points):
            log.warning(
                "The points passed to 'RadiusPenalty.evaluate()' are not a 'jax' array. "
                "If passing the 'JaxPolySlab.vertices' field directly, note that the "
                "derivative information for this field "
                "is no longer traced by jax as of "
                "version '2.7'. "
                "The derivative information is contained in 'JaxPolySlab.vertices_jax'. "
                "Therefore, we recommend changing your code to either pass that field or pass "
                "the output of the parameterization functions directly, eg. "
                "'penalty.evaluate(make_vertices(params))'."
            )

        def quad_fit(p0, pc, p2):
            """Quadratic bezier fit (and derivatives) for three points.
            (x(t), y(t)) = P(t) = P0*t^2 + P1*2*t*(1-t) + P2*(1-t)^2
             t in [0, 1]
            """

            # ensure curve goes through (x1, y1) at t=0.5
            p1 = 2 * pc - p0 / 2 - p2 / 2

            def p(t):
                """Bezier curve parameterization."""
                term0 = (1 - t) ** 2 * (p0 - p1)
                term1 = p1
                term2 = t**2 * (p2 - p1)
                return term0 + term1 + term2

            def d_p(t):
                """First derivative function."""
                d_term0 = 2 * (1 - t) * (p1 - p0)
                d_term2 = 2 * t * (p2 - p1)
                return d_term0 + d_term2

            def d2_p(t):
                """Second derivative function."""
                d2_term0 = 2 * p0
                d2_term1 = -4 * p1
                d2_term2 = 2 * p2
                return d2_term0 + d2_term1 + d2_term2

            return p, d_p, d2_p

        def get_fit_vals(xs, ys):
            """Get the values of the Bezier curve and its derivatives at t=0.5 along the points."""

            ps = jnp.stack((xs, ys), axis=1)
            p0 = ps[:-2]
            pc = ps[1:-1]
            p2 = ps[2:]

            p, d_p, d_2p = quad_fit(p0, pc, p2)

            ps = p(0.5)
            dps = d_p(0.5)
            d2ps = d_2p(0.5)
            return ps.T, dps.T, d2ps.T

        def get_radii_curvature(xs, ys):
            """Get the radii of curvature at each (internal) point along the set of points."""
            _, dps, d2ps = get_fit_vals(xs, ys)
            xp, yp = dps
            xp2, yp2 = d2ps
            num = (xp**2 + yp**2) ** (3.0 / 2.0)
            den = abs(xp * yp2 - yp * xp2)
            return num / den

        def penalty_fn(radius):
            """Get the penalty for a given radius."""
            arg = self.kappa * (radius - self.min_radius)
            exp_arg = jnp.exp(-arg)
            return self.alpha * (exp_arg / (1 + exp_arg))

        xs, ys = jnp.array(points).T
        rs = get_radii_curvature(xs, ys)

        # return the average penalty over the points
        return jnp.sum(penalty_fn(rs)) / len(rs)


class ErosionDilationPenalty(Penalty):
    """Computes a penalty for erosion / dilation of a parameter map not being unity.
    Accepts a parameter array normalized between 0 and 1.
    Uses filtering and projection methods to erode and dilate the features within this array.
    Measures the change in the array after eroding and dilating (and also dilating and eroding).
    Returns a penalty proportional to the magnitude of this change.
    The amount of change under dilation and erosion is minimized if the structure has large feature
    sizes and large radius of curvature relative to the length scale.

    Note
    ----
    For more details, refer to chapter 4 of Hammond, A., "High-Efficiency Topology Optimization
    for Very Large-Scale Integrated-Photonics Inverse Design" (2022).


    .. image:: ../../_static/img/erosion_dilation.png

    """

    length_scale: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Length Scale",
        description="Length scale of erosion and dilation. "
        "Corresponds to ``radius`` in the :class:`ConicFilter` used for filtering. "
        "The parameter array is dilated and eroded by half of this value with each operation. "
        "Roughly corresponds to the desired minimum feature size and radius of curvature.",
        units=MICROMETER,
    )

    pixel_size: pd.PositiveFloat = pd.Field(
        ...,
        title="Pixel Size",
        description="Size of each pixel in the array (must be the same along all dimensions). "
        "Corresponds to ``design_region_dl`` in the :class:`ConicFilter` used for filtering.",
        units=MICROMETER,
    )

    beta: pd.PositiveFloat = pd.Field(
        100.0,
        title="Projection Beta",
        description="Strength of the ``tanh`` projection. "
        "Corresponds to ``beta`` in the :class:`BinaryProjector. "
        "Higher values correspond to stronger discretization.",
    )

    eta0: pd.PositiveFloat = pd.Field(
        0.5,
        title="Projection Midpoint",
        description="Value between 0 and 1 that sets the projection midpoint. In other words, "
        "for values of ``eta0``, the projected values are halfway between minimum and maximum. "
        "Corresponds to ``eta`` in the :class:`BinaryProjector`.",
    )

    delta_eta: pd.PositiveFloat = pd.Field(
        0.01,
        title="Delta Eta Cutoff",
        description="The binarization threshold for erosion and dilation operations "
        "The thresholds are ``0 + delta_eta`` on the low end and ``1 - delta_eta`` on the high end. "
        "The default value balances binarization with differentiability so we strongly suggest "
        "using it unless there is a good reason to set it differently.",
    )

    def conic_filter(self) -> ConicFilter:
        """:class:`ConicFilter` associated with this object."""
        return ConicFilter(radius=self.length_scale, design_region_dl=self.pixel_size)

    def binary_projector(self, eta: float = None) -> BinaryProjector:
        """:class:`BinaryProjector` associated with this object."""

        if eta is None:
            eta = self.eta0

        return BinaryProjector(eta=eta, beta=self.beta, vmin=0.0, vmax=1.0, strict_binarize=False)

    def tanh_projection(self, x: jnp.ndarray, eta: float = None) -> jnp.ndarray:
        """Project an array ``x`` once using ``self.beta`` and ``self.eta0``."""
        return self.binary_projector(eta=eta).evaluate(x)

    def filter_project(self, x: jnp.ndarray, eta: float = None) -> jnp.ndarray:
        """Filter an array ``x`` using length scale and dL and then apply a projection."""
        filter = self.conic_filter()
        projector = self.binary_projector(eta=eta)

        y = filter.evaluate(x)
        return projector.evaluate(y)

    def evaluate(self, x: jnp.ndarray) -> float:
        """
        Penalty associated with erosion/dilation and dilation/erosion not being identity.
        Accepts a parameter array with values normalized between 0 and 1.
        Penalty value is normalized such that the maximum possible penalty is 1.
        """
        eta_dilate = 0.0 + self.delta_eta
        eta_eroded = 1.0 - self.delta_eta

        def fn_dilate(x):
            return self.filter_project(x, eta=eta_dilate)

        def fn_eroded(x):
            return self.filter_project(x, eta=eta_eroded)

        params_dilate_erode = fn_eroded(fn_dilate(x))
        params_erode_dilate = fn_dilate(fn_eroded(x))

        diff = params_dilate_erode - params_erode_dilate

        # edge case: if all diff == 0, then the gradient of sqrt() and norm() is not defined.
        if jnp.all(diff == 0.0):
            return 0.0

        return jnp.linalg.norm(diff) / jnp.linalg.norm(jnp.ones_like(diff))
