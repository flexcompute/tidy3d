"""Penalty Functions for adjoint plugin."""
from abc import ABC, abstractmethod

import pydantic.v1 as pd
import jax.numpy as jnp

from ....components.base import Tidy3dBaseModel
from ....components.types import ArrayFloat2D
from ....constants import MICROMETER

# Radius of Curvature Calculation


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

        p(r) = \\frac{exp(-\\mathrm{kappa}(r - r_{min}))}{1 + exp(-\\mathrm{kappa}(r - r_{min}))}

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
        units=MICROMETER + "^-1",
    )

    wrap: bool = pd.Field(
        False,
        title="Wrap",
        description="Whether to consider the first set of points as connected to the last.",
        units="",
    )

    def evaluate(self, points: ArrayFloat2D) -> float:
        """Get the average penalty as a function of supplied (x, y) points by
        fitting a spline to the curve and evaluating local radius of curvature compared to a
        desired minimum value. If ``wrap``, it is assumed that the points wrap around to form a
        closed geometry instead of an isolated line segment."""

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
