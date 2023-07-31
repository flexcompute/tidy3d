# pylint:disable=invalid-name
"""Penalty Functions for adjoint plugin."""
from abc import ABC, abstractmethod

import jax.numpy as jnp

from ....components.base import Tidy3dBaseModel
from ....components.types import Vertices

# Radius of Curvature Calculation


class Penalty(Tidy3dBaseModel, ABC):
    """Abstract penalty class. Initializes with parameters and .evaluate() on a design."""

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""


class RadiusPenalty(Penalty):
    """Generates a penalty for radius of curvature of set of points."""

    min_radius: float = 0.150
    alpha: float = 1.0
    kappa: float = 10.0
    wrap: bool = False

    # pylint: disable=arguments-differ
    def evaluate(self, points: Vertices) -> float:
        """Get the penalty as a function of supplied (x, y) points."""

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

            # pylint:disable=unused-argument
            def d2_p(t):
                """Second derivative function."""
                d2_term0 = 2 * p0
                d2_term1 = -4 * p1
                d2_term2 = 2 * p2
                return d2_term0 + d2_term1 + d2_term2

            return p, d_p, d2_p

        def get_fit_vals(xs, ys):
            """Get the values of the bezier curve and its derivatives at t=0.5 along the points."""

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
            den = abs(xp * yp2 - yp * xp2) + 1e-2
            return num / den

        def penalty_fn(radius):
            """Get the penalty for a given radius."""
            arg = -self.kappa * (self.min_radius - radius)
            return self.alpha * ((1 + jnp.exp(arg)) ** (-1))

        xs, ys = jnp.array(points).T
        rs = get_radii_curvature(xs, ys)

        # return the average penalty over the points
        return jnp.sum(penalty_fn(rs)) / len(rs)
