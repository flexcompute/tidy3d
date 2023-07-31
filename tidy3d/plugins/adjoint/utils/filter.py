"""Spatial filtering Functions for adjoint plugin."""
from abc import ABC, abstractmethod

import pydantic as pd
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from ....components.base import Tidy3dBaseModel
from ....constants import MICROMETER


class Filter(Tidy3dBaseModel, ABC):
    """Abstract filter class. Initializes with parameters and .evaluate() on a design."""

    @abstractmethod
    def evaluate(self, spatial_data: jnp.array) -> jnp.array:
        """Process supplied array containing spatial data."""


class ConicFilter(Filter):
    """Filter that convolves an image with a conical mask, used for larger feature sizes.

    Note
    ----
    .. math::

        filter(radius) = max(feature_radius - radius, 0)

    """

    feature_size: float = pd.Field(
        ...,
        title="Filter Radius",
        description="Convolve spatial data with a conic filter. "
        "Useful for smoothing feature sizes.",
        units=MICROMETER,
    )

    design_region_dl: float = pd.Field(
        ...,
        title="Grid Size in Design Region",
        description="Grid size in the design region. "
        "This sets the length scale for the conic convolution filter.",
        units=MICROMETER,
    )

    @property
    def filter_radius(self) -> float:
        """Filter radius."""
        return np.ceil((self.feature_size * np.sqrt(3)) / self.design_region_dl)

    def evaluate(self, spatial_data: jnp.array) -> jnp.array:
        """Process on supplied spatial data."""

        rho = jnp.squeeze(spatial_data)
        dims = len(rho.shape)

        # Builds the conic filter and apply it to design parameters.
        coords_1d = np.linspace(
            -self.filter_radius, self.filter_radius, int(2 * self.filter_radius + 1)
        )
        meshgrid_args = [coords_1d.copy() for _ in range(dims)]

        meshgrid_coords = np.meshgrid(*meshgrid_args)
        coords_rad = np.sqrt(np.sum([np.square(v) for v in meshgrid_coords], axis=0))
        kernel = jnp.where(self.filter_radius - coords_rad > 0, self.filter_radius - coords_rad, 0)
        filt_den = jsp.signal.convolve(jnp.ones_like(rho), kernel, mode="same")
        return jsp.signal.convolve(rho, kernel, mode="same") / filt_den


class BinaryProjector(Filter):
    """Projects a grayscale image towards min and max values using a smooth `tanh` function.

    Note
    ----
    .. math::

        v(x) = vmin + (vmax - vmin) \\frac{tanh(\\beta \\eta) +
        tanh(\\beta * (x - \\eta))}{tanh(\\beta * \\eta) + tanh(\\beta * (1 - \\eta))}

    """

    vmin: float = pd.Field(..., title="Min Value", description="Minimum value to project to.")

    vmax: float = pd.Field(..., title="Min Value", description="Maximum value to project to.")

    beta: float = pd.Field(
        1.0,
        title="Beta",
        description="Steepness of the binarization, "
        "higher means more sharp transition "
        "at the expense of gradient accuracy and ease of optimization. "
        "Can be useful to ramp up in a scheduled way during optimization.",
    )

    eta: float = pd.Field(0.5, title="Eta", description="Halfway point in projection function.")

    strict_binarize: bool = pd.Field(
        False,
        title="Binarize strictly",
        description="If False, the binarization is still continuous between min and max. "
        "If false, the values are snapped to the min and max values after projection.",
    )

    def evaluate(self, spatial_data: jnp.array) -> jnp.array:
        """Process on supplied spatial data."""

        # Applies a hyperbolic tangent binarization function.
        num = jnp.tanh(self.beta * self.eta) + jnp.tanh(self.beta * (spatial_data - self.eta))
        den = jnp.tanh(self.beta * self.eta) + jnp.tanh(self.beta * (1.0 - self.eta))
        rho_bar = num / den

        # Calculates the values from the transformed design parameters.
        vals = self.vmin + (self.vmax - self.vmin) * rho_bar
        if self.strict_binarize:
            vals = jnp.where(vals < (self.vmin + self.vmax) / 2, self.vmin, self.vmax)
        else:
            vals = jnp.where(vals < self.vmin, self.vmin, vals)
            vals = jnp.where(vals > self.vmax, self.vmax, vals)
        return vals
