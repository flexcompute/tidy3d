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


class AbstractCircularFilter(Filter, ABC):
    """Abstract circular filter class. Initializes with parameters and .evaluate() on a design."""

    radius: float = pd.Field(
        ...,
        title="Filter Radius",
        description="Radius of the filter to convolve with supplied spatial data. "
        "Note: the corresponding feature size expressed in the device is typically "
        "sqrt(3) times smaller than the radius. For best results, it is recommended to make your "
        "radius about twice as large as the desired feature size.",
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
    def filter_radius_pixels(self) -> int:
        """Filter radius in pixels."""
        return np.ceil(self.radius / self.design_region_dl)

    @pd.root_validator(pre=True)
    def _deprecate_feature_size(cls, values):
        """Extra warning for user using `feature_size` field."""
        if "feature_size" in values:
            raise pd.ValidationError(
                "The 'feature_size' field of circular filters available in 2.4 pre-releases was "
                "renamed to 'radius' for the official 2.4.0 release. "
                "If you're seeing this message, please change your script to use that field name."
            )
        return values

    @abstractmethod
    def make_kernel(self, coords_rad: jnp.array) -> jnp.array:
        """Function to make the kernel out of a coordinate grid of radius values."""

    def evaluate(self, spatial_data: jnp.array) -> jnp.array:
        """Process on supplied spatial data."""

        rho = jnp.squeeze(spatial_data)
        num_dims = len(rho.shape)

        # Builds the conic filter and apply it to design parameters.
        coords_1d = np.arange(-self.filter_radius_pixels, self.filter_radius_pixels + 1)
        meshgrid_args = [coords_1d.copy() for _ in range(num_dims)]
        meshgrid_coords = np.meshgrid(*meshgrid_args)
        coords_rad = np.sqrt(np.sum([np.square(v) for v in meshgrid_coords], axis=0))

        # construct the kernel
        kernel = self.make_kernel(coords_rad)

        # normalize by the kernel operating on a spatial_data of all ones
        num = jsp.signal.convolve(rho, kernel, mode="same")
        den = jsp.signal.convolve(jnp.ones_like(rho), kernel, mode="same")

        return num / den


class ConicFilter(AbstractCircularFilter):
    """Filter that convolves an image with a conical mask, used for larger feature sizes.

    Note
    ----
    .. math::

        filter(r) = max(radius - r, 0)

    """

    def make_kernel(self, coords_rad: jnp.array) -> jnp.array:
        """Function to make the kernel out of a coordinate grid of radius values (in pixels)."""

        kernel = self.filter_radius_pixels - coords_rad
        kernel[coords_rad > self.filter_radius_pixels] = 0.0
        return kernel


class CircularFilter(AbstractCircularFilter):
    """Filter that convolves an image with a circular mask, used for larger feature sizes.

    Note
    ----
    .. math::

        filter(r) = 1 if r <= radius else 0

    Note
    ----
    This uniform circular mask produces results that are harder to binarize than the conical mask.
    We recommend you use the ``ConicFilter`` instead for most applications.

    """

    def make_kernel(self, coords_rad: jnp.array) -> jnp.array:
        """Function to make the kernel out of a coordinate grid of radius values (in pixels)."""

        # construct the kernel
        kernel = np.ones_like(coords_rad)
        kernel[coords_rad > self.filter_radius_pixels] = 0.0
        return kernel


class BinaryProjector(Filter):
    """Projects a grayscale image towards min and max values using a smooth `tanh` function.

    Note
    ----
    .. math::

        v(x) = vmin + (vmax - vmin) \\frac{tanh(\\beta \\eta) +
        tanh(\\beta * (x - \\eta))}{tanh(\\beta * \\eta) + tanh(\\beta * (1 - \\eta))}

    """

    vmin: float = pd.Field(..., title="Min Value", description="Minimum value to project to.")

    vmax: float = pd.Field(..., title="Max Value", description="Maximum value to project to.")

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
