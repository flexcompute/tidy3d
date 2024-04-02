# transformations applied to design region

import typing
import abc

import jax.numpy as jnp
import pydantic.v1 as pd

import tidy3d as td
import tidy3d.plugins.adjoint.utils.filter as tda_filter

from .base import InvdesBaseModel


class AbstractTransformation(InvdesBaseModel, abc.ABC):
    """Base class for transformations."""


class FilterProject(InvdesBaseModel):
    """Transformation involving convolution by a conic filter followed by a ``tanh`` projection.

    .. image:: ../../_static/img/filter_project.png

    """

    radius: float = pd.Field(
        ...,
        title="Filter Radius",
        description="Radius of the filter to convolve with supplied spatial data. "
        "Note: the corresponding feature size expressed in the device is typically "
        "sqrt(3) times smaller than the radius. For best results, it is recommended to make your "
        "radius about twice as large as the desired feature size.",
        units=td.constants.MICROMETER,
    )

    beta: float = pd.Field(
        1.0,
        title="Beta",
        description="Steepness of the binarization, "
        "higher means more sharp transition "
        "at the expense of gradient accuracy and ease of optimization. ",
    )

    eta: float = pd.Field(0.5, title="Eta", description="Halfway point in projection function.")

    strict_binarize: bool = pd.Field(
        False,
        title="Binarize strictly",
        description="If ``False``, the binarization is still continuous between min and max. "
        "If ``True``, the values are snapped to the min and max values after projection.",
    )

    def to_filter(self, design_region_dl: float) -> tda_filter.ConicFilter:
        """Get the conic filter associated with this transformation (after supplying grid size)."""
        return tda_filter.ConicFilter(radius=self.radius, design_region_dl=design_region_dl)

    def to_projector(self) -> tda_filter.BinaryProjector:
        """Get the binary projector associated with this transformation."""
        return tda_filter.BinaryProjector(
            beta=self.beta, eta=self.eta, strict_binarize=self.strict_binarize, vmin=0, vmax=1
        )

    def evaluate(self, spatial_data: jnp.ndarray, design_region_dl: float) -> jnp.ndarray:
        """Evaluate this transformation on spatial data, given some grid size in the region."""
        conic_filter = self.to_filter(design_region_dl=design_region_dl)
        binary_projector = self.to_projector()

        data_filtered = conic_filter.evaluate(spatial_data)
        data_projected = binary_projector.evaluate(data_filtered)
        return data_projected


TransformationType = typing.Union[FilterProject]
