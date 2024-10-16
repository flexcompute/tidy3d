# transformations applied to design region

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.plugins.autograd.functions import threshold
from tidy3d.plugins.autograd.invdes import make_filter_and_project

from .base import InvdesBaseModel


class AbstractTransformation(InvdesBaseModel, abc.ABC):
    """Base class for transformations."""

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""

    def __call__(self, *args, **kwargs) -> float:
        return self.evaluate(*args, **kwargs)


class FilterProject(InvdesBaseModel):
    """Transformation involving convolution by a conic filter followed by a ``tanh`` projection.

    Notes
    -----

        .. image:: ../../_static/img/filter_project.png

    """

    radius: pd.PositiveFloat = pd.Field(
        ...,
        title="Filter Radius",
        description="Radius of the filter to convolve with supplied spatial data. "
        "Note: the corresponding feature size expressed in the device is typically "
        "sqrt(3) times smaller than the radius. For best results, it is recommended to make your "
        "radius about twice as large as the desired feature size. "
        "Note: the radius value is often only approximately related to the final feature sizes. "
        "It is useful to apply a ``FilterProject`` transformation to 'encourage' larger "
        "feature sizes, but we ultimately recommend creating a ``ErosionDilationPenalty`` to the "
        "``DesignRegion.penalties`` if you have strict fabrication constraints.",
        units=td.constants.MICROMETER,
    )

    beta: float = pd.Field(
        1.0,
        ge=1.0,
        title="Beta",
        description="Steepness of the binarization, "
        "higher means more sharp transition "
        "at the expense of gradient accuracy and ease of optimization. ",
    )

    eta: float = pd.Field(
        0.5, ge=0.0, le=1.0, title="Eta", description="Halfway point in projection function."
    )

    strict_binarize: bool = pd.Field(
        False,
        title="Binarize strictly",
        description="If ``False``, the binarization is still continuous between min and max. "
        "If ``True``, the values are snapped to the min and max values after projection.",
    )

    def evaluate(self, spatial_data: anp.ndarray, design_region_dl: float) -> anp.ndarray:
        """Evaluate this transformation on spatial data, given some grid size in the region."""
        filt_proj = make_filter_and_project(
            self.radius, design_region_dl, beta=self.beta, eta=self.eta
        )
        data_projected = filt_proj(spatial_data)

        if self.strict_binarize:
            data_projected = threshold(data_projected)

        return data_projected


TransformationType = typing.Union[FilterProject]
