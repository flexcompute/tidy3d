# transformations applied to design region
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from pydantic import Field, PositiveFloat

import tidy3d as td
from tidy3d.plugins.autograd.functions import threshold
from tidy3d.plugins.autograd.invdes import make_filter_and_project
from tidy3d.plugins.autograd.types import ProjectionType

from .base import InvdesBaseModel

if TYPE_CHECKING:
    import autograd.numpy as anp

    from tidy3d.plugins.autograd.invdes.symmetries import MirrorSymmetry


class AbstractTransformation(InvdesBaseModel, abc.ABC):
    """Base class for transformations."""

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return self.evaluate(*args, **kwargs)


class FilterProject(InvdesBaseModel):
    """Conic filtering followed by tanh, SSP1, or SSP2 projection.

    Choose ``projection='ssp2'`` for Hessian-regularized subpixel smoothing during
    topology changes. SSP projections require equal grid spacing and a 2D design
    (singleton axes are preserved). The filter radius should be substantially
    larger than the pixel size for the local boundary approximation to apply.

    Notes
    -----

        .. image:: ../../_static/img/filter_project.png

    """

    radius: PositiveFloat = Field(
        title="Filter Radius",
        description="Radius of the filter to convolve with supplied spatial data. "
        "Note: the corresponding feature size expressed in the device is typically "
        "sqrt(3) times smaller than the radius. For best results, it is recommended to make your "
        "radius about twice as large as the desired feature size. "
        "Note: the radius value is often only approximately related to the final feature sizes. "
        "It is useful to apply a ``FilterProject`` transformation to 'encourage' larger "
        "feature sizes, but we ultimately recommend creating a ``ErosionDilationPenalty`` to the "
        "``DesignRegion.penalties`` if you have strict fabrication constraints.",
        json_schema_extra={"units": td.constants.MICROMETER},
    )

    beta: float = Field(
        default=1.0,
        ge=1.0,
        title="Beta",
        description="Steepness of the binarization, "
        "higher means more sharp transition "
        "at the expense of gradient accuracy and ease of optimization. ",
    )

    eta: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Eta",
        description="Halfway point in projection function.",
    )

    projection: ProjectionType = Field(
        default="tanh",
        title="Projection",
        description="Projection applied after conic filtering: 'tanh', first-order 'ssp1', "
        "or Hessian-regularized 'ssp2'. SSP requires two non-singleton axes with at least "
        "two samples for SSP1 or three samples for SSP2.",
    )

    strict_binarize: bool = Field(
        default=False,
        title="Binarize strictly",
        description="If ``False``, the binarization is still continuous between min and max. "
        "If ``True``, the values are snapped to the min and max values after projection.",
    )

    def evaluate(
        self,
        spatial_data: anp.ndarray,
        design_region_dl: float,
        symmetry: MirrorSymmetry | None = None,
    ) -> anp.ndarray:
        """Evaluate this transformation on spatial data, given the region grid size."""
        filt_proj = make_filter_and_project(
            self.radius,
            design_region_dl,
            beta=self.beta,
            eta=self.eta,
            symmetry=symmetry,
            projection=self.projection,
        )
        data_projected = filt_proj(spatial_data)

        if self.strict_binarize:
            data_projected = threshold(data_projected)

        return data_projected


TransformationType = FilterProject
