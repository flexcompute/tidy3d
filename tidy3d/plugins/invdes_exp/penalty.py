# define penalties applied to parameters from design region

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

from tidy3d.constants import MICROMETER
from tidy3d.exceptions import ValidationError
from tidy3d.plugins.autograd.invdes import make_erosion_dilation_penalty

from .base import InvdesBaseModel


class AbstractPenalty(InvdesBaseModel, abc.ABC):
    """Base class for penalties added to ``invdes.DesignRegion`` objects."""

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""

    def __call__(self, *args, **kwargs) -> float:
        return self.evaluate(*args, **kwargs)


class SingleRegionPenalty(AbstractPenalty):
    region_name: str = pd.Field(
        ..., title="region name", description="region name for region to apply penalty to."
    )

    transformation_idx: int = pd.Field(
        ...,
        title="transformation index",
        description="index into transformation chain where penalty should be applied",
    )


class ErosionDilationPenalty(SingleRegionPenalty):
    """Erosion and dilation penalty, addable to ``tdi.TopologyDesignRegion.penalties``.
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

    length_scale: pd.PositiveFloat = pd.Field(
        ...,
        title="Length Scale",
        description="Length scale of erosion and dilation. "
        "Corresponds to ``radius`` in the :class:`ConicFilter` used for filtering. "
        "The parameter array is dilated and eroded by half of this value with each operation. "
        "Roughly corresponds to the desired minimum feature size and radius of curvature.",
        units=MICROMETER,
    )

    pixel_size: float = pd.Field(..., gt=0.0, title="Pixel size")

    beta: float = pd.Field(
        100.0,
        ge=1.0,
        title="Projection Beta",
        description="Strength of the ``tanh`` projection. "
        "Corresponds to ``beta`` in the :class:`BinaryProjector. "
        "Higher values correspond to stronger discretization.",
    )

    eta0: float = pd.Field(
        0.5,
        ge=0.0,
        le=1.0,
        title="Projection Midpoint",
        description="Value between 0 and 1 that sets the projection midpoint. In other words, "
        "for values of ``eta0``, the projected values are halfway between minimum and maximum. "
        "Corresponds to ``eta`` in the :class:`BinaryProjector`.",
    )

    delta_eta: float = pd.Field(
        0.01,
        ge=0.0,
        le=1.0,
        title="Delta Eta Cutoff",
        description="The binarization threshold for erosion and dilation operations "
        "The thresholds are ``0 + delta_eta`` on the low end and ``1 - delta_eta`` on the high end. "
        "The default value balances binarization with differentiability so we strongly suggest "
        "using it unless there is a good reason to set it differently.",
    )

    def evaluate(self, region_data) -> float:
        """Evaluate this penalty."""
        penalty_fn = make_erosion_dilation_penalty(
            self.length_scale,
            self.pixel_size,
            beta=self.beta,
            eta=self.eta0,
            delta_eta=self.delta_eta,
        )
        x = region_data[self.region_name][self.transformation_idx]
        return penalty_fn(x)


class BinarizationPenalty(SingleRegionPenalty):
    bounds: typing.Tuple[float, float] = pd.Field(
        (0, 1), title="bounds", description="clipping bounds for binarizatino"
    )

    @pd.validator("bounds")
    def validate_bounds(bounds, values):
        if bounds[0] >= bounds[1]:
            raise ValidationError("Bounds should be in ascending order.")

        return bounds

    def evaluate(self, region_data) -> float:
        """Evaluate this penalty."""
        x = anp.clip(
            region_data[self.region_name][self.transformation_idx],
            a_min=self.bounds[0],
            a_max=self.bounds[1],
        )
        recenter_x = (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return 1.0 - anp.mean(anp.abs(recenter_x - 0.5) / 0.5)


class CustomPenalty(AbstractPenalty):
    eval_fn: typing.Callable = pd.Field(
        ..., title="eval_fn", description="custom evaluation function for penalty"
    )

    def evaluate(self, region_data) -> float:
        """Evaluate this penalty."""
        return self.eval_fn(region_data)


PenaltyType = typing.Union[BinarizationPenalty, CustomPenalty, ErosionDilationPenalty]
