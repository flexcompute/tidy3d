# define penalties applied to parameters from design region

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

from tidy3d.constants import MICROMETER
from tidy3d.plugins.autograd.invdes import make_erosion_dilation_penalty

from .base import InvdesBaseModel


class AbstractPenalty(InvdesBaseModel, abc.ABC):
    """Base class for penalties added to ``invdes.DesignRegion`` objects."""

    weight: pd.NonNegativeFloat = pd.Field(
        1.0,
        title="Weight",
        description="When this penalty is evaluated, it will be weighted by this "
        "value. Note that the optimizer seeks to maximize the objective function and "
        "subtracts the penalty strength times this weight from the objective function. ",
    )

    @abc.abstractmethod
    def evaluate(self) -> float:
        """Evaluate the penalty on supplied values."""

    def __call__(self, *args, **kwargs) -> float:
        return self.evaluate(*args, **kwargs)


class ErosionDilationPenalty(AbstractPenalty):
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

    def evaluate(self, x: anp.ndarray, pixel_size: float) -> float:
        """Evaluate this penalty."""
        penalty_fn = make_erosion_dilation_penalty(
            self.length_scale, pixel_size, beta=self.beta, eta=self.eta0, delta_eta=self.delta_eta
        )
        penalty_unweighted = penalty_fn(x)
        return self.weight * penalty_unweighted


PenaltyType = typing.Union[ErosionDilationPenalty]
