# define penalties applied to parameters from design region

import typing
import abc

import pydantic.v1 as pd
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint.utils.penalty as tda_penalty

from .base import InvdesBaseModel
from .validators import ignore_inherited_field


class AbstractPenalty(InvdesBaseModel, abc.ABC):
    """Base class for penalties added to ``invdes.DesignRegion`` objects."""

    weight: pd.NonNegativeFloat = pd.Field(
        1.0,
        title="Weight",
        description="When this penalty is evaluated, it will be weighted by this "
        "value. Note that the optimizer seeks to maximize the objective function and "
        "subtracts the penalty strength times this weight from the objective function. ",
    )


class ErosionDilationPenalty(AbstractPenalty, tda_penalty.ErosionDilationPenalty):
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

    pixel_size: pd.PositiveFloat = pd.Field(
        None,
        title="Pixel Size",
        description="Size of each pixel in the array (must be the same along all dimensions). "
        "Corresponds to ``design_region_dl`` in the :class:`ConicFilter` used for filtering. "
        "NOTE: this is set internally in the ``invdes`` plugin. So supplied values will be "
        "ignored.",
        units=td.constants.MICROMETER,
    )

    _ignore_pixel_size = ignore_inherited_field("pixel_size")

    def to_tda(self, pixel_size: float) -> tda_penalty.ErosionDilationPenalty:
        self_dict = self.dict(exclude={"type", "pixel_size", "weight"})
        return tda_penalty.ErosionDilationPenalty(pixel_size=pixel_size, **self_dict)

    def evaluate(self, x: jnp.ndarray, pixel_size: float) -> float:
        """Evaluate this penalty."""
        penalty_tda = self.to_tda(pixel_size=pixel_size)
        penalty_unweighted = penalty_tda.evaluate(x)
        return self.weight * penalty_unweighted


PenaltyType = typing.Union[ErosionDilationPenalty]
