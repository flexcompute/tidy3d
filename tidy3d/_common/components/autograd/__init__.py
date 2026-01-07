from __future__ import annotations

from tidy3d._common.components.autograd.boxes import TidyArrayBox
from tidy3d._common.components.autograd.functions import interpn
from tidy3d._common.components.autograd.types import (
    AutogradFieldMap,
    InterpolationType,
    PathType,
    TracedArrayFloat2D,
    TracedArrayLike,
    TracedComplex,
    TracedCoordinate,
    TracedFloat,
    TracedPoleAndResidue,
    TracedPolesAndResidues,
    TracedPositiveFloat,
    TracedSize,
    TracedSize1D,
)
from tidy3d._common.components.autograd.utils import get_static, hasbox, is_tidy_box, split_list

__all__ = [
    "AutogradFieldMap",
    "InterpolationType",
    "PathType",
    "TidyArrayBox",
    "TracedArrayFloat2D",
    "TracedArrayLike",
    "TracedComplex",
    "TracedCoordinate",
    "TracedFloat",
    "TracedPoleAndResidue",
    "TracedPolesAndResidues",
    "TracedPositiveFloat",
    "TracedSize",
    "TracedSize1D",
    "get_static",
    "hasbox",
    "interpn",
    "is_tidy_box",
    "split_list",
]
