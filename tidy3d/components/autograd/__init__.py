"""Compatibility shim for :mod:`tidy3d._common.components.autograd`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd import (
    AutogradFieldMap,
    InterpolationType,
    PathType,
    TidyArrayBox,
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
    get_static,
    hasbox,
    interpn,
    is_tidy_box,
    split_list,
)
