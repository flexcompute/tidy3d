"""Compatibility shim for :mod:`tidy3d._common.components.autograd.types`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.autograd.types import (
    AutogradFieldMap,
    InterpolationType,
    PathType,
    TracedArrayFloat2D,
    TracedArrayLike,
    TracedComplex,
    TracedCoordinate,
    TracedDict,
    TracedFloat,
    TracedPoleAndResidue,
    TracedPolesAndResidues,
    TracedPositiveFloat,
    TracedSize,
    TracedSize1D,
    _copy,
    _deepcopy,
    traced_alias,
)
