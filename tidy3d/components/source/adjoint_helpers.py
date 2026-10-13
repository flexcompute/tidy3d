"""Shared helper utilities for source-adjoint gradient processing."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.autograd.path_utils import format_traced_path
from tidy3d.components.autograd.utils import get_static
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tidy3d.components.autograd.types import PathType
    from tidy3d.components.types.base import Size


def validate_center_path_axes(path: PathType) -> tuple[int, ...]:
    """Validate and return axis indices addressed by a traced center path."""
    path = tuple(path)
    if not path or path[0] != "center":
        raise AdjointError(
            f"Expected 'center' traced source parameter, got '{format_traced_path(path)}'."
        )
    if len(path) == 1:
        return (0, 1, 2)
    if len(path) == 2:
        axis = path[1]
        if not isinstance(axis, Integral) or axis not in (0, 1, 2):
            raise AdjointError(
                f"Unsupported traced source parameter '{format_traced_path(path)}'. "
                "Only center, center[0], center[1], and center[2] are supported."
            )
        return (int(axis),)
    raise AdjointError(
        f"Unsupported traced source parameter '{format_traced_path(path)}'. "
        "Only full-vector paths or single-axis paths are supported for center."
    )


def validate_source_field_component(field_name: str, *, source_name: str) -> None:
    """Validate a source field component name like ``Ex`` or ``Hz``."""
    if (
        len(field_name) != 2
        or field_name[0] not in ("E", "H")
        or field_name[1] not in ("x", "y", "z")
    ):
        raise AdjointError(
            f"Unsupported field component '{field_name}' in {source_name}. "
            "Expected one of Ex, Ey, Ez, Hx, Hy, Hz."
        )


def validate_no_zero_dim_center_paths(
    center_paths: Sequence[PathType],
    *,
    source_size: Size,
    source_name: str,
) -> None:
    """Reject center derivatives on collapsed source axes."""
    source_size_arr = np.asarray(get_static(source_size), dtype=float)
    for field_path in center_paths:
        path = tuple(field_path)
        axes = validate_center_path_axes(path)

        for axis in axes:
            if np.isclose(source_size_arr[axis], 0.0):
                raise AdjointError(
                    f"{source_name} does not support derivatives on collapsed axis "
                    f"'{'xyz'[axis]}' for source parameter "
                    f"'{format_traced_path(path)}'."
                )
