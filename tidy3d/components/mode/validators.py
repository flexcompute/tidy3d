"""Defines various validation functions that are specific to the mode solver"""

import numpy as np

from ..geometry.base import Box
from ..mode_spec import ModeSpec


def validate_mode_plane_radius(mode_spec: ModeSpec, plane: Box, msg_prefix: str = ""):
    """Validate that the radius of a mode spec with a bend is not smaller than half the size of
    the plane along the radial direction."""

    if not mode_spec.bend_radius:
        return

    # radial axis is the plane axis that is not the bend axis
    _, plane_axs = plane.pop_axis([0, 1, 2], plane.size.index(0.0))
    radial_ax = plane_axs[(mode_spec.bend_axis + 1) % 2]

    if np.abs(mode_spec.bend_radius) < plane.size[radial_ax] / 2:
        raise ValueError(
            f"{msg_prefix} bend radius is smaller than half the mode plane size "
            "along the radial axis, which can produce wrong results."
        )
