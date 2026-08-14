"""Validation helpers shared by mode solving and FDTD simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from tidy3d.components.material.tensor_rotation import (
    medium_is_rotation_invariant,
    rotation_matrix_about_local_axis,
)
from tidy3d.components.medium import (
    AnisotropicMedium,
    FullyAnisotropicMedium,
    IsotropicUniformMediumType,
)
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError
from tidy3d.log import log

from .geometry import effective_mode_plane, mode_plane_grid

if TYPE_CHECKING:
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Axis, FreqArray, TensorReal
    from tidy3d.components.types.mode_spec import ModeSpecType

WARN_THICK_PML_PERCENT = 50


def warn_thick_pml(
    simulation: Any,
    plane: Box,
    mode_spec: ModeSpecType,
    msg_prefix: str = "'ModeSolver'",
) -> None:
    """Warn if mode-solver PML covers a significant portion of the mode plane."""
    coord_0, coord_1 = mode_plane_grid(simulation=simulation, plane=plane)
    num_cells = [len(coord_0), len(coord_1)]
    effective_num_pml = (
        min(mode_spec.num_pml[0], len(coord_0) - 1),
        min(mode_spec.num_pml[1], len(coord_1) - 1),
    )
    for index in (0, 1):
        if 2 * effective_num_pml[index] > (WARN_THICK_PML_PERCENT / 100) * num_cells[index]:
            log.warning(
                f"{msg_prefix}: "
                f"The mode solver pml in tangential axis '{index}' "
                f"covers more than '{WARN_THICK_PML_PERCENT}%' of the "
                "mode plane cells. Consider using a larger mode plane "
                "or smaller 'num_pml'."
            )


def validate_mode_plane_radius(mode_spec: ModeSpecType, plane: Box, sim_geom: Box) -> None:
    """Validate that a bend radius is not smaller than half the radial plane size."""
    if not mode_spec.bend_radius:
        return

    mode_plane = effective_mode_plane(plane, sim_geom)
    _, plane_axes = mode_plane.pop_axis([0, 1, 2], mode_plane.size.index(0.0))
    radial_axis = plane_axes[(mode_spec.bend_axis + 1) % 2]
    if np.abs(mode_spec.bend_radius) <= mode_plane.size[radial_axis] / 2 + fp_eps:
        raise ValueError(
            "Mode solver bend radius is smaller than half the mode plane size "
            "along the radial axis, which can produce wrong results."
        )


def medium_supports_plane_rotation(
    medium: object, rotation_matrix: TensorReal, freqs: FreqArray
) -> bool:
    """Return whether a medium supports angled-plane structure rotation."""
    is_uniform_isotropic = isinstance(medium, get_args(IsotropicUniformMediumType))
    is_rotation_invariant_anisotropic = isinstance(
        medium, AnisotropicMedium | FullyAnisotropicMedium
    ) and medium_is_rotation_invariant(medium=medium, rotation_matrix=rotation_matrix, freqs=freqs)
    return is_uniform_isotropic or is_rotation_invariant_anisotropic


def validate_plane_rotation_media(
    mediums: list[object],
    rotate_kwargs: dict[str, float | Axis],
    freqs: FreqArray,
) -> None:
    """Reject angled-plane rotations through unsupported intersecting media."""
    rotation_matrix = rotation_matrix_about_local_axis(
        axis=rotate_kwargs["axis"], angle=rotate_kwargs["angle"]
    )
    if all(
        medium_supports_plane_rotation(
            medium=medium,
            rotation_matrix=rotation_matrix,
            freqs=freqs,
        )
        for medium in mediums
    ):
        return
    raise SetupError(  # post-init-tidy3d-error: ignore
        "'angle_rotation' set to True but the mode solver plane intersects an unsupported "
        "medium. Only uniform isotropic media and rotation-invariant anisotropic media are "
        "supported for the plane rotation."
    )


def make_rotated_structures(
    structures: list[Structure],
    translate_kwargs: dict[str, float],
    rotate_kwargs: dict[str, float | Axis],
    freqs: FreqArray,
) -> list[Structure]:
    """Rotate structures intersecting an angled mode plane."""
    try:
        rotated_structures = []
        rotation_matrix = rotation_matrix_about_local_axis(
            axis=rotate_kwargs["axis"], angle=rotate_kwargs["angle"]
        )
        for structure in structures:
            if not medium_supports_plane_rotation(
                medium=structure.medium,
                rotation_matrix=rotation_matrix,
                freqs=freqs,
            ):
                raise NotImplementedError(
                    "Mode solver plane intersects an unsupported medium. "
                    "Only uniform isotropic media and rotation-invariant anisotropic "
                    "media are supported for the plane rotation."
                )

            geometry = structure.geometry
            geometry = (
                geometry.translated(**{key: -val for key, val in translate_kwargs.items()})
                .rotated(**rotate_kwargs)
                .translated(**translate_kwargs)
            )
            rotated_structures.append(structure.updated_copy(geometry=geometry))
        return rotated_structures
    except Exception as exc:
        raise SetupError(  # post-init-tidy3d-error: ignore
            f"'angle_rotation' set to True but could not rotate structures: {exc!s}"
        ) from exc


def validate_microwave_mode_spec(mode_spec: MicrowaveModeSpec, plane: Box) -> None:
    """Validate path integrals configured on a microwave mode specification."""
    mode_spec._check_path_integrals_within_box(plane)
