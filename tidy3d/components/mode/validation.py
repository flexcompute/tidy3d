"""Validation helpers shared by mode solving and FDTD simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from tidy3d.components.material.tensor_rotation import (
    medium_is_rotation_invariant,
    rotation_matrix_about_local_axis,
)
from tidy3d.components.medium import (
    AnisotropicMedium,
    FullyAnisotropicMedium,
    IsotropicUniformMediumType,
)
from tidy3d.components.structure import Structure
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Axis, FreqArray, TensorReal


def medium_supports_plane_rotation(
    medium: object, rotation_matrix: TensorReal, freqs: FreqArray
) -> bool:
    """Return whether a medium supports angled-plane structure rotation."""
    medium = Structure._get_public_optical_medium(medium)
    is_uniform_isotropic = isinstance(medium, get_args(IsotropicUniformMediumType))
    is_rotation_invariant_anisotropic = isinstance(
        medium, AnisotropicMedium | FullyAnisotropicMedium
    ) and medium_is_rotation_invariant(medium=medium, rotation_matrix=rotation_matrix, freqs=freqs)
    return is_uniform_isotropic or is_rotation_invariant_anisotropic


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
