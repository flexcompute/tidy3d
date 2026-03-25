"""Shared helpers for rotating tensor media and accumulating bend-frame rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from tidy3d.components.medium import AbstractCustomMedium, AnisotropicMedium, FullyAnisotropicMedium
from tidy3d.components.types import ArrayComplex2D, Axis2D
from tidy3d.constants import fp_eps

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from tidy3d.components.material.types import StructureMediumType
    from tidy3d.components.types import ArrayFloat1D, Axis, FreqArray, TensorReal

EMEAnisotropicMedium: TypeAlias = AnisotropicMedium | FullyAnisotropicMedium
MediumTensorSequence: TypeAlias = tuple[ArrayComplex2D, ...]


class BendModeSpecLike(Protocol):
    """Attributes needed to accumulate bend-frame rotations across cells."""

    bend_radius: float | None
    bend_axis: Axis2D | None
    bend_medium_frame: str


def medium_frequency_tensors(
    medium: EMEAnisotropicMedium, freqs: FreqArray
) -> MediumTensorSequence:
    """Complex permittivity tensors for ``medium`` at each frequency in ``freqs``."""
    tensors = []
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    for freq in freqs:
        if isinstance(medium, FullyAnisotropicMedium):
            permittivity = np.asarray(
                [
                    [medium.eps_comp(row=row, col=col, frequency=float(freq)) for col in range(3)]
                    for row in range(3)
                ],
                dtype=complex,
            )
        else:
            components = (medium.xx, medium.yy, medium.zz)
            permittivity = np.diag(
                [component.eps_model(float(freq)) for component in components]
            ).astype(complex)
        tensors.append(permittivity)
    return tuple(tensors)


def rotate_tensor(tensor: ArrayComplex2D, rotation_matrix: TensorReal) -> ArrayComplex2D:
    """Express a 3x3 tensor in a frame rotated by ``rotation_matrix``."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=complex)
    tensor = np.asarray(tensor, dtype=complex)
    return rotation_matrix @ tensor @ rotation_matrix.T


def medium_rotated_tensors(
    medium: EMEAnisotropicMedium, freqs: FreqArray, rotation_matrix: TensorReal
) -> MediumTensorSequence:
    """Complex permittivity tensors after applying ``rotation_matrix``."""
    return tuple(
        rotate_tensor(tensor, rotation_matrix)
        for tensor in medium_frequency_tensors(medium=medium, freqs=freqs)
    )


def rotated_tensors_equal(
    reference_tensors: MediumTensorSequence, comparison_tensors: MediumTensorSequence
) -> bool:
    """Whether two rotated-tensor collections match."""
    if len(reference_tensors) != len(comparison_tensors):
        return False

    def tensor_parts_close(
        reference_tensor: ArrayComplex2D, comparison_tensor: ArrayComplex2D
    ) -> bool:
        # Compare real and imaginary parts separately so a large conductive response does not
        # drown out a smaller real-valued anisotropy, while still allowing roundoff to scale with
        # the magnitude of each tensor part.
        for reference_part, comparison_part in (
            (reference_tensor.real, comparison_tensor.real),
            (reference_tensor.imag, comparison_tensor.imag),
        ):
            part_scale = max(
                float(np.max(np.abs(reference_part))),
                float(np.max(np.abs(comparison_part))),
                1.0,
            )
            if not np.allclose(
                reference_part,
                comparison_part,
                atol=fp_eps * part_scale,
                rtol=fp_eps,
            ):
                return False
        return True

    return all(
        tensor_parts_close(reference_tensor, comparison_tensor)
        for reference_tensor, comparison_tensor in zip(reference_tensors, comparison_tensors)
    )


def medium_is_rotation_invariant(
    medium: StructureMediumType, rotation_matrix: TensorReal, freqs: FreqArray
) -> bool:
    """Whether rotating a medium tensor by ``rotation_matrix`` leaves it unchanged."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if np.allclose(rotation_matrix, np.eye(3), atol=fp_eps, rtol=0):
        return True
    if not isinstance(medium, (AnisotropicMedium, FullyAnisotropicMedium)):
        return True
    # Custom anisotropic media expose only spatially averaged tensor components via
    # ``eps_model()``, which is insufficient to certify invariance under plane rotation.
    if isinstance(medium, AbstractCustomMedium):
        return False
    return rotated_tensors_equal(
        reference_tensors=medium_rotated_tensors(
            medium=medium, freqs=freqs, rotation_matrix=np.eye(3)
        ),
        comparison_tensors=medium_rotated_tensors(
            medium=medium, freqs=freqs, rotation_matrix=rotation_matrix
        ),
    )


def rotation_matrix_about_local_axis(axis: Axis, angle: float) -> TensorReal:
    """Rotation matrix about an axis in the local mode-solver basis."""
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    rotation = np.eye(3, dtype=float)
    if axis == 0:
        rotation[1, 1] = cos_t
        rotation[1, 2] = -sin_t
        rotation[2, 1] = sin_t
        rotation[2, 2] = cos_t
    elif axis == 1:
        rotation[0, 0] = cos_t
        rotation[0, 2] = sin_t
        rotation[2, 0] = -sin_t
        rotation[2, 2] = cos_t
    else:
        rotation[0, 0] = cos_t
        rotation[0, 1] = -sin_t
        rotation[1, 0] = sin_t
        rotation[1, 1] = cos_t
    return rotation


def bend_axis_global_axis(normal_axis: Axis, bend_axis: Axis2D) -> Axis:
    """Global axis matching the bend convention used by the mode solver."""
    axis_order = {
        0: (1, 2, 0),
        1: (0, 2, 1),
        2: (0, 1, 2),
    }[normal_axis]
    return axis_order[bend_axis]


def cell_center_rotations_from_lengths(
    lengths: ArrayFloat1D, mode_specs: Sequence[BendModeSpecLike], normal_axis: Axis
) -> tuple[TensorReal, ...]:
    """Absolute local-frame rotations at the center of each EME cell."""
    rotations = []
    cumulative_rotation = np.eye(3, dtype=float)
    for cell_length, mode_spec in zip(lengths, mode_specs):
        use_global_frame = getattr(mode_spec, "bend_medium_frame", "global") == "global"
        if mode_spec.bend_radius is None or mode_spec.bend_axis is None:
            rotations.append(
                np.array(cumulative_rotation, copy=True) if use_global_frame else np.eye(3)
            )
            continue

        bend_angle = cell_length / mode_spec.bend_radius
        bend_axis = bend_axis_global_axis(normal_axis=normal_axis, bend_axis=mode_spec.bend_axis)
        half_rotation = rotation_matrix_about_local_axis(axis=bend_axis, angle=0.5 * bend_angle)
        full_rotation = rotation_matrix_about_local_axis(axis=bend_axis, angle=bend_angle)
        rotations.append(cumulative_rotation @ half_rotation if use_global_frame else np.eye(3))
        cumulative_rotation = cumulative_rotation @ full_rotation

    return tuple(rotations)
