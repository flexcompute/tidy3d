"""Defines geometric transformation classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

import autograd.numpy as np
from pydantic import Field, field_validator

from tidy3d.constants import RADIAN
from tidy3d.exceptions import ValidationError

from .autograd import TracedFloat
from .base import Tidy3dBaseModel, cached_property
from .types import Axis, Coordinate

if TYPE_CHECKING:
    from .types import ArrayFloat2D, TensorReal


def rotation_matrix_around_axis(axis: Coordinate, angle: Any) -> TensorReal:
    """Rotation matrix from axis-angle representation (Rodrigues form)."""
    axis_arr = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis_arr)
    n = axis_arr / norm
    c = np.cos(angle)
    s = np.sin(angle)
    k_mat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + s * k_mat + (1 - c) * k_mat @ k_mat


def rotate_points_around_axis(points: ArrayFloat2D, axis: Coordinate, angle: Any) -> ArrayFloat2D:
    """Rotate point cloud or vector stack with shape ``(3, ...)``."""
    rot = rotation_matrix_around_axis(axis=axis, angle=angle)
    flat = points.reshape((3, -1))
    return np.matmul(rot, flat).reshape(points.shape)


class AbstractRotation(ABC, Tidy3dBaseModel):
    """Abstract rotation of vectors and tensors."""

    @cached_property
    @abstractmethod
    def matrix(self) -> TensorReal:
        """Rotation matrix."""

    @cached_property
    @abstractmethod
    def isidentity(self) -> bool:
        """Check whether rotation is identity."""

    def rotate_vector(self, vector: ArrayFloat2D) -> ArrayFloat2D:
        """Rotate a vector/point or a list of vectors/points.

        Parameters
        ----------
        points : ArrayLike[float]
            Array of shape ``(3, ...)``.

        Returns
        -------
        Coordinate
            Rotated vector.
        """

        if self.isidentity:
            return vector

        if len(vector.shape) == 1:
            return self.matrix @ vector

        return np.tensordot(self.matrix, vector, axes=1)

    def rotate_tensor(self, tensor: TensorReal) -> TensorReal:
        """Rotate a tensor.

        Parameters
        ----------
        tensor : ArrayLike[float]
            Array of shape ``(3, 3)``.

        Returns
        -------
        TensorReal
            Rotated tensor.
        """

        if self.isidentity:
            return tensor

        return np.matmul(self.matrix, np.matmul(tensor, self.matrix.T))


class RotationAroundAxis(AbstractRotation):
    """Rotation of vectors and tensors around a given vector."""

    axis: Union[Axis, Coordinate] = Field(
        0,
        title="Axis of Rotation",
        description="A vector that specifies the axis of rotation, or a single int: 0, 1, or 2, "
        "indicating x, y, or z.",
    )

    angle: TracedFloat = Field(
        0.0,
        title="Angle of Rotation",
        description="Angle of rotation in radians.",
        json_schema_extra={"units": RADIAN},
    )

    @field_validator("axis")
    @classmethod
    def _validate_axis_vector(cls, val: Union[Axis, Coordinate]) -> Coordinate:
        if not isinstance(val, tuple):
            axis = [0.0, 0.0, 0.0]
            axis[val] = 1.0
            val = tuple(axis)
        return val

    @field_validator("axis")
    @classmethod
    def _validate_axis_nonzero_norm(cls, val: Coordinate) -> Coordinate:
        norm = np.linalg.norm(val)
        if np.isclose(norm, 0):
            raise ValidationError(
                "The norm of vector 'axis' cannot be zero. Please provide a proper rotation axis."
            )
        return val

    @cached_property
    def isidentity(self) -> bool:
        """Check whether rotation is identity."""

        return np.isclose(self.angle % (2 * np.pi), 0)

    @cached_property
    def matrix(self) -> TensorReal:
        """Rotation matrix."""

        if self.isidentity:
            return np.eye(3)
        return rotation_matrix_around_axis(axis=self.axis, angle=self.angle)


class AbstractReflection(ABC, Tidy3dBaseModel):
    """Abstract reflection of vectors and tensors."""

    @cached_property
    @abstractmethod
    def matrix(self) -> TensorReal:
        """Reflection matrix."""

    def reflect_vector(self, vector: ArrayFloat2D) -> ArrayFloat2D:
        """Reflect a vector/point or a list of vectors/points.

        Parameters
        ----------
        vector : ArrayLike[float]
            Array of shape ``(3, ...)``.

        Returns
        -------
        Coordinate
            Reflected vector.
        """

        if len(vector.shape) == 1:
            return self.matrix @ vector

        return np.tensordot(self.matrix, vector, axes=1)

    def reflect_tensor(self, tensor: TensorReal) -> TensorReal:
        """Reflect a tensor.

        Parameters
        ----------
        tensor : ArrayLike[float]
            Array of shape ``(3, 3)``.

        Returns
        -------
        TensorReal
            Reflected tensor.
        """

        return np.matmul(self.matrix, np.matmul(tensor, self.matrix.T))


class ReflectionFromPlane(AbstractReflection):
    """Reflection of vectors and tensors around a given vector."""

    normal: Coordinate = Field(
        (1, 0, 0),
        title="Normal of the reflecting plane",
        description="A vector that specifies the normal of the plane of reflection",
    )

    @field_validator("normal")
    @classmethod
    def _validate_normal_nonzero_norm(cls, val: Coordinate) -> Coordinate:
        norm = np.linalg.norm(val)
        if np.isclose(norm, 0):
            raise ValidationError(
                "The norm of vector 'normal' cannot be zero. Please provide a proper normal vector."
            )
        return val

    @cached_property
    def matrix(self) -> TensorReal:
        """Reflection matrix."""

        norm = np.linalg.norm(self.normal)
        n = self.normal / norm
        R = np.eye(3) - 2 * np.outer(n, n)

        return R


RotationType = Union[RotationAroundAxis]
ReflectionType = Union[ReflectionFromPlane]
