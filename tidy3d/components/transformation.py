"""Defines geometric transformation classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pydantic.v1 as pd

from ..constants import RADIAN
from ..exceptions import ValidationError
from .base import Tidy3dBaseModel, cached_property
from .types import ArrayFloat2D, Axis, Coordinate, TensorReal


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

    axis: Union[Axis, Coordinate] = pd.Field(
        0,
        title="Axis of Rotation",
        description="A vector that specifies the axis of rotation, or a single int: 0, 1, or 2, "
        "indicating x, y, or z.",
    )

    angle: float = pd.Field(
        0.0,
        title="Angle of Rotation",
        description="Angle of rotation in radians.",
        units=RADIAN,
    )

    @pd.validator("axis", always=True)
    def _convert_axis_index_to_vector(cls, val):
        if not isinstance(val, tuple):
            axis = [0.0, 0.0, 0.0]
            axis[val] = 1.0
            val = tuple(axis)
        return val

    @pd.validator("axis")
    def _guarantee_nonzero_axis(cls, val):
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

        norm = np.linalg.norm(self.axis)
        n = self.axis / norm
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.zeros((3, 3))
        tan_dim = [[1, 2], [2, 0], [0, 1]]

        for dim in range(3):
            R[dim, dim] = c + n[dim] ** 2 * (1 - c)
            R[dim, tan_dim[dim][0]] = n[dim] * n[tan_dim[dim][0]] * (1 - c) - n[tan_dim[dim][1]] * s
            R[dim, tan_dim[dim][1]] = n[dim] * n[tan_dim[dim][1]] * (1 - c) + n[tan_dim[dim][0]] * s

        return R


RotationType = Union[RotationAroundAxis]
