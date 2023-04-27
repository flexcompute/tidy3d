"""Defines geometric transformation classes"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import pydantic as pd
import numpy as np

from .base import Tidy3dBaseModel, cached_property
from .types import Coordinate, TensorReal, ArrayFloat2D
from ..constants import RADIAN


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

    axis: Coordinate = pd.Field(
        [1, 0, 0],
        title="Axis of Rotation",
        description="A vector that specifies the axis of rotation.",
    )

    angle: float = pd.Field(
        0.0,
        title="Angle of Rotation",
        description="Angle of rotation in radians.",
        units=RADIAN,
    )

    @cached_property
    def isidentity(self) -> bool:
        """Check whether rotation is identity."""

        return np.isclose(self.angle % (2 * np.pi), 0)

    @cached_property
    # pylint: disable=invalid-name
    def matrix(self) -> TensorReal:
        """Rotation matrix."""

        if self.isidentity:
            return np.eye(3)

        n = self.axis / np.linalg.norm(self.axis)
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
