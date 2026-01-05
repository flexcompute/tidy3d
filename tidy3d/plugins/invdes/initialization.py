# module providing classes for initializing the parameters in an inverse design problem

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from pydantic import Field, NonNegativeInt, field_validator, model_validator

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self


class AbstractInitializationSpec(Tidy3dBaseModel, ABC):
    """Abstract base class for initialization specifications."""

    @abstractmethod
    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""


class RandomInitializationSpec(AbstractInitializationSpec):
    """Specification for random initial parameters.

    Notes
    -----
        When a seed is provided, a call to `create_parameters` will always return the same array.
    """

    min_value: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        title="Minimum Value",
        description="Minimum value for the random parameters (inclusive).",
    )
    max_value: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        title="Maximum Value",
        description="Maximum value for the random parameters (exclusive).",
    )
    seed: Optional[NonNegativeInt] = Field(
        None,
        description="Seed for the random number generator.",
    )

    @model_validator(mode="after")
    def _validate_max_ge_min(self) -> Self:
        """Ensure that max_value is greater than or equal to min_value."""
        if self.min_value > self.max_value:
            raise ValidationError(
                f"'max_value' ({self.max_value}) must be greater or equal than 'min_value' ({self.min_value})"
            )
        return self

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.min_value, self.max_value, shape)


class UniformInitializationSpec(AbstractInitializationSpec):
    """Specification for uniform initial parameters."""

    value: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        title="Value",
        description="Value to use for all elements in the parameter array.",
    )

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""
        return np.full(shape, self.value)


class CustomInitializationSpec(AbstractInitializationSpec):
    """Specification for custom initial parameters provided by the user."""

    params: ArrayLike = Field(
        ...,
        title="Parameters",
        description="Custom parameters provided by the user.",
    )

    @field_validator("params")
    @classmethod
    def _validate_params_range(cls, val: NDArray) -> NDArray:
        """Ensure that all parameter values are between 0 and 1."""
        if np.any((val < 0) | (val > 1)):
            raise ValidationError("'params' need to be between 0 and 1.")
        return val

    @field_validator("params")
    @classmethod
    def _validate_params_dtype(cls, val: NDArray) -> NDArray:
        """Ensure that params is real-valued."""
        if np.issubdtype(val.dtype, np.bool_):
            td.log.warning(
                "Got a boolean array for 'params'. This will be treated as a floating point array."
            )
            val = val.astype(float)
        elif not np.issubdtype(val.dtype, np.floating):
            raise ValidationError(f"'params' need to be real-valued, but got '{val.dtype}'.")
        return val

    @field_validator("params")
    @classmethod
    def _validate_params_3d(cls, val: NDArray) -> NDArray:
        """Ensure that params is a 3D array."""
        if val.ndim != 3:
            raise ValidationError(f"'params' must be 3D, but got {val.ndim}D.")
        return val

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Return the custom parameters provided by the user."""
        params = np.asarray(self.params)
        if params.shape != shape:
            raise ValueError(
                f"Provided 'params.shape' ('{params.shape}') does not match "
                f"the shape of the custom parameters ('{shape}')."
            )
        return params


InitializationSpecType = Union[
    RandomInitializationSpec,
    UniformInitializationSpec,
    CustomInitializationSpec,
]
