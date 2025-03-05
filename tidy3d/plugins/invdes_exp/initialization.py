# module providing classes for initializing the parameters in an inverse design problem

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd
from numpy.typing import NDArray

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike
from tidy3d.exceptions import ValidationError


class AbstractInitializationSpec(Tidy3dBaseModel, ABC):
    """Abstract base class for initialization specifications."""

    @abstractmethod
    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""
        pass


class RandomInitializationSpec(AbstractInitializationSpec):
    """Specification for random initial parameters.

    When a seed is provided, a call to `create_parameters` will always return the same array.
    """

    min_value: float = pd.Field(
        0.0,
        ge=0.0,
        le=1.0,
        title="Minimum Value",
        description="Minimum value for the random parameters (inclusive).",
    )
    max_value: float = pd.Field(
        1.0,
        ge=0.0,
        le=1.0,
        title="Maximum Value",
        description="Maximum value for the random parameters (exclusive).",
    )
    seed: Optional[pd.NonNegativeInt] = pd.Field(
        None, description="Seed for the random number generator."
    )

    @pd.root_validator(pre=False)
    def _validate_max_ge_min(cls, values):
        """Ensure that max_value is greater than or equal to min_value."""
        minval = values.get("min_value")
        maxval = values.get("max_value")
        if minval > maxval:
            raise ValidationError(
                f"'max_value' ({maxval}) must be greater or equal than 'min_value' ({minval})"
            )
        return values

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.min_value, self.max_value, shape)


class UniformInitializationSpec(AbstractInitializationSpec):
    """Specification for uniform initial parameters."""

    value: float = pd.Field(
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

    params: ArrayLike = pd.Field(
        ...,
        title="Parameters",
        description="Custom parameters provided by the user.",
    )

    @pd.validator("params")
    def _validate_params_range(cls, value, values):
        """Ensure that all parameter values are between 0 and 1."""
        if np.any((value < 0) | (value > 1)):
            raise ValidationError("'params' need to be between 0 and 1.")
        return value

    @pd.validator("params")
    def _validate_params_dtype(cls, value, values):
        """Ensure that params is real-valued."""
        if np.issubdtype(value.dtype, np.bool_):
            td.log.warning(
                "Got a boolean array for 'params'. "
                "This will be treated as a floating point array."
            )
            value = value.astype(float)
        elif not np.issubdtype(value.dtype, np.floating):
            raise ValidationError(f"'params' need to be real-valued, but got '{value.dtype}'.")
        return value

    @pd.validator("params")
    def _validate_params_3d(cls, value, values):
        """Ensure that params is a 3D array."""
        if value.ndim != 3:
            raise ValidationError(f"'params' must be 3D, but got {value.ndim}D.")
        return value

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
