# module providing classes for initializing the parameters in an inverse design problem

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from flexcompute.core._migration.em.invdes.validation import (
    validate_params_dtype,
    validate_params_range,
    validate_params_rank,
    validate_params_shape,
)
from flexcompute.core.validation import raise_first_error
from pydantic import Field, NonNegativeInt, field_validator

import tidy3d as td
from tidy3d._runtime import always_validate
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Minimum Value",
        description="Minimum value for the random parameters (inclusive).",
    )
    max_value: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Maximum Value",
        description="Maximum value for the random parameters (exclusive).",
    )
    seed: NonNegativeInt | None = Field(
        default=None,
        description="Seed for the random number generator.",
    )

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Generate the parameter array based on the specification."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.min_value, self.max_value, shape)


class UniformInitializationSpec(AbstractInitializationSpec):
    """Specification for uniform initial parameters."""

    value: float = Field(
        default=0.5,
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

    # The three checks stay field validators because of fast mode: the dtype check is
    # ``@always_validate`` (it casts and warns), so it runs there while range and rank are skipped, and
    # range has to precede it. Moving range and rank onto the construction path would run them in fast
    # mode and change which error comes first.
    @field_validator("params")
    @classmethod
    def _validate_params_range(cls, val: NDArray) -> NDArray:
        """Ensure that all parameter values are between 0 and 1."""
        raise_first_error(validate_params_range(val), ValidationError)
        return val

    @field_validator("params")
    @classmethod
    @always_validate
    def _validate_params_dtype(cls, val: NDArray) -> NDArray:
        """Ensure that params is real-valued."""
        raise_first_error(validate_params_dtype(val.dtype), ValidationError)
        if np.issubdtype(val.dtype, np.bool_):
            td.log.warning(
                "Got a boolean array for 'params'. This will be treated as a floating point array."
            )
            val = val.astype(float)
        return val

    @field_validator("params")
    @classmethod
    def _validate_params_3d(cls, val: NDArray) -> NDArray:
        """Ensure that params is a 3D array."""
        raise_first_error(validate_params_rank(val.ndim), ValidationError)
        return val

    def create_parameters(self, shape: tuple[int, ...]) -> NDArray:
        """Return the custom parameters provided by the user."""
        params = np.asarray(self.params)
        raise_first_error(validate_params_shape(params.shape, shape), ValueError)
        return params


InitializationSpecType = (
    RandomInitializationSpec | UniformInitializationSpec | CustomInitializationSpec
)
