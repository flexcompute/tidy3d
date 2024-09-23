# module providing classes for initializing the parameters in an inverse design problem

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike


class AbstractInitializationSpec(Tidy3dBaseModel, ABC):
    """Abstract base class for initialization specifications."""

    @abstractmethod
    def create_parameters(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        pass


class RandomInitializationSpec(AbstractInitializationSpec):
    """Specification for random initial parameters.

    When a seed is provided, a call to `create_parameters` will always return the same array.
    """

    min_value: float = pd.Field(
        0.0,
        title="Minimum Value",
        description="Minimum value for the random parameters (inclusive).",
    )
    max_value: float = pd.Field(
        1.0,
        title="Minimum Value",
        description="Maximum value for the random parameters (exclusive).",
    )
    seed: Optional[int] = pd.Field(None, description="Seed for the random number generator.")

    def create_parameters(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.min_value, self.max_value, shape)


class UniformInitializationSpec(AbstractInitializationSpec):
    """Specification for uniform initial parameters."""

    value: float = pd.Field(
        0.5,
        title="Value",
        description="Value to use for all elements in the parameter array.",
    )

    def create_parameters(self, shape: tuple[int, ...]) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        return np.full(shape, self.value)


class CustomInitializationSpec(AbstractInitializationSpec):
    """Specification for custom initial parameters provided by the user."""

    params: ArrayLike = pd.Field(
        ...,
        title="Parameters",
        description="Custom parameters provided by the user.",
    )

    def create_parameters(self, shape: tuple[int, ...]) -> np.ndarray:
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
