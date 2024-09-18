# parameters.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayLike
from tidy3d.plugins.invdes.region import DesignRegion


class AbstractParameterSpec(Tidy3dBaseModel, ABC):
    """Abstract base class for parameter specifications."""

    shape: Optional[tuple[int, ...]] = pd.Field(None, description="Shape of the parameter array.")

    @abstractmethod
    def create_parameters(self, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        pass

    def _get_shape(self, shape: Optional[tuple[int, ...]] = None) -> tuple[int, ...]:
        """Get the shape for the parameter array, ensuring only one of shape or self.shape is provided."""
        if shape is not None and self.shape is not None:
            raise ValueError("Only one of shape or self.shape can be provided, not both.")
        if shape is None and self.shape is None:
            raise ValueError(
                "Shape must be provided either on initialization or when creating parameters."
            )
        return shape if shape is not None else self.shape


class RandomParameterSpec(AbstractParameterSpec):
    """Specification for random initial parameters."""

    min_value: float = pd.Field(0.0, description="Minimum value for the random parameters.")
    max_value: float = pd.Field(1.0, description="Maximum value for the random parameters.")
    seed: Optional[int] = pd.Field(None, description="Seed for the random number generator.")

    @classmethod
    def from_design_region(
        cls,
        design_region: DesignRegion,
        min_value: float = 0.0,
        max_value: float = 1.0,
        seed: int = None,
    ):
        """Create a RandomParameterSpec with the correct shape from a design region."""
        shape = design_region.params_shape
        return cls(min_value=min_value, max_value=max_value, shape=shape, seed=seed)

    def create_parameters(self, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        shape = self._get_shape(shape)
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.min_value, self.max_value, shape)


class UniformParameterSpec(AbstractParameterSpec):
    """Specification for uniform initial parameters."""

    value: float = pd.Field(
        0.5, description="Value to use for all elements in the parameter array."
    )

    @classmethod
    def from_design_region(cls, design_region: DesignRegion, value: Optional[float] = None):
        """Create a UniformParameterSpec with the correct shape from a design region."""
        if value is None:
            value = cls.__fields__["value"].default
        shape = design_region.params_shape
        return cls(value=value, shape=shape)

    def create_parameters(self, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Generate the parameter array based on the specification."""
        shape = self._get_shape(shape)
        return np.full(shape, self.value)


class CustomParameterSpec(AbstractParameterSpec):
    """Specification for custom initial parameters provided by the user."""

    params: ArrayLike = pd.Field(..., description="Custom parameters provided by the user.")

    def create_parameters(self, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Return the custom parameters provided by the user."""
        params = np.asarray(self.params)
        if shape is not None and params.shape != shape:
            raise ValueError("Provided shape does not match the shape of the custom parameters.")
        return params


ParameterSpecType = Union[RandomParameterSpec, UniformParameterSpec, CustomParameterSpec]
