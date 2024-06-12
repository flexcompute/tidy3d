"""Defines the dimensions of the parameter sweep and their properties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel


class Parameter(Tidy3dBaseModel, ABC):
    """Specification for a single variable / dimension in a design problem."""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the variable. Used as a key into the parameter sweep results.",
    )

    values: Tuple[Any, ...] = pd.Field(
        None,
        title="Custom Values",
        description="If specified, the parameter scan uses these values for grid search methods.",
    )

    @pd.validator("values", always=True)
    def _values_unique(cls, val):
        """Supplied unique values."""
        if (val is not None) and (len(set(val)) != len(val)):
            raise ValueError("Supplied 'values' were not unique.")
        return val

    def sample_grid(self) -> List[Any]:
        """Sample design variable on grid, checking for custom values."""
        if self.values is not None:
            return self.values
        return self._sample_grid()

    @abstractmethod
    def sample_random(self, num_samples: int) -> List[Any]:
        """Sample this design variable randomly 'num_samples' times."""

    @abstractmethod
    def _sample_grid(self) -> List[Any]:
        """Sample this design variable on a grid."""

    @abstractmethod
    def select_from_01(self, pts_01: np.ndarray) -> List[Any]:
        """Select values given a set of points between 0, 1."""

    @abstractmethod
    def sample_first(self) -> Any:
        """Output the first allowed sample."""


class ParameterNumeric(Parameter, ABC):
    """A variable with numeric values."""

    span: Tuple[Union[float, int], Union[float, int]] = pd.Field(
        ...,
        title="Span",
        description="(min, max) range within which are allowed values for the variable. Is inclusive of max value.",
    )

    @pd.validator("span", always=True)
    def _span_valid(cls, val):
        """Span min <= span max."""
        span_min, span_max = val
        if span_min > span_max:
            raise ValueError(
                f"Given invalid span '{val}'. The 1st value can't be greater than the 2nd value."
            )
        return val

    @property
    def span_size(self):
        """Size of the span of this numeric variable."""
        span_min = min(self.span)
        span_max = max(self.span)
        return span_max - span_min

    def sample_first(self) -> tuple:
        """Output the first allowed sample."""
        return self.span[0]


class ParameterFloat(ParameterNumeric):
    """Parameter containing floats.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> var = tdd.ParameterFloat(name="x", num_points=10, span=(1, 2.5))
    """

    num_points: pd.PositiveInt = pd.Field(
        None,
        title="Number of Points",
        description="Number of uniform sampling points for this variable. "
        "Only used for 'MethodGrid'. ",
    )

    @pd.validator("span", always=True)
    def _span_is_float(cls, val):
        """Make sure the span contains floats."""
        low, high = val
        return float(low), float(high)

    def sample_random(self, num_samples: int) -> List[float]:
        """Sample this design variable randomly 'num_samples' times."""
        low, high = self.span
        return np.random.uniform(low=low, high=high, size=num_samples).tolist()

    def _sample_grid(self) -> List[float]:
        """Sample this design variable on a grid."""
        if self.num_points is None:
            raise ValueError(
                "'ParameterFloat' sampled on a grid must have '.num_points' defined." ""
            )
        low, high = self.span
        return np.linspace(low, high, self.num_points).tolist()

    def select_from_01(self, pts_01: np.ndarray) -> List[Any]:
        """Select values given a set of points between 0, 1."""
        return (min(self.span) + pts_01 * self.span_size).tolist()


class ParameterInt(ParameterNumeric):
    """Parameter containing integers.


    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> var = tdd.ParameterInt(name="x", span=(1, 4))
    """

    span: Tuple[int, int] = pd.Field(
        ...,
        title="Span",
        description="``(min, max)`` range within which are allowed values for the variable. "
        "The ``min`` value is inclusive and the ``max`` value is exclusive. In other words, "
        "a grid search over this variable will iterate over ``np.arange(min, max)``.",
    )

    @pd.validator("span", always=True)
    def _span_is_int(cls, val):
        """Make sure the span contains ints."""
        low, high = val
        return int(low), int(high)

    def sample_random(self, num_samples: int) -> List[int]:
        """Sample this design variable randomly 'num_samples' times."""
        low, high = self.span
        return np.random.randint(low=low, high=high, size=num_samples).tolist()

    def _sample_grid(self) -> List[float]:
        """Sample this design variable on a grid."""
        low, high = self.span
        return np.arange(low, high).tolist()

    def select_from_01(self, pts_01: np.ndarray) -> List[Any]:
        """Select values given a set of points between 0, 1."""
        pts_continuous = min(self.span) + pts_01 * self.span_size
        return np.floor(pts_continuous).astype(int).tolist()


class ParameterAny(Parameter):
    """Parameter containing a set of of anything.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> var = tdd.ParameterAny(name="x", allowed_values=("a", "b", "c"))
    """

    allowed_values: Tuple[Any, ...] = pd.Field(
        ...,
        title="Allowed Values",
        description="The discrete set of values that this variable can take on.",
    )

    @pd.validator("allowed_values", always=True)
    def _given_any_allowed_values(cls, val):
        """Need at least one allowed value."""
        if not len(val):
            raise ValueError("Given empty tuple of allowed values. Must have at least one.")
        return val

    @pd.validator("allowed_values", always=True)
    def _no_duplicate_allowed_values(cls, val):
        """No duplicates in allowed_values."""
        if len(val) != len(set(val)):
            raise ValueError("'allowed_values' has duplicate entries, must be unique.")
        return val

    def sample_random(self, num_samples: int) -> List[Any]:
        """Sample this design variable randomly 'num_samples' times."""
        return np.random.choice(self.allowed_values, size=int(num_samples)).tolist()

    def _sample_grid(self) -> List[Any]:
        """Sample this design variable uniformly, ie just take all allowed values."""
        return list(self.allowed_values)

    def select_from_01(self, pts_01: np.ndarray) -> List[Any]:
        """Select values given a set of points between 0, 1."""
        pts_continuous = pts_01 * len(self.allowed_values)
        indices = np.floor(pts_continuous).astype(int)
        return np.array(self.allowed_values)[indices].tolist()

    def sample_first(self) -> Any:
        """Output the first allowed sample."""
        return self.allowed_values[0]


ParameterType = Union[ParameterInt, ParameterFloat, ParameterAny]
