from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from tidy3d.compat import Self

Comparator: TypeAlias = Literal["lt", "lte", "gt", "gte", "eq"]
ResultStatus: TypeAlias = Literal["pass", "fail"]


class Metric(BaseModel):
    """Machine-readable record for one evaluated numerical metric."""

    name: str
    observed: float
    expected: float
    comparator: Comparator

    model_config = ConfigDict(extra="forbid")

    def passes(self) -> bool:
        """Evaluate the metric against its expected value."""
        if self.comparator == "lt":
            return self.observed < self.expected
        if self.comparator == "lte":
            return self.observed <= self.expected
        if self.comparator == "gt":
            return self.observed > self.expected
        if self.comparator == "gte":
            return self.observed >= self.expected
        return self.observed == self.expected


class NumericalResult(BaseModel):
    """Machine-readable result for one numerical pytest case."""

    pytest_nodeid: str
    status: ResultStatus
    regression_metrics: list[Metric]
    observation_metrics: list[Metric] = []

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _require_regression_metrics(self) -> Self:
        """Ensure every numerical result contains at least one regression metric."""
        if not self.regression_metrics:
            raise ValueError("NumericalResult must contain at least one regression metric.")
        return self

    @model_validator(mode="after")
    def _validate_status_matches_regression_metrics(self) -> Self:
        """Ensure serialized status matches the regression metrics."""
        derived_status = self._status_from_regression_metrics(self.regression_metrics)
        if self.status != derived_status:
            raise ValueError(
                f"NumericalResult status {self.status!r} does not match regression metrics; "
                f"expected {derived_status!r}."
            )
        return self

    @staticmethod
    def _status_from_regression_metrics(regression_metrics: list[Metric]) -> ResultStatus:
        """Derive pass/fail status from regression metrics only."""
        return "pass" if all(metric.passes() for metric in regression_metrics) else "fail"

    @classmethod
    def from_metrics(
        cls,
        *,
        pytest_nodeid: str,
        regression_metrics: list[Metric],
        observation_metrics: list[Metric] | None = None,
    ) -> Self:
        """Build a numerical result directly from regression and observation metrics."""
        observation_metrics = observation_metrics or []
        status = cls._status_from_regression_metrics(regression_metrics)
        return cls(
            pytest_nodeid=pytest_nodeid,
            status=status,
            regression_metrics=regression_metrics,
            observation_metrics=observation_metrics,
        )

    def passes(self) -> bool:
        """Return whether the result passed all regression metrics."""
        return self._status_from_regression_metrics(self.regression_metrics) == "pass"

    def to_json_file(self, path: Path) -> Path:
        """Serialize the numerical result to a JSON file."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        """Load a numerical result from a JSON file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
