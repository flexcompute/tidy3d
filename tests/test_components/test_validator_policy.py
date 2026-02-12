"""Tests enforcing validator policy for selected class families."""

from __future__ import annotations

from pydantic import model_validator

import tidy3d as td
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.medium import AbstractMedium


def _iter_subclasses(cls: type) -> list[type]:
    subclasses: list[type] = []
    for subcls in cls.__subclasses__():
        subclasses.append(subcls)
        subclasses.extend(_iter_subclasses(subcls))
    return subclasses


ALLOWED_AFTER_MODEL_VALIDATORS = {
    "_run_after_validators",
    "__tidy3d_end_capture__",
}


def _find_after_model_validators(classes: list[type], allowed: set[str]) -> list[tuple[type, str]]:
    violations: list[tuple[type, str]] = []
    for cls in classes:
        decorators = getattr(cls, "__pydantic_decorators__", None)
        if not decorators:
            continue
        for name, decorator in decorators.model_validators.items():
            if decorator.info.mode == "after" and name not in allowed:
                violations.append((cls, name))
    return violations


def test_no_after_model_validators_in_medium_and_simulation_families() -> None:
    _ = td.Medium
    classes = [AbstractMedium, AbstractSimulation]
    classes.extend(_iter_subclasses(AbstractMedium))
    classes.extend(_iter_subclasses(AbstractSimulation))

    violations = _find_after_model_validators(classes, ALLOWED_AFTER_MODEL_VALIDATORS)

    assert not violations, (
        "Found @model_validator(mode='after') outside _run_after_validators(): "
        + ", ".join(f"{cls.__module__}.{cls.__name__}.{name}" for cls, name in violations)
    )


def test_after_model_validator_policy_detection_example() -> None:
    class BadModel(Tidy3dBaseModel):
        @model_validator(mode="after")
        def _bad_validator(self) -> BadModel:
            return self

    violations = _find_after_model_validators([BadModel], ALLOWED_AFTER_MODEL_VALIDATORS)

    assert violations == [(BadModel, "_bad_validator")]
