"""Resolve medium-model forward references after package assembly."""

from __future__ import annotations

from typing import TypeVar

from .abstract_custom import AbstractCustomMedium
from .anisotropic import AnisotropicMedium, AnisotropicMediumFromMedium2D
from .medium_types import (
    IsotropicCustomMediumInternalType,
    IsotropicCustomMediumType,
    IsotropicUniformMediumFor2DType,
    IsotropicUniformMediumType,
    MediumType3D,
)
from .perturbation import PerturbationMediumType
from .two_d import Medium2D

T = TypeVar("T")


def _get_all_subclasses(cls: type[T]) -> list[type[T]]:
    """Recursively collect every subclass of ``cls``."""
    subclasses: list[type[T]] = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(_get_all_subclasses(subclass))
    return subclasses


def rebuild_medium_models() -> None:
    """Resolve aliases that require all medium families to be defined first."""
    types_namespace = {
        "IsotropicCustomMediumInternalType": IsotropicCustomMediumInternalType,
        "IsotropicCustomMediumType": IsotropicCustomMediumType,
        "IsotropicUniformMediumFor2DType": IsotropicUniformMediumFor2DType,
        "IsotropicUniformMediumType": IsotropicUniformMediumType,
        "MediumType3D": MediumType3D,
        "PerturbationMediumType": PerturbationMediumType,
    }

    model_classes = (
        AbstractCustomMedium,
        AnisotropicMedium,
        AnisotropicMediumFromMedium2D,
        Medium2D,
        *_get_all_subclasses(AbstractCustomMedium),
    )
    for model_class in dict.fromkeys(model_classes):
        model_class.model_rebuild(force=True, _types_namespace=types_namespace)


__all__ = ["rebuild_medium_models"]
