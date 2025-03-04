"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from typing import Union

from .current import CustomCurrentSource, PointDipole, UniformCurrentSource
from .field import (
    TFSF,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    GaussianBeam,
    ModeSource,
    PlaneWave,
)

# sources allowed in Simulation.sources
SourceType = Union[
    UniformCurrentSource,
    PointDipole,
    GaussianBeam,
    AstigmaticGaussianBeam,
    ModeSource,
    PlaneWave,
    CustomFieldSource,
    CustomCurrentSource,
    TFSF,
]
