"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from tidy3d.components.microwave.source import MicrowaveTerminalSource

from .current import CustomCurrentSource, PointDipole, UniformCurrentSource
from .field import (
    TFSF,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    GaussianBeam,
    ModeSource,
    PlaneWave,
    ThinLensBeam,
)

# Gaussian-like beam sources.
GaussianBeamType = GaussianBeam | AstigmaticGaussianBeam | ThinLensBeam

# sources allowed in Simulation.sources
SourceType = (
    UniformCurrentSource
    | PointDipole
    | GaussianBeam
    | AstigmaticGaussianBeam
    | ThinLensBeam
    | ModeSource
    | PlaneWave
    | CustomFieldSource
    | CustomCurrentSource
    | TFSF
    | MicrowaveTerminalSource
)
