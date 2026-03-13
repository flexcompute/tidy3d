"""Imports from dispersion fitter plugin."""

from __future__ import annotations

from tidy3d.components.dispersion_fitter import AdvancedFastFitterParam

from .fit import DispersionFitter
from .fit_fast import FastDispersionFitter
from .web import AdvancedFitterParam, StableDispersionFitter

__all__ = [
    "AdvancedFastFitterParam",
    "AdvancedFitterParam",
    "DispersionFitter",
    "FastDispersionFitter",
    "StableDispersionFitter",
]
