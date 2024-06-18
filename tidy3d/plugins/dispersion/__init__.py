"""Imports from dispersion fitter plugin."""

from ...components.fitter.fit import DispersionFitter
from ...components.fitter.fit_fast import AdvancedFastFitterParam, FastDispersionFitter
from ...log import log
from .web import AdvancedFitterParam, StableDispersionFitter

log.warning(
    "Dispersive material fitter in the module 'plugins.dispersion' has "
    "been moved to Tidy3D module. 'plugins.dispersion' will be removed in future versions."
)

__all__ = [
    "DispersionFitter",
    "AdvancedFitterParam",
    "StableDispersionFitter",
    "FastDispersionFitter",
    "AdvancedFastFitterParam",
]
