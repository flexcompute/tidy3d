"""Dipole emission study plugin.

The plugin evaluates angular radiation intensity from electric dipoles by
running reciprocal far-field probes generated from a source-free base
simulation. Most users should construct :class:`.DipoleEmissionStudy` rather
than the lower-level monitors used by the generated simulations.
"""

from __future__ import annotations

from tidy3d.plugins.dipole_emission.data import DipoleEmissionStudyData
from tidy3d.plugins.dipole_emission.data_array import (
    DipoleEmissionStudyDataArray,
    DipoleEmissionStudyPositionDataArray,
)
from tidy3d.plugins.dipole_emission.study import (
    DipoleEmissionStudy,
    EmissionAnalysisRegion,
)

__all__ = [
    "DipoleEmissionStudy",
    "DipoleEmissionStudyData",
    "DipoleEmissionStudyDataArray",
    "DipoleEmissionStudyPositionDataArray",
    "EmissionAnalysisRegion",
]
