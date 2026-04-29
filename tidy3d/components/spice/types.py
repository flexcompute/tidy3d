from __future__ import annotations

from tidy3d.components.spice.analysis.ac import IsothermalSSACAnalysis, SSACAnalysis
from tidy3d.components.spice.analysis.dc import (
    IsothermalSteadyChargeDCAnalysis,
    SteadyChargeDCAnalysis,
)

ElectricalAnalysisType = (
    SteadyChargeDCAnalysis
    | IsothermalSteadyChargeDCAnalysis
    | SSACAnalysis
    | IsothermalSSACAnalysis
)
