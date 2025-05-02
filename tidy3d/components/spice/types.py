from __future__ import annotations

from typing import Union

from tidy3d.components.spice.analysis.dc import (
    IsothermalSteadyChargeDCAnalysis,
    SteadyChargeDCAnalysis,
)

ElectricalAnalysisType = Union[SteadyChargeDCAnalysis, IsothermalSteadyChargeDCAnalysis]
