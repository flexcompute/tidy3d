"""RF and microwave classes retained in Tidy3D.

New RF development continues in Flexcompute RF; install ``flexcompute-rf`` and
import ``flexcompute.rf.tidy3d``. The classes still reachable here are the ones
defined under ``tidy3d.components``, also available from the top level where
they have a name there; they are deprecated and will be removed in Tidy3D 3.0.
The plugin-owned RF classes -- the antenna array calculators, the lobe
measurer, the RF material library, the microstrip models, and the terminal
component modeler with its ports and data -- are already gone; import them from
``flexcompute.rf.tidy3d``.
"""

from __future__ import annotations

from tidy3d._rf_migration import (
    missing_rf_attribute,
    relocated_rf_attribute,
    warn_rf_deprecated,
)

# Boundary
from tidy3d.components.boundary import InternalAbsorber

# Directivity monitor
from tidy3d.components.data.monitor_data import DirectivityData

# Frequency extrapolation
from tidy3d.components.frequency_extrapolation import LowFrequencySmoothingSpec

# Grid spec
from tidy3d.components.grid.grid_spec import CornerFinderSpec, LayerRefinementSpec

# Lumped elements
from tidy3d.components.lumped_element import (
    AdmittanceNetwork,
    CircuitImpedanceModel,
    CoaxialLumpedResistor,
    LinearLumpedElement,
    LumpedCircuitComponent,
    LumpedResistor,
    RectangularLumpedElement,
    RLCNetwork,
)

# Material
from tidy3d.components.medium import (
    HammerstadSurfaceRoughness,
    HuraySurfaceRoughness,
    LossyMetalMedium,
    SurfaceImpedanceFitterParam,
)

# Microwave data
from tidy3d.components.microwave.data.monitor_data import (
    AntennaMetricsData,
    MicrowaveModeData,
    MicrowaveModeSolverData,
)

# Impedance calculator
from tidy3d.components.microwave.impedance_calculator import (
    CurrentIntegralType,
    ImpedanceCalculator,
    VoltageIntegralType,
)

# Lumped port impedance specification
from tidy3d.components.microwave.impedance_spec import ImpedanceSpec

# Microwave mode spec
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec

# Microwave monitors
from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor

# Path integrals (actual integrals, not specs)
from tidy3d.components.microwave.path_integrals.integrals.auto import (
    path_integrals_from_lumped_element,
)
from tidy3d.components.microwave.path_integrals.integrals.base import (
    AxisAlignedPathIntegral,
    Custom2DPathIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.current import (
    AxisAlignedCurrentIntegral,
    CompositeCurrentIntegral,
    Custom2DCurrentIntegral,
)
from tidy3d.components.microwave.path_integrals.integrals.voltage import (
    AxisAlignedVoltageIntegral,
    Custom2DVoltageIntegral,
)

# Path integral specs
from tidy3d.components.microwave.path_integrals.specs.current import (
    AxisAlignedCurrentIntegralSpec,
    CompositeCurrentIntegralSpec,
    Custom2DCurrentIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    CustomImpedanceSpec,
)
from tidy3d.components.microwave.path_integrals.specs.voltage import (
    AxisAlignedVoltageIntegralSpec,
    Custom2DVoltageIntegralSpec,
)

# Microwave sources
from tidy3d.components.microwave.source import MicrowaveTerminalSource

# Baseband source times
from tidy3d.components.microwave.time import (
    BasebandCustomSourceTime,
    BasebandGaussianPulse,
    BasebandRectangularPulse,
    BasebandStep,
)
from tidy3d.components.monitor import DirectivityMonitor

# Source frame
from tidy3d.components.source.frame import PECFrame

# Subpixel spec
from tidy3d.components.subpixel_spec import SurfaceImpedance

# Backwards compatibility
CurrentIntegralTypes = CurrentIntegralType
VoltageIntegralTypes = VoltageIntegralType

# Shared photonics S-matrix classes that ``tidy3d.rf`` used to re-export. They
# stayed in Tidy3D with the modal modeler, so point at them rather than
# Flexcompute RF.
_RELOCATED_SMATRIX_NAMES = {
    "AbstractComponentModeler",
    "ComponentModelerDataType",
    "ComponentModelerType",
}

# Plugin-owned RF classes that Flexcompute RF now owns. They were exported from
# here before the move, so name them explicitly rather than letting access fail
# with a bare ``AttributeError``.
_MIGRATED_RF_NAMES = {
    "BlackmanHarrisWindow",
    "BlackmanWindow",
    "ChebWindow",
    "CoaxialLumpedPort",
    "DirectivityMonitorSpec",
    "HammingWindow",
    "HannWindow",
    "KaiserWindow",
    "LobeMeasurer",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "ModelerLowFrequencySmoothingSpec",
    "PortDataArray",
    "RadialTaper",
    "RectangularAntennaArrayCalculator",
    "RectangularTaper",
    "TaylorWindow",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "TerminalWavePort",
    "WavePort",
    "models",
    "rf_material_library",
}


def __getattr__(name: str) -> None:
    if name in _RELOCATED_SMATRIX_NAMES:
        relocated_rf_attribute(__name__, name, f"tidy3d.plugins.smatrix.{name}")
    if name in _MIGRATED_RF_NAMES:
        missing_rf_attribute(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


warn_rf_deprecated(__name__)

__all__ = [
    "AdmittanceNetwork",
    "AntennaMetricsData",
    "AutoImpedanceSpec",
    "AxisAlignedCurrentIntegral",
    "AxisAlignedCurrentIntegralSpec",
    "AxisAlignedPathIntegral",
    "AxisAlignedVoltageIntegral",
    "AxisAlignedVoltageIntegralSpec",
    "BasebandCustomSourceTime",
    "BasebandGaussianPulse",
    "BasebandRectangularPulse",
    "BasebandStep",
    "CircuitImpedanceModel",
    "CoaxialLumpedResistor",
    "CompositeCurrentIntegral",
    "CompositeCurrentIntegralSpec",
    "CornerFinderSpec",
    "CurrentIntegralTypes",
    "Custom2DCurrentIntegral",
    "Custom2DCurrentIntegralSpec",
    "Custom2DPathIntegral",
    "Custom2DVoltageIntegral",
    "Custom2DVoltageIntegralSpec",
    "CustomImpedanceSpec",
    "DirectivityData",
    "DirectivityMonitor",
    "HammerstadSurfaceRoughness",
    "HuraySurfaceRoughness",
    "ImpedanceCalculator",
    "ImpedanceSpec",
    "InternalAbsorber",
    "LayerRefinementSpec",
    "LinearLumpedElement",
    "LossyMetalMedium",
    "LowFrequencySmoothingSpec",
    "LumpedCircuitComponent",
    "LumpedResistor",
    "MicrowaveModeData",
    "MicrowaveModeMonitor",
    "MicrowaveModeSolverData",
    "MicrowaveModeSolverMonitor",
    "MicrowaveModeSpec",
    "MicrowaveTerminalSource",
    "PECFrame",
    "RLCNetwork",
    "RectangularLumpedElement",
    "SurfaceImpedance",
    "SurfaceImpedanceFitterParam",
    "VoltageIntegralTypes",
    "path_integrals_from_lumped_element",
]
