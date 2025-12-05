from __future__ import annotations

import warnings

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
    CoaxialLumpedResistor,
    LinearLumpedElement,
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
from tidy3d.components.monitor import DirectivityMonitor

# Source frame
from tidy3d.components.source.frame import PECFrame

# Subpixel spec
from tidy3d.components.subpixel_spec import SurfaceImpedance
from tidy3d.plugins.microwave import models
from tidy3d.plugins.microwave.array_factor import (
    BlackmanHarrisWindow,
    BlackmanWindow,
    ChebWindow,
    HammingWindow,
    HannWindow,
    KaiserWindow,
    RadialTaper,
    RectangularAntennaArrayCalculator,
    RectangularTaper,
    TaylorWindow,
)
from tidy3d.plugins.microwave.lobe_measurer import LobeMeasurer
from tidy3d.plugins.microwave.rf_material_library import rf_material_library
from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.terminal import (
    DirectivityMonitorSpec,
    ModelerLowFrequencySmoothingSpec,
    TerminalComponentModeler,
)
from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType
from tidy3d.plugins.smatrix.data.data_array import (
    PortDataArray,
    TerminalPortDataArray,
)
from tidy3d.plugins.smatrix.data.terminal import (
    MicrowaveSMatrixData,
    TerminalComponentModelerData,
)
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.ports.wave import WavePort

# Backwards compatibility
CurrentIntegralTypes = CurrentIntegralType
VoltageIntegralTypes = VoltageIntegralType
# Instantiate on plugin import till we unite with toplevel
warnings.filterwarnings(
    "once",
    message="RF simulations and functionality will require new license requirements in an upcoming release. All RF-specific classes are now available within the sub-package 'tidy3d.rf'.",
    category=FutureWarning,
)


__all__ = [
    "AbstractComponentModeler",
    "AdmittanceNetwork",
    "AntennaMetricsData",
    "AutoImpedanceSpec",
    "AxisAlignedCurrentIntegral",
    "AxisAlignedCurrentIntegralSpec",
    "AxisAlignedPathIntegral",
    "AxisAlignedVoltageIntegral",
    "AxisAlignedVoltageIntegralSpec",
    "BlackmanHarrisWindow",
    "BlackmanWindow",
    "ChebWindow",
    "CoaxialLumpedPort",
    "CoaxialLumpedResistor",
    "ComponentModelerDataType",
    "ComponentModelerType",
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
    "DirectivityMonitorSpec",
    "HammerstadSurfaceRoughness",
    "HammingWindow",
    "HannWindow",
    "HuraySurfaceRoughness",
    "ImpedanceCalculator",
    "InternalAbsorber",
    "KaiserWindow",
    "LayerRefinementSpec",
    "LinearLumpedElement",
    "LobeMeasurer",
    "LossyMetalMedium",
    "LowFrequencySmoothingSpec",
    "LumpedPort",
    "LumpedResistor",
    "MicrowaveModeData",
    "MicrowaveModeMonitor",
    "MicrowaveModeSolverData",
    "MicrowaveModeSolverMonitor",
    "MicrowaveModeSpec",
    "MicrowaveSMatrixData",
    "ModelerLowFrequencySmoothingSpec",
    "PECFrame",
    "PortDataArray",
    "RLCNetwork",
    "RadialTaper",
    "RectangularAntennaArrayCalculator",
    "RectangularLumpedElement",
    "RectangularTaper",
    "SurfaceImpedance",
    "SurfaceImpedanceFitterParam",
    "TaylorWindow",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "VoltageIntegralTypes",
    "WavePort",
    "models",
    "path_integrals_from_lumped_element",
    "rf_material_library",
]
