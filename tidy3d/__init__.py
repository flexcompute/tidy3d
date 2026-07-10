"""Tidy3d package imports"""

from __future__ import annotations

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.boundary import BroadbandModeABCFitterParam, BroadbandModeABCSpec
from tidy3d.components.data.index import SimulationDataMap
from tidy3d.components.frequency_extrapolation import LowFrequencySmoothingSpec
from tidy3d.components.index import SimulationMap
from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.material.tcad.charge import (
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
    AnisotropicConductivity,
    FluidMedium,
    FluidSpec,
    SolidMedium,
    SolidSpec,
)
from tidy3d.components.microwave.data.monitor_data import (
    AntennaMetricsData,
    MicrowaveModeData,
    MicrowaveModeSolverData,
)
from tidy3d.components.microwave.impedance_calculator import ImpedanceCalculator
from tidy3d.components.microwave.mode_spec import (
    MicrowaveModeSpec,
)
from tidy3d.components.microwave.monitor import (
    MicrowaveModeMonitor,
    MicrowaveModeSolverMonitor,
)
from tidy3d.components.microwave.path_integrals.integrals.auto import (
    path_integrals_from_lumped_element,
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
from tidy3d.components.spice.analysis.ac import IsothermalSSACAnalysis, SSACAnalysis
from tidy3d.components.spice.analysis.dc import (
    ChargeToleranceSpec,
    IsothermalSteadyChargeDCAnalysis,
    SteadyChargeDCAnalysis,
)
from tidy3d.components.spice.sources.ac import SSACVoltageSource
from tidy3d.components.spice.sources.dc import DCCurrentSource, DCVoltageSource, GroundVoltage
from tidy3d.components.spice.sources.types import VoltageSourceType
from tidy3d.components.tcad.analysis.heat_simulation_type import UnsteadyHeatAnalysis, UnsteadySpec
from tidy3d.components.tcad.boundary.heat import VerticalNaturalConvectionCoeffModel
from tidy3d.components.tcad.boundary.specification import (
    HeatBoundarySpec,
    HeatChargeBoundarySpec,
)
from tidy3d.components.tcad.data.monitor_data.mesh import VolumeMeshData
from tidy3d.components.tcad.data.sim_data import (
    DeviceCharacteristics,
    HeatChargeSimulationData,
    HeatSimulationData,
    SteadyConvergenceData,
    VolumeMesherData,
)
from tidy3d.components.tcad.data.types import (
    SteadyCapacitanceData,
    SteadyChargeResidualData,
    SteadyCurrentDensityData,
    SteadyElectricFieldData,
    SteadyEnergyBandData,
    SteadyFreeCarrierData,
    SteadyPotentialData,
    TemperatureData,
)
from tidy3d.components.tcad.doping import ConstantDoping, CustomDoping, GaussianDoping
from tidy3d.components.tcad.generation_recombination import (
    FossumCarrierLifetime,
    PalankovskiQuayApproxCarrierLifetime,
    SurfaceShockleyReedHallRecombination,
)
from tidy3d.components.tcad.grid import (
    DistanceUnstructuredGrid,
    GridRefinementLine,
    GridRefinementRegion,
    UniformUnstructuredGrid,
)
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyChargeResidualMonitor,
    SteadyCurrentDensityMonitor,
    SteadyElectricFieldMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.monitors.mesh import VolumeMeshMonitor
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.tcad.types import (
    AugerRecombination,
    CaugheyThomasMobility,
    ConstantEffectiveDOS,
    ConstantEnergyBandGap,
    ConstantMobilityModel,
    ConvectionBC,
    CurrentBC,
    DistributedGeneration,
    DualValleyEffectiveDOS,
    HeatFluxBC,
    HeatFromElectricSource,
    HeatSource,
    HurkxDirectBandToBandTunneling,
    InsulatingBC,
    IsotropicEffectiveDOS,
    MasettiMobility,
    MultiValleyEffectiveDOS,
    RadiationBC,
    RadiativeRecombination,
    SelberherrImpactIonization,
    ShockleyReedHallRecombination,
    SlotboomBandGapNarrowing,
    SurfaceRecombinationBC,
    TemperatureBC,
    ThermalContactResistance,
    UniformHeatSource,
    VarshniEnergyBandGap,
    VoltageBC,
)

from .components.apodization import ApodizationSpec

# boundary placement for other solvers
from .components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)

# analytic beams
from .components.beam import (
    AstigmaticGaussianBeamProfile,
    GaussianBeamProfile,
    PlaneWaveBeamProfile,
    ThinLensProfile,
)

# boundary
from .components.boundary import (
    PML,
    ABCBoundary,
    Absorber,
    AbsorberParams,
    BlochBoundary,
    Boundary,
    BoundaryEdge,
    BoundaryEdgeType,
    BoundarySpec,
    DefaultAbsorberParameters,
    DefaultPMLParameters,
    DefaultStablePMLParameters,
    InternalAbsorber,
    ModeABCBoundary,
    PECBoundary,
    Periodic,
    PMCBoundary,
    PMLParams,
    PMLTypes,
    StablePML,
)

# data
from .components.data.data_array import (
    CellDataArray,
    ChargeDataArray,
    ConvergenceHistoryDataArray,
    DiffractionDataArray,
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
    EMECoefficientDataArray,
    EMEFluxDataArray,
    EMEInterfaceCellIndexDataArray,
    EMEInterfaceDiagnosticDataArray,
    EMEInterfaceSMatrixDataArray,
    EMEModeIndexDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    EMETraceMetricDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    FreqDataArray,
    FreqModeDataArray,
    GroupIndexDataArray,
    HeatDataArray,
    IndexedDataArray,
    IndexedFieldDataArray,
    IndexedFieldTimeDataArray,
    IndexedFieldVoltageDataArray,
    IndexedFreqDataArray,
    IndexedSurfaceFieldDataArray,
    IndexedSurfaceFieldTimeDataArray,
    IndexedSurfaceFreqDataArray,
    IndexedSurfaceTimeDataArray,
    IndexedTimeDataArray,
    IndexedVoltageDataArray,
    ModeAmpsDataArray,
    ModeAmpsTimeDataArray,
    ModeIndexDataArray,
    PointDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
    SpatialDataArray,
    SpatialVoltageDataArray,
    SphericalAngleDataArray,
    SteadyVoltageDataArray,
)
from .components.data.dataset import (
    FieldDataset,
    FieldTimeDataset,
    MediumDataset,
    ModeSolverDataset,
    PermittivityDataset,
    PointCloudFieldDataset,
    PointCloudPermittivityDataset,
)
from .components.data.monitor_data import (
    AbstractFieldProjectionData,
    AuxFieldTimeData,
    DiffractionData,
    DipoleEmissionData,
    DirectivityData,
    FieldData,
    FieldOverlapData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
    FieldTimeData,
    FluxData,
    FluxTimeData,
    MediumData,
    ModeData,
    ModeSolverData,
    ModeTimeData,
    PermittivityData,
    PointCloudFieldData,
    PointCloudPermittivityData,
    SurfaceFieldData,
    SurfaceFieldTimeData,
)
from .components.data.sim_data import DATA_TYPE_MAP, SimulationData
from .components.data.unstructured.surface import TriangularSurfaceDataset
from .components.data.utils import (
    TetrahedralGridDataset,
    TriangularGridDataset,
)
from .components.eme.data.dataset import (
    EMECoefficientDataset,
    EMEDiagnosticsData,
    EMEFieldDataset,
    EMEInterfaceDiagnostics,
    EMEInterfaceSMatrixDataset,
    EMEModeSolverDataset,
    EMEOverlapDataset,
    EMESMatrixDataset,
)
from .components.eme.data.monitor_data import EMECoefficientData, EMEFieldData, EMEModeSolverData
from .components.eme.data.sim_data import EMESimulationData
from .components.eme.data.stage import (
    EMEStageCellModes,
    EMEStageCellOverlap,
    EMEStageCellSMatrix,
    EMEStageInterfaceDiagnostics,
    EMEStageInterfaceOverlap,
    EMEStageInterfaceSMatrix,
)
from .components.eme.grid import (
    EMECompositeGrid,
    EMEExplicitGrid,
    EMEGrid,
    EMEModeSpec,
    EMEUniformGrid,
)
from .components.eme.monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
    EMEMonitor,
)

# EME
from .components.eme.simulation import EMESimulation
from .components.eme.sweep import EMEFreqSweep, EMELengthSweep, EMEModeSweep, EMEPeriodicitySweep

# field projection
from .components.field_projection import FieldProjector

# frequency conversion utilities
from .components.frequencies import FreqRange, FrequencyUtils, frequencies, wavelengths

# geometry
from .components.geometry.base import (
    Box,
    ClipOperation,
    Geometry,
    GeometryArray,
    GeometryGroup,
    Transformed,
)
from .components.geometry.mesh import TriangleMesh
from .components.geometry.polyslab import PolySlab
from .components.geometry.primitives import Cylinder, Sphere
from .components.grid.corner_finder import CornerFinderSpec
from .components.grid.grid import Coords, Coords1D, FieldGrid, Grid, YeeGrid
from .components.grid.grid_spec import (
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridRefinement,
    GridSpec,
    LayerRefinementSpec,
    QuasiUniformGrid,
    UniformGrid,
)

# lumped elements
from .components.lumped_element import (
    AdmittanceNetwork,
    CircuitImpedanceModel,
    CoaxialLumpedResistor,
    LinearLumpedElement,
    LumpedCircuitComponent,
    LumpedElement,
    LumpedResistor,
    RectangularLumpedElement,
    RLCNetwork,
)

# medium
# for docs
from .components.medium import (
    PEC,
    PEC2D,
    PMC,
    AbstractMedium,
    AnisotropicMedium,
    CustomAnisotropicMedium,
    CustomDebye,
    CustomDrude,
    CustomLorentz,
    CustomMedium,
    CustomPoleResidue,
    CustomSellmeier,
    Debye,
    Drude,
    FullyAnisotropicMedium,
    HammerstadSurfaceRoughness,
    HuraySurfaceRoughness,
    Lorentz,
    LossyMetalMedium,
    Medium,
    Medium2D,
    PECMedium,
    PerturbationMedium,
    PerturbationPoleResidue,
    PMCMedium,
    PoleResidue,
    Sellmeier,
    SurfaceImpedanceFitterParam,
    medium_from_nk,
)

# sources
from .components.microwave.time import (
    BasebandCustomSourceTime,
    BasebandGaussianPulse,
    BasebandRectangularPulse,
    BasebandStep,
)
from .components.mode.data.sim_data import ModeSimulationData

# Mode
from .components.mode.simulation import ModeSimulation

# modes
from .components.mode_spec import (
    ChebSampling,
    CustomSampling,
    ModeInterpSpec,
    ModeSortSpec,
    ModeSpec,
    UniformSampling,
)

# monitors
from .components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    DirectivityMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    GaussianOverlapMonitor,
    MediumMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    ModeTimeMonitor,
    Monitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
    ThinLensOverlapMonitor,
)

# nonlinear
from .components.nonlinear import (
    KerrNonlinearity,
    NonlinearModel,
    NonlinearSpec,
    NonlinearSusceptibility,
    TwoPhotonAbsorption,
)
from .components.parameter_perturbation import (
    CustomChargePerturbation,
    CustomHeatPerturbation,
    IndexPerturbation,
    LinearChargePerturbation,
    LinearHeatPerturbation,
    NedeljkovicSorefMashanovich,
    ParameterPerturbation,
    PermittivityPerturbation,
)

# run time spec
from .components.run_time_spec import RunTimeSpec

# scene
from .components.scene import Scene

# simulation
from .components.simulation import Simulation
from .components.source.base import Source
from .components.source.current import (
    CustomCurrentSource,
    PointDipole,
    UniformCurrentSource,
)
from .components.source.field import (
    TFSF,
    AstigmaticGaussianBeam,
    CustomFieldSource,
    FixedAngleSpec,
    FixedInPlaneKSpec,
    GaussianBeam,
    ModeSource,
    PlaneWave,
    ThinLensBeam,
)
from .components.source.frame import (
    PECFrame,
)
from .components.source.time import (
    BroadbandPulse,
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    SourceTime,
)

# structures
from .components.structure import MeshOverrideStructure, Structure

# subpixel
from .components.subpixel_spec import (
    ContourPathAveraging,
    HeuristicPECStaircasing,
    PECConformal,
    PolarizedAveraging,
    Staircasing,
    SubpixelSpec,
    SurfaceImpedance,
    VolumetricAveraging,
)

# time modulation
from .components.time_modulation import (
    ContinuousWaveTimeModulation,
    ModulationSpec,
    SpaceModulation,
    SpaceTimeModulation,
)
from .components.transformation import RotationAroundAxis
from .components.viz import VisualizationSpec, restore_matplotlib_rcparams

# config
from .config import config

# constants imported as `C_0 = td.C_0` or `td.constants.C_0`
from .constants import C_0, EPSILON_0, ETA_0, HBAR, K_B, MU_0, Q_e, inf
from .log import log, set_logging_console, set_logging_file

# material library dict imported as `from tidy3d import material_library`
# get material `mat` and variant `var` as `material_library[mat][var]`
from .material_library.material_library import material_library
from .material_library.parametric_materials import Graphene

# lumped port impedance specification
from .plugins.smatrix.ports.base_lumped import ImpedanceSpec

# updater
from .updater import Updater

# version
from .version import __version__


def set_logging_level(level: str) -> None:
    """Raise a warning here instead of setting the logging level."""
    raise DeprecationWarning(
        "``set_logging_level`` no longer supported. "
        f"To set the logging level, call ``tidy3d.config.logging.level = {level}``."
    )


log.info(f"Using client version: {__version__}")

Transformed.model_rebuild()
ClipOperation.model_rebuild()
GeometryGroup.model_rebuild()
GeometryArray.model_rebuild()

# Backwards compatibility: Remove 2.11 renamed integral classes
VoltageIntegralAxisAligned = AxisAlignedVoltageIntegral
CurrentIntegralAxisAligned = AxisAlignedCurrentIntegral
CustomVoltageIntegral2D = Custom2DVoltageIntegral
CustomCurrentIntegral2D = Custom2DCurrentIntegral

__all__ = [
    "C_0",
    "DATA_TYPE_MAP",
    "EPSILON_0",
    "ETA_0",
    "HBAR",
    "K_B",
    "MU_0",
    "PEC",
    "PEC2D",
    "PMC",
    "PML",
    "TFSF",
    "ABCBoundary",
    "Absorber",
    "AbsorberParams",
    "AbstractFieldProjectionData",
    "AbstractMedium",
    "AdmittanceNetwork",
    "AnisotropicConductivity",
    "AnisotropicMedium",
    "AntennaMetricsData",
    "ApodizationSpec",
    "AstigmaticGaussianBeam",
    "AstigmaticGaussianBeamProfile",
    "AstigmaticGaussianOverlapMonitor",
    "AugerRecombination",
    "AutoGrid",
    "AutoImpedanceSpec",
    "AuxFieldTimeData",
    "AuxFieldTimeMonitor",
    "AxisAlignedCurrentIntegral",
    "AxisAlignedCurrentIntegralSpec",
    "AxisAlignedVoltageIntegral",
    "AxisAlignedVoltageIntegralSpec",
    "BasebandCustomSourceTime",
    "BasebandGaussianPulse",
    "BasebandRectangularPulse",
    "BasebandStep",
    "BlochBoundary",
    "Boundary",
    "BoundaryEdge",
    "BoundaryEdgeType",
    "BoundarySpec",
    "Box",
    "BroadbandModeABCFitterParam",
    "BroadbandModeABCSpec",
    "BroadbandPulse",
    "CaugheyThomasMobility",
    "CellDataArray",
    "ChargeConductorMedium",
    "ChargeDataArray",
    "ChargeInsulatorMedium",
    "ChargeToleranceSpec",
    "ChebSampling",
    "CircuitImpedanceModel",
    "ClipOperation",
    "CoaxialLumpedResistor",
    "CompositeCurrentIntegral",
    "CompositeCurrentIntegralSpec",
    "ConstantDoping",
    "ConstantEffectiveDOS",
    "ConstantEnergyBandGap",
    "ConstantMobilityModel",
    "ContinuousWave",
    "ContinuousWaveTimeModulation",
    "ContourPathAveraging",
    "ConvectionBC",
    "ConvergenceHistoryDataArray",
    "Coords",
    "Coords1D",
    "CornerFinderSpec",
    "CurrentBC",
    "CurrentIntegralAxisAligned",  # Backwards compatibility alias
    "Custom2DCurrentIntegral",
    "Custom2DCurrentIntegralSpec",
    "Custom2DVoltageIntegral",
    "Custom2DVoltageIntegralSpec",
    "CustomAnisotropicMedium",
    "CustomChargePerturbation",
    "CustomCurrentIntegral2D",  # Backwards compatibility alias
    "CustomCurrentSource",
    "CustomDebye",
    "CustomDoping",
    "CustomDrude",
    "CustomFieldSource",
    "CustomGrid",
    "CustomGridBoundaries",
    "CustomHeatPerturbation",
    "CustomImpedanceSpec",
    "CustomLorentz",
    "CustomMedium",
    "CustomPoleResidue",
    "CustomSampling",
    "CustomSellmeier",
    "CustomSourceTime",
    "CustomVoltageIntegral2D",  # Backwards compatibility alias
    "Cylinder",
    "DCCurrentSource",
    "DCVoltageSource",
    "Debye",
    "DefaultAbsorberParameters",
    "DefaultPMLParameters",
    "DefaultStablePMLParameters",
    "DeviceCharacteristics",
    "DiffractionData",
    "DiffractionDataArray",
    "DiffractionMonitor",
    "DipoleEmissionData",
    "DipoleEmissionDataArray",
    "DipoleEmissionMonitor",
    "DipoleEmissionPositionDataArray",
    "DirectivityData",
    "DirectivityMonitor",
    "DistanceUnstructuredGrid",
    "DistributedGeneration",
    "Drude",
    "DualValleyEffectiveDOS",
    "EMECoefficientData",
    "EMECoefficientDataArray",
    "EMECoefficientDataset",
    "EMECoefficientMonitor",
    "EMECompositeGrid",
    "EMEDiagnosticsData",
    "EMEExplicitGrid",
    "EMEFieldData",
    "EMEFieldDataset",
    "EMEFieldMonitor",
    "EMEFluxDataArray",
    "EMEFreqSweep",
    "EMEGrid",
    "EMEInterfaceCellIndexDataArray",
    "EMEInterfaceDiagnosticDataArray",
    "EMEInterfaceDiagnostics",
    "EMEInterfaceSMatrixDataArray",
    "EMEInterfaceSMatrixDataset",
    "EMELengthSweep",
    "EMEModeIndexDataArray",
    "EMEModeSolverData",
    "EMEModeSolverDataset",
    "EMEModeSolverMonitor",
    "EMEModeSpec",
    "EMEModeSweep",
    "EMEMonitor",
    "EMEOverlapDataset",
    "EMEPeriodicitySweep",
    "EMESMatrixDataArray",
    "EMESMatrixDataset",
    "EMEScalarFieldDataArray",
    "EMEScalarModeFieldDataArray",
    "EMESimulation",
    "EMESimulationData",
    "EMEStageCellModes",
    "EMEStageCellOverlap",
    "EMEStageCellSMatrix",
    "EMEStageInterfaceDiagnostics",
    "EMEStageInterfaceOverlap",
    "EMEStageInterfaceSMatrix",
    "EMETraceMetricDataArray",
    "EMEUniformGrid",
    "FieldData",
    "FieldDataset",
    "FieldGrid",
    "FieldMonitor",
    "FieldOverlapData",
    "FieldProjectionAngleData",
    "FieldProjectionAngleDataArray",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianData",
    "FieldProjectionCartesianDataArray",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceData",
    "FieldProjectionKSpaceDataArray",
    "FieldProjectionKSpaceMonitor",
    "FieldProjectionSurface",
    "FieldProjector",
    "FieldTimeData",
    "FieldTimeDataset",
    "FieldTimeMonitor",
    "FixedAngleSpec",
    "FixedInPlaneKSpec",
    "FluidMedium",
    "FluidSpec",
    "FluxData",
    "FluxDataArray",
    "FluxMonitor",
    "FluxTimeData",
    "FluxTimeDataArray",
    "FluxTimeMonitor",
    "FossumCarrierLifetime",
    "FreqDataArray",
    "FreqModeDataArray",
    "FreqRange",
    "FrequencyUtils",
    "FullyAnisotropicMedium",
    "GaussianBeam",
    "GaussianBeamProfile",
    "GaussianDoping",
    "GaussianOverlapMonitor",
    "GaussianPulse",
    "Geometry",
    "GeometryArray",
    "GeometryGroup",
    "Graphene",
    "Grid",
    "GridRefinement",
    "GridRefinementLine",
    "GridRefinementRegion",
    "GridSpec",
    "GroundVoltage",
    "GroupIndexDataArray",
    "HammerstadSurfaceRoughness",
    "HeatBoundarySpec",
    "HeatChargeBoundarySpec",
    "HeatChargeSimulation",
    "HeatChargeSimulationData",
    "HeatDataArray",
    "HeatFluxBC",
    "HeatFromElectricSource",
    "HeatSimulation",
    "HeatSimulationData",
    "HeatSource",
    "HeuristicPECStaircasing",
    "HuraySurfaceRoughness",
    "HurkxDirectBandToBandTunneling",
    "ImpedanceCalculator",
    "ImpedanceSpec",
    "IndexPerturbation",
    "IndexedDataArray",
    "IndexedFieldDataArray",
    "IndexedFieldTimeDataArray",
    "IndexedFieldVoltageDataArray",
    "IndexedFreqDataArray",
    "IndexedSurfaceFieldDataArray",
    "IndexedSurfaceFieldTimeDataArray",
    "IndexedSurfaceFreqDataArray",
    "IndexedSurfaceTimeDataArray",
    "IndexedTimeDataArray",
    "IndexedVoltageDataArray",
    "InsulatingBC",
    "InternalAbsorber",
    "IsothermalSSACAnalysis",
    "IsothermalSteadyChargeDCAnalysis",
    "IsotropicEffectiveDOS",
    "KerrNonlinearity",
    "LayerRefinementSpec",
    "LinearChargePerturbation",
    "LinearHeatPerturbation",
    "LinearLumpedElement",
    "Lorentz",
    "LossyMetalMedium",
    "LowFrequencySmoothingSpec",
    "LumpedCircuitComponent",
    "LumpedElement",
    "LumpedResistor",
    "MasettiMobility",
    "Medium",
    "Medium2D",
    "MediumData",
    "MediumDataset",
    "MediumMediumInterface",
    "MediumMonitor",
    "MeshOverrideStructure",
    "MicrowaveModeData",
    "MicrowaveModeMonitor",
    "MicrowaveModeSolverData",
    "MicrowaveModeSolverMonitor",
    "MicrowaveModeSpec",
    "ModeABCBoundary",
    "ModeAmpsDataArray",
    "ModeAmpsTimeDataArray",
    "ModeData",
    "ModeIndexDataArray",
    "ModeInterpSpec",
    "ModeMonitor",
    "ModeSimulation",
    "ModeSimulationData",
    "ModeSolverData",
    "ModeSolverDataset",
    "ModeSolverMonitor",
    "ModeSortSpec",
    "ModeSource",
    "ModeSpec",
    "ModeTimeData",
    "ModeTimeMonitor",
    "ModulationSpec",
    "Monitor",
    "MultiPhysicsMedium",
    "MultiValleyEffectiveDOS",
    "NedeljkovicSorefMashanovich",
    "NonlinearModel",
    "NonlinearSpec",
    "NonlinearSusceptibility",
    "PECBoundary",
    "PECConformal",
    "PECFrame",
    "PECMedium",
    "PMCBoundary",
    "PMCMedium",
    "PMLParams",
    "PMLTypes",
    "PalankovskiQuayApproxCarrierLifetime",
    "ParameterPerturbation",
    "Periodic",
    "PermittivityData",
    "PermittivityDataset",
    "PermittivityMonitor",
    "PermittivityPerturbation",
    "PerturbationMedium",
    "PerturbationPoleResidue",
    "PlaneWave",
    "PlaneWaveBeamProfile",
    "PointCloudFieldData",
    "PointCloudFieldDataset",
    "PointCloudFieldMonitor",
    "PointCloudPermittivityData",
    "PointCloudPermittivityDataset",
    "PointCloudPermittivityMonitor",
    "PointDataArray",
    "PointDipole",
    "PolarizedAveraging",
    "PoleResidue",
    "PolySlab",
    "Q_e",
    "QuasiUniformGrid",
    "RLCNetwork",
    "RadiationBC",
    "RadiativeRecombination",
    "RectangularLumpedElement",
    "RotationAroundAxis",
    "RunTimeSpec",
    "SSACAnalysis",
    "SSACVoltageSource",
    "ScalarFieldDataArray",
    "ScalarFieldTimeDataArray",
    "ScalarModeFieldCylindricalDataArray",
    "ScalarModeFieldDataArray",
    "Scene",
    "SelberherrImpactIonization",
    "Sellmeier",
    "SemiconductorMedium",
    "ShockleyReedHallRecombination",
    "Simulation",
    "SimulationBoundary",
    "SimulationData",
    "SimulationDataMap",
    "SimulationMap",
    "SlotboomBandGapNarrowing",
    "SolidMedium",
    "SolidSpec",
    "Source",
    "SourceTime",
    "SpaceModulation",
    "SpaceTimeModulation",
    "SpatialDataArray",
    "SpatialVoltageDataArray",
    "Sphere",
    "SphericalAngleDataArray",
    "StablePML",
    "Staircasing",
    "SteadyCapacitanceData",
    "SteadyCapacitanceMonitor",
    "SteadyChargeDCAnalysis",
    "SteadyChargeResidualData",
    "SteadyChargeResidualMonitor",
    "SteadyConvergenceData",
    "SteadyCurrentDensityData",
    "SteadyCurrentDensityMonitor",
    "SteadyElectricFieldData",
    "SteadyElectricFieldMonitor",
    "SteadyEnergyBandData",
    "SteadyEnergyBandMonitor",
    "SteadyFreeCarrierData",
    "SteadyFreeCarrierMonitor",
    "SteadyPotentialData",
    "SteadyPotentialMonitor",
    "SteadyVoltageDataArray",
    "Structure",
    "StructureBoundary",
    "StructureSimulationBoundary",
    "StructureStructureInterface",
    "SubpixelSpec",
    "SurfaceFieldData",
    "SurfaceFieldMonitor",
    "SurfaceFieldTimeData",
    "SurfaceFieldTimeMonitor",
    "SurfaceImpedance",
    "SurfaceImpedanceFitterParam",
    "SurfaceRecombinationBC",
    "SurfaceShockleyReedHallRecombination",
    "TemperatureBC",
    "TemperatureData",
    "TemperatureMonitor",
    "TetrahedralGridDataset",
    "ThermalContactResistance",
    "ThinLensBeam",
    "ThinLensOverlapMonitor",
    "ThinLensProfile",
    "Tidy3dBaseModel",
    "Transformed",
    "TriangleMesh",
    "TriangularGridDataset",
    "TriangularSurfaceDataset",
    "TwoPhotonAbsorption",
    "UniformCurrentSource",
    "UniformGrid",
    "UniformHeatSource",
    "UniformSampling",
    "UniformUnstructuredGrid",
    "UnsteadyHeatAnalysis",
    "UnsteadySpec",
    "Updater",
    "VarshniEnergyBandGap",
    "VerticalNaturalConvectionCoeffModel",
    "VisualizationSpec",
    "VoltageBC",
    "VoltageIntegralAxisAligned",  # Backwards compatibility alias
    "VoltageSourceType",
    "VolumeMeshData",
    "VolumeMeshMonitor",
    "VolumeMesher",
    "VolumeMesherData",
    "VolumetricAveraging",
    "YeeGrid",
    "__version__",
    "config",
    "frequencies",
    "inf",
    "log",
    "material_library",
    "medium_from_nk",
    "path_integrals_from_lumped_element",
    "restore_matplotlib_rcparams",
    "set_logging_console",
    "set_logging_file",
    "wavelengths",
]
