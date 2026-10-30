"""Type definitions for monitor data."""

from __future__ import annotations

from tidy3d.components.data.monitor_data import (
    AuxFieldTimeData,
    DiffractionData,
    DipoleEmissionData,
    DirectivityData,
    FieldData,
    FieldOverlapData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
    FieldStructureData,
    FieldStructureTimeData,
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
    SurfaceFieldStructureData,
    SurfaceFieldTimeData,
)
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData, MicrowaveModeSolverData

# Type aliases
ModeDataType = ModeData | MicrowaveModeData
ModeSolverDataType = ModeSolverData | MicrowaveModeSolverData
MonitorDataTypes = (
    FieldData,
    FieldStructureData,
    FieldStructureTimeData,
    FieldTimeData,
    PermittivityData,
    MediumData,
    ModeSolverData,
    ModeData,
    ModeTimeData,
    FluxData,
    FluxTimeData,
    AuxFieldTimeData,
    FieldProjectionKSpaceData,
    FieldProjectionCartesianData,
    FieldProjectionAngleData,
    DiffractionData,
    DipoleEmissionData,
    DirectivityData,
    FieldOverlapData,
    MicrowaveModeData,
    MicrowaveModeSolverData,
    PointCloudFieldData,
    PointCloudPermittivityData,
    SurfaceFieldData,
    SurfaceFieldStructureData,
    SurfaceFieldTimeData,
)
MonitorDataType = (
    FieldData
    | FieldStructureData
    | FieldStructureTimeData
    | FieldTimeData
    | PermittivityData
    | MediumData
    | ModeSolverData
    | ModeData
    | ModeTimeData
    | FluxData
    | FluxTimeData
    | AuxFieldTimeData
    | FieldProjectionKSpaceData
    | FieldProjectionCartesianData
    | FieldProjectionAngleData
    | DiffractionData
    | DipoleEmissionData
    | DirectivityData
    | FieldOverlapData
    | MicrowaveModeData
    | MicrowaveModeSolverData
    | PointCloudFieldData
    | PointCloudPermittivityData
    | SurfaceFieldData
    | SurfaceFieldStructureData
    | SurfaceFieldTimeData
)
