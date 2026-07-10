"""Type definitions for monitors."""

from __future__ import annotations

from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from tidy3d.components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    DirectivityMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    GaussianOverlapMonitor,
    MediumMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    ModeTimeMonitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
    ThinLensOverlapMonitor,
)

# types of monitors that are accepted by simulation
MonitorType = (
    FieldMonitor
    | FieldTimeMonitor
    | AuxFieldTimeMonitor
    | MediumMonitor
    | PermittivityMonitor
    | FluxMonitor
    | FluxTimeMonitor
    | ModeMonitor
    | ModeSolverMonitor
    | ModeTimeMonitor
    | FieldProjectionAngleMonitor
    | FieldProjectionCartesianMonitor
    | FieldProjectionKSpaceMonitor
    | DiffractionMonitor
    | DirectivityMonitor
    | MicrowaveModeMonitor
    | MicrowaveModeSolverMonitor
    | GaussianOverlapMonitor
    | AstigmaticGaussianOverlapMonitor
    | ThinLensOverlapMonitor
    | DipoleEmissionMonitor
    | PointCloudFieldMonitor
    | PointCloudPermittivityMonitor
    | SurfaceFieldMonitor
    | SurfaceFieldTimeMonitor
)


SurfaceMonitorType = SurfaceFieldMonitor | SurfaceFieldTimeMonitor
