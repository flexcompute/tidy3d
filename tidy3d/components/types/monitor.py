"""Type definitions for monitors."""

from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from tidy3d.components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
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
    PermittivityMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
)

# types of monitors that are accepted by simulation
MonitorType = Union[
    FieldMonitor,
    FieldTimeMonitor,
    AuxFieldTimeMonitor,
    MediumMonitor,
    PermittivityMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    DiffractionMonitor,
    DirectivityMonitor,
    MicrowaveModeMonitor,
    MicrowaveModeSolverMonitor,
    GaussianOverlapMonitor,
    AstigmaticGaussianOverlapMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
]


SurfaceMonitorType = Union[
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
]
