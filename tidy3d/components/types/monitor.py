"""Type definitions for monitors."""

from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.monitor import MicrowaveModeMonitor, MicrowaveModeSolverMonitor
from tidy3d.components.monitor import (
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
    MediumMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    PermittivityMonitor,
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
]
