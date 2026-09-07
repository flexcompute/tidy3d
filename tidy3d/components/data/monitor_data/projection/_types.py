from __future__ import annotations

from tidy3d.components.data.data_array import (
    DiffractionDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
)
from tidy3d.components.monitor import (
    DiffractionMonitor,
    DirectivityMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
)

ProjFieldType = (
    FieldProjectionAngleDataArray
    | FieldProjectionCartesianDataArray
    | FieldProjectionKSpaceDataArray
    | DiffractionDataArray
)

ProjMonitorType = (
    FieldProjectionAngleMonitor
    | FieldProjectionCartesianMonitor
    | FieldProjectionKSpaceMonitor
    | DiffractionMonitor
    | DirectivityMonitor
)
