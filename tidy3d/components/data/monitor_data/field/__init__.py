"""Field monitor-data models and compatibility exports."""

from __future__ import annotations

from .data import FieldData
from .diagnostic import (
    FieldStructureData,
    FieldStructureTimeData,
    MediumData,
    ModeFieldStructureData,
    ModeStructureData,
    PermittivityData,
)
from .dipole import DipoleEmissionData
from .electromagnetic import ElectromagneticFieldData
from .point_cloud import PointCloudFieldData, PointCloudPermittivityData
from .surface import (
    ElectromagneticSurfaceFieldData,
    SurfaceFieldData,
    SurfaceFieldStructureData,
    SurfaceFieldTimeData,
)
from .time import AuxFieldTimeData, FieldTimeData

__all__ = [
    "AuxFieldTimeData",
    "DipoleEmissionData",
    "ElectromagneticFieldData",
    "ElectromagneticSurfaceFieldData",
    "FieldData",
    "FieldStructureData",
    "FieldStructureTimeData",
    "FieldTimeData",
    "MediumData",
    "ModeFieldStructureData",
    "ModeStructureData",
    "PermittivityData",
    "PointCloudFieldData",
    "PointCloudPermittivityData",
    "SurfaceFieldData",
    "SurfaceFieldStructureData",
    "SurfaceFieldTimeData",
]
