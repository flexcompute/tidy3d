"""EME monitor data"""

from __future__ import annotations

from typing import Union

import pydantic.v1 as pd

from ...base_sim.data.monitor_data import AbstractMonitorData
from ...data.monitor_data import ElectromagneticFieldData, ModeSolverData, PermittivityData
from ..monitor import EMECoefficientMonitor, EMEFieldMonitor, EMEModeSolverMonitor
from .dataset import EMECoefficientDataset, EMEFieldDataset, EMEModeSolverDataset


class EMEModeSolverData(ElectromagneticFieldData, EMEModeSolverDataset):
    """Data associated with an EME mode solver monitor."""

    monitor: EMEModeSolverMonitor = pd.Field(
        ...,
        title="EME Mode Solver Monitor",
        description="EME mode solver monitor associated with this data.",
    )


class EMEFieldData(ElectromagneticFieldData, EMEFieldDataset):
    """Data associated with an EME field monitor."""

    monitor: EMEFieldMonitor = pd.Field(
        ..., title="EME Field Monitor", description="EME field monitor associated with this data."
    )


class EMECoefficientData(AbstractMonitorData, EMECoefficientDataset):
    """Data associated with an EME coefficient monitor."""

    monitor: EMECoefficientMonitor = pd.Field(
        ...,
        title="EME Coefficient Monitor",
        description="EME coefficient monitor associated with this data.",
    )


EMEMonitorDataType = Union[
    EMEModeSolverData, EMEFieldData, EMECoefficientData, ModeSolverData, PermittivityData
]
