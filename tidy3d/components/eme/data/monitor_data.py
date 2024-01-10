"""EME monitor data"""
from __future__ import annotations

from abc import ABC

from typing import Union, List

import pydantic.v1 as pd

from ...base_sim.data.monitor_data import AbstractMonitorData
from ..monitor import EMEModeSolverMonitor, EMEFieldMonitor, EMECoefficientMonitor
from ...data.monitor_data import ModeSolverData, ElectromagneticFieldData
from ...base import Tidy3dBaseModel
from ....exceptions import ValidationError

from .dataset import EMEFieldDataset, EMECoefficientDataset


class EMEGridData(Tidy3dBaseModel, ABC):
    """Abstract class defining data indexed by a subset of the cells in the EME grid."""

    cell_indices: List[pd.NonNegativeInt] = pd.Field(
        ..., title="Cell indices", description="Cell indices"
    )


class EMEModeSolverData(AbstractMonitorData, EMEGridData):
    """Data associated with an EME mode solver monitor."""

    monitor: EMEModeSolverMonitor = pd.Field(
        ...,
        title="EME Mode Solver Monitor",
        description="EME mode solver monitor associated with this data.",
    )

    modes: List[ModeSolverData] = pd.Field(
        ...,
        title="Modes",
        description="Modes recorded by the EME mode solver monitor. "
        "The corresponding cell indices are stored in 'cell_indices'. "
        "A mode is recorded if its mode plane is contained in the monitor geometry.",
    )

    @pd.validator("modes", always=True)
    def _validate_num_modes(cls, val, values):
        """Check that the number of modes equals the number of cells inside the monitor."""
        num_cells = len(values["cell_indices"])
        num_modes = len(val)
        if num_cells != num_modes:
            raise ValidationError("The number of 'modes' must equal the number of 'cell_indices'.")
        return val


class EMEFieldData(EMEFieldDataset, ElectromagneticFieldData):
    """Data associated with an EME field monitor."""

    monitor: EMEFieldMonitor = pd.Field(
        ..., title="EME Field Monitor", description="EME field monitor associated with this data."
    )


class EMECoefficientData(AbstractMonitorData, EMEGridData):
    """Data associated with an EME coefficient monitor."""

    monitor: EMECoefficientMonitor = pd.Field(
        ...,
        title="EME Coefficient Monitor",
        description="EME coefficient monitor associated with this data.",
    )

    coeffs: List[EMECoefficientDataset] = pd.Field(
        ...,
        title="Coefficients",
        description="Coefficients of the forward and backward traveling modes in each cell "
        "contained in the monitor geometry.",
    )

    @pd.validator("coeffs", always=True)
    def _validate_num_coeffs(cls, val, values):
        """Check that the number of coeffs equals the number of cells inside the monitor."""
        num_cells = len(values["cell_indices"])
        num_coeffs = len(val)
        if num_cells != num_coeffs:
            raise ValidationError("The number of 'coeffs' must equal the number of 'cell_indices'.")
        return val


EMEMonitorDataType = Union[EMEModeSolverData, EMEFieldData, EMECoefficientData, ModeSolverData]
