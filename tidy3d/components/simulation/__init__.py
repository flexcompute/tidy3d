"""Compatibility exports for FDTD simulation models and limits."""

from __future__ import annotations

from tidy3d.components.monitor import DiffractionMonitor, FieldMonitor, ModeMonitor
from tidy3d.components.scene import MAX_NUM_MEDIUMS
from tidy3d.components.thin_lens import MAX_THIN_LENS_SETUP_WORK_UNITS
from tidy3d.packaging import tidy3d_extras

from .boundaries import validate_boundaries_for_zero_dims
from .constants import (
    FIXED_ANGLE_DT_SAFETY_FACTOR,
    MAX_CELLS_TIMES_STEPS,
    MAX_DIFFRACTION_ORDER_GRID_SIZE,
    MAX_GRID_CELLS,
    MAX_MONITOR_FREQUENCY_RANGE_PARAMETER,
    MAX_MONITOR_INTERNAL_DATA_SIZE_GB,
    MAX_NUM_SOURCES,
    MAX_SIMULATION_DATA_SIZE_GB,
    MAX_TIME_MONITOR_STEPS,
    MAX_TIME_STEPS,
    MIN_GRIDS_PER_WVL,
    MIN_MONITOR_FREQUENCY_RANGE_PARAMETER,
    MODAL_PEC_FRAME_NAME_PREFIX,
    NUM_CELLS_WARN_EPSILON,
    NUM_STRUCTURES_WARN_EPSILON,
    PML_HEIGHT_FOR_0_DIMS,
    RF_FREQ_WARNING,
    THIN_LENS_FIELD_COMPONENTS,
    THIN_LENS_MONITOR_SETUP_EVALUATIONS,
    THIN_LENS_SOURCE_SETUP_EVALUATIONS,
    WARN_MODE_NUM_CELLS,
    WARN_MONITOR_DATA_SIZE_GB,
    WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML,
    WARN_TIME_STEPS,
)
from .export import OpticalMediumExportKey
from .model import Simulation
from .yee import AbstractYeeGridSimulation

# Retained for the inverse-design plugin's legacy default monitor selection.
OutputMonitorTypes = (DiffractionMonitor, FieldMonitor, ModeMonitor)

__all__ = [
    "FIXED_ANGLE_DT_SAFETY_FACTOR",
    "MAX_CELLS_TIMES_STEPS",
    "MAX_DIFFRACTION_ORDER_GRID_SIZE",
    "MAX_GRID_CELLS",
    "MAX_MONITOR_FREQUENCY_RANGE_PARAMETER",
    "MAX_MONITOR_INTERNAL_DATA_SIZE_GB",
    "MAX_NUM_MEDIUMS",
    "MAX_NUM_SOURCES",
    "MAX_SIMULATION_DATA_SIZE_GB",
    "MAX_THIN_LENS_SETUP_WORK_UNITS",
    "MAX_TIME_MONITOR_STEPS",
    "MAX_TIME_STEPS",
    "MIN_GRIDS_PER_WVL",
    "MIN_MONITOR_FREQUENCY_RANGE_PARAMETER",
    "MODAL_PEC_FRAME_NAME_PREFIX",
    "NUM_CELLS_WARN_EPSILON",
    "NUM_STRUCTURES_WARN_EPSILON",
    "PML_HEIGHT_FOR_0_DIMS",
    "RF_FREQ_WARNING",
    "THIN_LENS_FIELD_COMPONENTS",
    "THIN_LENS_MONITOR_SETUP_EVALUATIONS",
    "THIN_LENS_SOURCE_SETUP_EVALUATIONS",
    "WARN_MODE_NUM_CELLS",
    "WARN_MONITOR_DATA_SIZE_GB",
    "WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML",
    "WARN_TIME_STEPS",
    "AbstractYeeGridSimulation",
    "OpticalMediumExportKey",
    "OutputMonitorTypes",
    "Simulation",
    "tidy3d_extras",
    "validate_boundaries_for_zero_dims",
]
