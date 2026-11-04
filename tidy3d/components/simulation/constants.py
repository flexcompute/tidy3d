"""Limits and warning thresholds for FDTD simulations."""

from __future__ import annotations

from flexcompute.core._migration.solver.fdtd.schema.validation_schema import (
    MAX_DIFFRACTION_ORDER_GRID_SIZE,  # noqa: F401 - re-exported; the limit lives once in core
    MAX_MONITOR_FREQUENCY_RANGE_PARAMETER,  # noqa: F401 - re-exported; the limit lives once in core
    MAX_NUM_SOURCES,  # noqa: F401 - re-exported; the limit lives once in core
    MIN_GRIDS_PER_WVL,  # noqa: F401 - re-exported; the limit lives once in core
    MIN_MONITOR_FREQUENCY_RANGE_PARAMETER,  # noqa: F401 - re-exported; the limit lives once in core
)

from tidy3d.constants import inf

MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16
WARN_SIM_DOMAIN_CELLS_EXCLUDING_PML = 100
MAX_TIME_MONITOR_STEPS = 5000
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
MAX_ADJOINT_SAMPLE_SETS_SIZE_GB = 5
WARN_MODE_NUM_CELLS = 1e5
NUM_CELLS_WARN_EPSILON = 100_000_000
NUM_STRUCTURES_WARN_EPSILON = 10_000
PML_HEIGHT_FOR_0_DIMS = inf
FIXED_ANGLE_DT_SAFETY_FACTOR = 0.9
RF_FREQ_WARNING = 300e9
MODAL_PEC_FRAME_NAME_PREFIX = "__tidy3d_modal_pec_frame_"
THIN_LENS_SOURCE_SETUP_EVALUATIONS = 4
THIN_LENS_MONITOR_SETUP_EVALUATIONS = 2
THIN_LENS_FIELD_COMPONENTS = 6
