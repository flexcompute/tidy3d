"""Compatibility facade for historical data_array imports."""

from __future__ import annotations

import sys
import types
from typing import Any

from . import _spatial_sampling
from ._constants import (
    DATA_ARRAY_VALUE_NAME,
    DIM_ATTRS,
    DIPOLE_EMISSION_INTENSITY_UNITS,
    HDF5_SHUFFLE_MIN_BYTES,
)
from ._plot import (
    _SanitizedPlotProxy,
    _StructuredSpatialPlotProxy,
)
from ._spatial_sampling import (
    _check_sample_count,
    _clamp_steps_to_budget,
    _grid_from_bounds,
    _grid_steps,
    _sampling_counts,
)
from ._traced import _TracedDataset
from .abstract_spatial import AbstractSpatialDataArray
from .base import DataArray
from .eme import (
    EMECoefficientDataArray,
    EMEFluxDataArray,
    EMEFreqModeDataArray,
    EMEInterfaceCellIndexDataArray,
    EMEInterfaceDiagnosticDataArray,
    EMEInterfaceSMatrixDataArray,
    EMEModeIndexDataArray,
    EMEOverlapDataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    EMESMatrixDataArray,
    EMETraceMetricDataArray,
)
from .microwave import (
    CurrentArray,
    CurrentFreqDataArray,
    CurrentFreqModeDataArray,
    CurrentFreqTerminalDataArray,
    CurrentFreqTerminalModeDataArray,
    CurrentIntegralResultType,
    CurrentTimeDataArray,
    ImpedanceArray,
    ImpedanceFreqDataArray,
    ImpedanceFreqModeDataArray,
    ImpedanceFreqModeModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    ImpedanceModeDataArray,
    ImpedanceResultType,
    ImpedanceTerminalDataArray,
    ImpedanceTimeDataArray,
    IntegralResultType,
    VoltageArray,
    VoltageFreqDataArray,
    VoltageFreqModeDataArray,
    VoltageFreqModeTerminalDataArray,
    VoltageFreqTerminalDataArray,
    VoltageFreqTerminalModeDataArray,
    VoltageIntegralResultType,
    VoltageTimeDataArray,
    _make_base_result_data_array,
    _make_current_data_array,
    _make_impedance_data_array,
    _make_voltage_data_array,
)
from .monitor import (
    DiffractionDataArray,
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    GroupIndexDataArray,
    ModeAmpsDataArray,
    ModeAmpsTimeDataArray,
    ModeDispersionDataArray,
    ModeIndexDataArray,
    StructureLossDataArray,
)
from .registry import (
    DATA_ARRAY_MAP,
    DATA_ARRAY_TYPES,
)
from .spatial import (
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
    ScalarTerminalFieldDataArray,
    SpatialDataArray,
    SphericalAngleDataArray,
)
from .spectral import (
    FreqDataArray,
    FreqModeDataArray,
    FreqModeModeDataArray,
    FreqModeTerminalDataArray,
    FreqTerminalDataArray,
    FreqTerminalModeDataArray,
    FreqTerminalTerminalDataArray,
    MixedModeDataArray,
    ModeDataArray,
    TerminalDataArray,
    TimeDataArray,
)
from .tcad import (
    CFLHistoryDataArray,
    ChargeDataArray,
    ConvergenceHistoryDataArray,
    FreqVoltageDataArray,
    HeatDataArray,
    PerturbationCoefficientDataArray,
    SpatialVoltageDataArray,
    SteadyVoltageDataArray,
)
from .unstructured import (
    CellDataArray,
    IndexedDataArray,
    IndexedDataArrayTypes,
    IndexedFieldDataArray,
    IndexedFieldTimeDataArray,
    IndexedFieldVoltageDataArray,
    IndexedFreqDataArray,
    IndexedSurfaceFieldDataArray,
    IndexedSurfaceFieldTimeDataArray,
    IndexedSurfaceFreqDataArray,
    IndexedSurfaceTimeDataArray,
    IndexedTimeDataArray,
    IndexedVoltageDataArray,
    PointDataArray,
    TriangleMeshDataArray,
)

__all__ = [
    "DATA_ARRAY_MAP",
    "DATA_ARRAY_TYPES",
    "DATA_ARRAY_VALUE_NAME",
    "DIM_ATTRS",
    "DIPOLE_EMISSION_INTENSITY_UNITS",
    "HDF5_SHUFFLE_MIN_BYTES",
    "MAX_SPATIAL_SAMPLES",
    "AbstractSpatialDataArray",
    "CFLHistoryDataArray",
    "CellDataArray",
    "ChargeDataArray",
    "ConvergenceHistoryDataArray",
    "CurrentArray",
    "CurrentFreqDataArray",
    "CurrentFreqModeDataArray",
    "CurrentFreqTerminalDataArray",
    "CurrentFreqTerminalModeDataArray",
    "CurrentIntegralResultType",
    "CurrentTimeDataArray",
    "DataArray",
    "DiffractionDataArray",
    "DipoleEmissionDataArray",
    "DipoleEmissionPositionDataArray",
    "EMECoefficientDataArray",
    "EMEFluxDataArray",
    "EMEFreqModeDataArray",
    "EMEInterfaceCellIndexDataArray",
    "EMEInterfaceDiagnosticDataArray",
    "EMEInterfaceSMatrixDataArray",
    "EMEModeIndexDataArray",
    "EMEOverlapDataArray",
    "EMESMatrixDataArray",
    "EMEScalarFieldDataArray",
    "EMEScalarModeFieldDataArray",
    "EMETraceMetricDataArray",
    "FieldProjectionAngleDataArray",
    "FieldProjectionCartesianDataArray",
    "FieldProjectionKSpaceDataArray",
    "FluxDataArray",
    "FluxTimeDataArray",
    "FreqDataArray",
    "FreqModeDataArray",
    "FreqModeModeDataArray",
    "FreqModeTerminalDataArray",
    "FreqTerminalDataArray",
    "FreqTerminalModeDataArray",
    "FreqTerminalTerminalDataArray",
    "FreqVoltageDataArray",
    "GroupIndexDataArray",
    "HeatDataArray",
    "ImpedanceArray",
    "ImpedanceFreqDataArray",
    "ImpedanceFreqModeDataArray",
    "ImpedanceFreqModeModeDataArray",
    "ImpedanceFreqTerminalTerminalDataArray",
    "ImpedanceModeDataArray",
    "ImpedanceResultType",
    "ImpedanceTerminalDataArray",
    "ImpedanceTimeDataArray",
    "IndexedDataArray",
    "IndexedDataArrayTypes",
    "IndexedFieldDataArray",
    "IndexedFieldTimeDataArray",
    "IndexedFieldVoltageDataArray",
    "IndexedFreqDataArray",
    "IndexedSurfaceFieldDataArray",
    "IndexedSurfaceFieldTimeDataArray",
    "IndexedSurfaceFreqDataArray",
    "IndexedSurfaceTimeDataArray",
    "IndexedTimeDataArray",
    "IndexedVoltageDataArray",
    "IntegralResultType",
    "MixedModeDataArray",
    "ModeAmpsDataArray",
    "ModeAmpsTimeDataArray",
    "ModeDataArray",
    "ModeDispersionDataArray",
    "ModeIndexDataArray",
    "PerturbationCoefficientDataArray",
    "PointDataArray",
    "ScalarFieldDataArray",
    "ScalarFieldTimeDataArray",
    "ScalarModeFieldCylindricalDataArray",
    "ScalarModeFieldDataArray",
    "ScalarTerminalFieldDataArray",
    "SpatialDataArray",
    "SpatialVoltageDataArray",
    "SphericalAngleDataArray",
    "SteadyVoltageDataArray",
    "StructureLossDataArray",
    "TerminalDataArray",
    "TimeDataArray",
    "TriangleMeshDataArray",
    "VoltageArray",
    "VoltageFreqDataArray",
    "VoltageFreqModeDataArray",
    "VoltageFreqModeTerminalDataArray",
    "VoltageFreqTerminalDataArray",
    "VoltageFreqTerminalModeDataArray",
    "VoltageIntegralResultType",
    "VoltageTimeDataArray",
    "_SanitizedPlotProxy",
    "_StructuredSpatialPlotProxy",
    "_TracedDataset",
    "_check_sample_count",
    "_clamp_steps_to_budget",
    "_grid_from_bounds",
    "_grid_steps",
    "_make_base_result_data_array",
    "_make_current_data_array",
    "_make_impedance_data_array",
    "_make_voltage_data_array",
    "_sampling_counts",
]

# 'MAX_SPATIAL_SAMPLES' lives in '_spatial_sampling', which reads it at call time. Re-exporting
# the int would hand this module a copy, so patching 'data_array.MAX_SPATIAL_SAMPLES' would no
# longer reach the sampling checks; reads and writes of that name are forwarded instead.
_FORWARDED = "MAX_SPATIAL_SAMPLES"


def __getattr__(name: str) -> Any:
    if name == _FORWARDED:
        return _spatial_sampling.MAX_SPATIAL_SAMPLES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), _FORWARDED])


class _ForwardingModule(types.ModuleType):
    """Module type that sends writes of the sample cap to '_spatial_sampling'."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == _FORWARDED:
            setattr(_spatial_sampling, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == _FORWARDED:
            delattr(_spatial_sampling, name)
        else:
            super().__delattr__(name)


sys.modules[__name__].__class__ = _ForwardingModule
