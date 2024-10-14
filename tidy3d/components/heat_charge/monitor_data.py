"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import KELVIN, VOLT
from ...log import log
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..base_sim.data.monitor_data import AbstractMonitorData
from ..data.data_array import CapacitanceCurveDataArray, SpatialDataArray
from ..data.dataset import IndexedDataArray, TetrahedralGridDataset, TriangularGridDataset
from ..types import Coordinate, ScalarSymmetry, annotate_type
from .monitor import (
    ChargeSimulationMonitor,
    HeatChargeMonitorType,
    TemperatureMonitor,
    TemporalTemperatureMonitor,
    TemporalVoltageMonitor,
    VoltageMonitor,
)

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]


class HeatChargeMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: Tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        (0, 0, 0),
        title="Symmetry Center",
        description="Symmetry center of the original simulation in x, y, and z.",
    )

    @property
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()

    @abstractmethod
    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""

    # def _symmetry_expanded_copy(self, property):
    def _symmetry_expanded_copy(self, property: FieldDataset) -> FieldDataset:
        """Return the property with symmetry applied."""

        # no symmetry
        if all(sym == 0 for sym in self.symmetry):
            return property

        new_property = copy.copy(property)

        mnt_bounds = np.array(self.monitor.bounds)

        if isinstance(new_property, SpatialDataArray):
            data_bounds = [
                [np.min(new_property.x), np.min(new_property.y), np.min(new_property.z)],
                [np.max(new_property.x), np.max(new_property.y), np.max(new_property.z)],
            ]
        else:
            data_bounds = new_property.bounds

        dims_need_clipping_left = []
        dims_need_clipping_right = []
        for dim in range(3):
            # do not expand monitor with zero size along symmetry direction
            # this is done because 2d unstructured data does not support this
            if self.symmetry[dim] == 1:
                center = self.symmetry_center[dim]

                if mnt_bounds[1][dim] < data_bounds[0][dim]:
                    # (note that mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim] will be satisfied based on backend behavior)
                    # simple reflection
                    new_property = new_property.reflect(
                        axis=dim, center=center, reflection_only=True
                    )
                elif mnt_bounds[0][dim] < 2 * center - data_bounds[0][dim]:
                    # expand only if monitor bounds missing data
                    # if we do expand, simply reflect symmetrically the whole data
                    new_property = new_property.reflect(axis=dim, center=center)

                    # if it turns out that we expanded too much, we will trim unnecessary data later
                    if mnt_bounds[0][dim] > 2 * center - data_bounds[1][dim]:
                        dims_need_clipping_left.append(dim)

                    # likewise, if some of original data was only for symmetry expansion, thim excess on the right
                    if mnt_bounds[1][dim] < data_bounds[1][dim]:
                        dims_need_clipping_right.append(dim)

        # trim over-expanded data
        if len(dims_need_clipping_left) > 0 or len(dims_need_clipping_right) > 0:
            # enlarge clipping domain on positive side arbitrary by 1
            # should not matter by how much
            clip_bounds = [mnt_bounds[0] - 1, mnt_bounds[1] + 1]
            for dim in dims_need_clipping_left:
                clip_bounds[0][dim] = mnt_bounds[0][dim]

            for dim in dims_need_clipping_right:
                clip_bounds[1][dim] = mnt_bounds[1][dim]

            if isinstance(new_property, SpatialDataArray):
                new_property = new_property.sel_inside(clip_bounds)
            else:
                new_property = new_property.box_clip(bounds=clip_bounds)

        return new_property


class TemperatureData(HeatChargeMonitorData):
    """Data associated with a :class:`TemperatureMonitor`: spatial temperature field.

    Example
    -------
    >>> from tidy3d import TemperatureMonitor, SpatialDataArray
    >>> import numpy as np
    >>> temp_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="temperature")
    >>> temp_mnt_data = TemperatureData(
    ...     monitor=temp_mnt, temperature=temp_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> temp_mnt_data_expanded = temp_mnt_data.symmetry_expanded_copy
    """

    monitor: TemperatureMonitor = pd.Field(
        ..., title="Monitor", description="Temperature monitor associated with the data."
    )

    temperature: Optional[FieldDataset] = pd.Field(
        ...,
        title="Temperature",
        description="Spatial temperature field.",
        units=KELVIN,
    )

    @pd.validator("temperature", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|T|², K²"
        else:
            return "T, K"

    @property
    def symmetry_expanded_copy(self) -> TemperatureData:
        """Return copy of self with symmetry applied."""

        new_temp = self._symmetry_expanded_copy(property=self.temperature)
        return self.updated_copy(temperature=new_temp, symmetry=(0, 0, 0))


class VoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`VoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import VoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = VoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = VoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: VoltageMonitor = pd.Field(
        ..., title="Monitor", description="Electric potential monitor associated with the data."
    )

    voltage: Optional[FieldDataset] = pd.Field(
        ...,
        title="Voltage (electric potential)",
        description="Spatial electric potential field.",
        units=VOLT,
    )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|², sigma²"
        else:
            return "V, sigma"

    @pd.validator("voltage", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> VoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))


class HeatChargeDataset(Tidy3dBaseModel):
    """Class that deals with parameter-depending fields."""

    base_data: FieldDataset = pd.Field(title="Base data", description="Spatial dataset")

    field_series: Dict[str, Tuple[IndexedDataArray, ...]] = pd.Field(
        title="Field series", description="Dictionary of field solutions. "
    )

    parameter_array: Tuple[pd.FiniteFloat, ...] = pd.Field(
        title="Parameter array",
        description="Array containing the parameter values at which the field series are stored.",
    )

    @cached_property
    def num_fields_saved(self):
        """Number of fields stored"""
        return len(self.parameter_array)

    def get_field(self, field: str, loc: int):
        """Returns the field specified by 'field' stored at the position specified by 'loc'"""

        assert loc < self.num_fields_saved
        return self.base_data.updated_copy(values=self.field_series[field][loc])

    @pd.root_validator(skip_on_failure=True)
    def check_data_and_params_have_same_length(cls, values):
        """Check that both field series and parameter array have the same length."""

        field_series = values["field_series"]
        parameter_array = values["parameter_array"]
        for key in field_series.keys():
            assert len(field_series[key]) == len(parameter_array)

        return values


class ChargeSimulationData(HeatChargeMonitorData):
    """Class that stores Charge simulation data.
    Example
    -------
    ...
    """

    monitor: Union[ChargeSimulationMonitor] = pd.Field(
        ..., title="Monitor", description="Data associated with a Charge simulation."
    )

    data_series: Optional[HeatChargeDataset] = pd.Field(
        None, title="Data series", description="Contains the data."
    )

    capacitance_curve: Optional[Dict[str, CapacitanceCurveDataArray]] = pd.Field(
        None,
        title="Capacitance curve",
        description="Small signal capacitance associated to the monitor.",
    )

    @pd.validator("data_series", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> ChargeSimulationData:
        """Return copy of self with symmetry applied."""

        new_series = {}

        for key, data in self.data_series.field_series.items():
            new_series[key] = self._symmetry_expanded_copy(property=data)

        return self.updated_copy(data_series=self.data_series.updated_copy(field_series=new_series))

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|², Electrons², Holes², Donors², Acceptors²"
        else:
            return "V, Electrons, Holes, Donors, Acceptors"


class TemporalHeatChargeDataset(Tidy3dBaseModel):
    """Class to deal with time-varying device fields."""

    base_data: FieldDataset = pd.Field(title="Base data", description="Spatial dataset")

    field_time_series: Tuple[IndexedDataArray, ...] = pd.Field(
        title="Field time-series", description="Field values at different time steps."
    )

    time_steps_array: Tuple[pd.NonNegativeInt, ...] = pd.Field(
        title="Time stes array",
        description="Array containing the time steps at which the field is stored.",
    )

    @cached_property
    def num_time_steps(self):
        """Number of time steps in series"""
        return len(self.field_time_series)

    def return_time_step(self, time_step: int):
        """Add a little description"""

        assert time_step < self.num_time_steps
        return self.base_data.updated_copy(values=self.field_time_series[time_step])

    @pd.root_validator(skip_on_failure=True)
    def check_data_and_times_have_same_length(cls, values):
        """Check that both time series and data series have the same length."""

        field_time_series = values["field_time_series"]
        time_steps_array = values["time_steps_array"]
        assert len(field_time_series) == len(time_steps_array)

        return values


class TemporalData(HeatChargeMonitorData):
    """Data associated with a :class:`TemporalVoltageMonitor`: spatial voltage field.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> temp_data = td.SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> vals = [temp_data.values, temp_data.values]
    >>> time_steps = [0, 1]
    >>> time_dataset = td.TemporalDeviceDataset(
    ...     base_data=temp_data,
    ...     field_time_series=vals,
    ...     time_steps_array=time_steps
    ... )
    >>> temp_mnt = td.TemporalTemperatureMonitor(size=(1, 2, 3), name="temperature")
    >>> temp_mnt_data = TemporalData(
    ...     monitor=temp_mnt, time_series=time_dataset, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> temp_mnt_data_expanded = temp_mnt_data.symmetry_expanded_copy
    """

    monitor: Union[TemporalTemperatureMonitor, TemporalVoltageMonitor] = pd.Field(
        ..., title="Monitor", description="Time-varying temperature or voltage monitor."
    )

    time_series: Optional[TemporalHeatChargeDataset] = pd.Field(
        None,
        title="time series",
        description="Container for time-varying data. This can contain either "
        f"temperature ({KELVIN}) or voltage ({VOLT}) data",
    )

    @pd.validator("time_series", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> TemporalData:
        """Return copy of self with symmetry applied."""

        new_time_series = []

        for n in range(self.time_series.num_time_steps):
            data = self.time_series.return_time_step(n)
            new_time_series.append((self._symmetry_expanded_copy(property=data)).values)

        return self.updated_copy(
            time_series=self.time_series.updated_copy(field_time_series=new_time_series)
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            if isinstance(self.monitor, TemporalTemperatureMonitor):
                return "|T|², K²"
            elif isinstance(self.monitor, TemporalVoltageMonitor):
                return "|V|², sigma²"
        else:
            if isinstance(self.monitor, TemporalTemperatureMonitor):
                return "T, K"
            elif isinstance(self.monitor, TemporalVoltageMonitor):
                return "V, sigma"


HeatChargeMonitorDataType = Union[TemperatureData, VoltageData, TemporalData, ChargeSimulationData]
