"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import KELVIN, VOLT
from ...log import log
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..base_sim.data.monitor_data import AbstractMonitorData
from ..data.data_array import DCCapacitanceDataArray, SpatialDataArray
from ..data.dataset import IndexedDataArray, TetrahedralGridDataset, TriangularGridDataset
from ..types import Coordinate, ScalarSymmetry, annotate_type
from .monitor import (
    HeatChargeMonitorTypes,
    StaticCapacitanceMonitor,
    StaticChargeCarrierMonitor,
    StaticVoltageMonitor,
    TemperatureMonitor,
)

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]


class HeatChargeMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorTypes = pd.Field(
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


class StaticVoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`VoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import StaticVoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = StaticVoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = StaticVoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: StaticVoltageMonitor = pd.Field(
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
    def symmetry_expanded_copy(self) -> StaticVoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))


class HeatChargeDataset(Tidy3dBaseModel):
    """Class that deals with parameter-depending fields."""

    base_data: FieldDataset = pd.Field(title="Base data", description="Spatial dataset")

    field_series: Tuple[IndexedDataArray, ...] = pd.Field(
        title="Field series", description="Tuple of field solutions. "
    )

    parameter_array: Tuple[pd.FiniteFloat, ...] = pd.Field(
        title="Parameter array",
        description="Array containing the parameter values at which the field series are stored.",
    )

    @cached_property
    def num_fields_saved(self):
        """Number of fields stored"""
        return len(self.parameter_array)

    def get_field(self, loc: int):
        """Returns the field specified by 'field' stored at the position specified by 'loc'"""

        assert loc < self.num_fields_saved
        return self.base_data.updated_copy(values=self.field_series[loc])

    @pd.root_validator(skip_on_failure=True)
    def check_data_and_params_have_same_length(cls, values):
        """Check that both field series and parameter array have the same length."""

        field_series = values["field_series"]
        parameter_array = values["parameter_array"]
        assert len(field_series) == len(parameter_array)

        return values


class StaticPotentialData(HeatChargeMonitorData):
    """Class that stores electric potential from a charge simulation."""

    monitor: StaticVoltageMonitor = pd.Field(
        ...,
        title="Voltage monitor",
        description="Electric potential monitor associated with a Charge simulation.",
    )

    voltage_series: HeatChargeDataset = pd.Field(
        None, title="Voltage series", description="Contains the voltages."
    )

    @pd.validator("voltage_series", always=True)
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
    def symmetry_expanded_copy(self) -> StaticPotentialData:
        """Return copy of self with symmetry applied."""

        new_voltages = self._symmetry_expanded_copy(property=self.voltage_series.field_series)

        return self.updated_copy(
            voltage_series=self.voltage_series.updated_copy(field_series=new_voltages)
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|²"
        else:
            return "V"


class StaticFreeCarrierData(HeatChargeMonitorData):
    """Class that stores free carrier concentration in Charge simulations."""

    monitor: StaticChargeCarrierMonitor = pd.Field(
        ...,
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons_series: HeatChargeDataset = pd.Field(
        None, title="Electrons series", description="Contains the electrons."
    )

    holes_series: HeatChargeDataset = pd.Field(
        None, title="Holes series", description="Contains the electrons."
    )

    @pd.root_validator(skip_on_failure=True)
    def warn_no_data(cls, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")
        electrons = values.get("electrons_series")
        holes = values.get("holes_series")

        if electrons is None or holes is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return values

    @property
    def symmetry_expanded_copy(self) -> StaticFreeCarrierData:
        """Return copy of self with symmetry applied."""

        new_electrons = self._symmetry_expanded_copy(property=self.electrons_series.field_series)
        new_holes = self._symmetry_expanded_copy(property=self.holes_series.field_series)

        return self.updated_copy(
            electrons_series=self.electrons_series.updated_copy(field_series=new_electrons),
            holes_series=self.holes_series.updated_copy(field_series=new_holes),
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "Electrons², Holes²"
        else:
            return "Electrons, Holes"


class StaticCapacitanceData(HeatChargeMonitorData):
    """Class that stores capacitance data from a Charge simulation."""

    monitor: StaticCapacitanceMonitor = pd.Field(
        ...,
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: DCCapacitanceDataArray = pd.Field(
        None,
        title="Hole capacitance",
        description="Small signal capacitance (dQh/dV) associated to the monitor.",
    )

    electron_capacitance: DCCapacitanceDataArray = pd.Field(
        None,
        title="Electron capacitance",
        description="Small signal capacitance (dQe/dV) associated to the monitor.",
    )

    @pd.validator("hole_capacitance", always=True)
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
        return ""


HeatChargeMonitorDataType = Union[
    TemperatureData,
    StaticVoltageData,
    StaticPotentialData,
    StaticFreeCarrierData,
    StaticCapacitanceData,
]
