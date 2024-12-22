"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.base_sim.data.monitor_data import AbstractMonitorData
from tidy3d.components.data.data_array import (
    DataArray,
    IndexedDataArray,
    IndexVoltageDataArray,
    SpatialDataArray,
    SteadyCapacitanceVoltageDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.monitors.heat import (
    SteadyCapacitanceMonitor,
    SteadyFreeChargeCarrierMonitor,
    SteadyVoltageMonitor,
    TCADMonitorTypes,
    TemperatureMonitor,
)
from tidy3d.components.types import TYPE_TAG_STR, Coordinate, ScalarSymmetry, annotate_type
from tidy3d.constants import KELVIN, VOLT
from tidy3d.log import log

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]
UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class HeatChargeMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: TCADMonitorTypes = pd.Field(
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

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(temperature=self.temperature)

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

    @pd.validator("temperature", always=True)
    @skip_if_fields_missing(["monitor"])
    def check_correct_data_type(cls, val, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")

        if isinstance(val, TetrahedralGridDataset) or isinstance(val, TriangularGridDataset):
            if not isinstance(val.values, IndexedDataArray):
                raise ValueError(
                    f"Monitor {mnt} of type 'TemperatureMonitor' cannot be associated with data arrays "
                    "of type 'IndexVoltageDataArray'."
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


class SteadyVoltageData(HeatChargeMonitorData):
    """Data associated with a :class:`SteadyVoltageMonitor`: spatial electric potential field.

    Example
    -------
    >>> from tidy3d import SteadyVoltageMonitor, SpatialDataArray
    >>> import numpy as np
    >>> voltage_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> voltage_mnt = SteadyVoltageMonitor(size=(1, 2, 3), name="voltage")
    >>> voltage_mnt_data = SteadyVoltageData(
    ...     monitor=voltage_mnt, voltage=voltage_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> voltage_mnt_data_expanded = voltage_mnt_data.symmetry_expanded_copy
    """

    monitor: SteadyVoltageMonitor = pd.Field(
        ..., title="Monitor", description="Electric potential monitor associated with the data."
    )

    voltage: Optional[FieldDataset] = pd.Field(
        ...,
        title="Voltage (electric potential)",
        description="Spatial electric potential field.",
        units=VOLT,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(voltage=self.voltage)

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

    @pd.validator("voltage", always=True)
    @skip_if_fields_missing(["monitor"])
    def check_correct_data_type(cls, val, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")

        if isinstance(val, TetrahedralGridDataset) or isinstance(val, TriangularGridDataset):
            if not isinstance(val.values, IndexedDataArray):
                raise ValueError(
                    f"Monitor {mnt} of type 'SteadyVoltageMonitor' cannot be associated with data arrays "
                    "of type 'IndexVoltageDataArray'."
                )

        return val

    @property
    def symmetry_expanded_copy(self) -> SteadyVoltageData:
        """Return copy of self with symmetry applied."""

        new_phi = self._symmetry_expanded_copy(property=self.voltage)
        return self.updated_copy(voltage=new_phi, symmetry=(0, 0, 0))


class SteadyPotentialData(HeatChargeMonitorData):
    """Class that stores electric potential from a charge simulation."""

    monitor: SteadyVoltageMonitor = pd.Field(
        ...,
        title="Voltage monitor",
        description="Electric potential monitor associated with a Charge simulation.",
    )

    potential: UnstructuredFieldType = pd.Field(
        None,
        title="Voltage series",
        description="Contains the voltages.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(potential=self.potential)

    @pd.validator("potential", always=True)
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

    @pd.validator("potential", always=True)
    @skip_if_fields_missing(["monitor"])
    def check_correct_data_type(cls, val, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")

        if isinstance(val, TetrahedralGridDataset) or isinstance(val, TriangularGridDataset):
            if not isinstance(val.values, IndexVoltageDataArray):
                raise ValueError(
                    f"Monitor {mnt} of type 'SteadyVoltageMonitor' is not associated with data arrays "
                    "of type 'IndexVoltageDataArray' and cannot be associated with an applied voltage."
                )

        return val

    @property
    def symmetry_expanded_copy(self) -> SteadyPotentialData:
        """Return copy of self with symmetry applied."""

        new_potential = self._symmetry_expanded_copy(property=self.potential)
        return self.updated_copy(potential=new_potential, symmetry=(0, 0, 0))

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|²"
        else:
            return "V"


class SteadyFreeCarrierData(HeatChargeMonitorData):
    """Class that stores free carrier concentration in Charge simulations."""

    monitor: SteadyFreeChargeCarrierMonitor = pd.Field(
        ...,
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons: UnstructuredFieldType = pd.Field(
        None,
        title="Electrons series",
        description="Contains the electrons.",
        discriminator=TYPE_TAG_STR,
    )

    holes: UnstructuredFieldType = pd.Field(
        None,
        title="Holes series",
        description="Contains the electrons.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(electrons=self.electrons, holes=self.holes)

    @pd.root_validator(skip_on_failure=True)
    def check_correct_data_type(cls, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")
        field_data = {field: values.get(field) for field in ["electrons", "holes"]}

        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {mnt}, the field {field} does not contain "
                        "data associated to any voltage value."
                    )

        return values

    @pd.root_validator(skip_on_failure=True)
    def warn_no_data(cls, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")
        electrons = values.get("electrons")
        holes = values.get("holes")

        if electrons is None or holes is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return values

    @property
    def symmetry_expanded_copy(self) -> SteadyFreeCarrierData:
        """Return copy of self with symmetry applied."""

        new_electrons = self._symmetry_expanded_copy(property=self.electrons)
        new_holes = self._symmetry_expanded_copy(property=self.holes)

        return self.updated_copy(
            electrons=new_electrons,
            holes=new_holes,
            symmetry=(0, 0, 0),
        )

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "Electrons², Holes²"
        else:
            return "Electrons, Holes"


class SteadyCapacitanceData(HeatChargeMonitorData):
    """Class that stores capacitance data from a Charge simulation."""

    monitor: SteadyCapacitanceMonitor = pd.Field(
        ...,
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: SteadyCapacitanceVoltageDataArray = pd.Field(
        None,
        title="Hole capacitance",
        description="Small signal capacitance (dQh/dV) associated to the monitor.",
    )

    electron_capacitance: SteadyCapacitanceVoltageDataArray = pd.Field(
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


TCADMonitorDataTypes = Union[
    TemperatureData,
    SteadyVoltageData,
    SteadyPotentialData,
    SteadyFreeCarrierData,
    SteadyCapacitanceData,
]
