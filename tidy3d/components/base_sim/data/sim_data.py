"""Abstract base for simulation data structures."""

from __future__ import annotations

from abc import ABC
from typing import Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ....exceptions import DataError, Tidy3dKeyError, ValidationError
from ...base import Tidy3dBaseModel, skip_if_fields_missing
from ...data.utils import UnstructuredGridDatasetType
from ...types import FieldVal
from ..simulation import AbstractSimulation
from .monitor_data import AbstractMonitorData


class AbstractSimulationData(Tidy3dBaseModel, ABC):
    """Stores data from a collection of :class:`AbstractMonitor` objects in
    a :class:`AbstractSimulation`.
    """

    simulation: AbstractSimulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`AbstractSimulation` associated with the data.",
    )

    data: Tuple[AbstractMonitorData, ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`AbstractMonitorData` instances "
        "associated with the monitors of the original :class:`AbstractSimulation`.",
    )

    log: str = pd.Field(
        None,
        title="Solver Log",
        description="A string containing the log information from the simulation run.",
    )

    def __getitem__(self, monitor_name: str) -> AbstractMonitorData:
        """Get a :class:`.AbstractMonitorData` by name. Apply symmetry if applicable."""
        monitor_data = self.monitor_data[monitor_name]
        return monitor_data.symmetry_expanded_copy

    @property
    def monitor_data(self) -> Dict[str, AbstractMonitorData]:
        """Dictionary mapping monitor name to its associated :class:`AbstractMonitorData`."""
        return {monitor_data.monitor.name: monitor_data for monitor_data in self.data}

    @pd.validator("data", always=True)
    @skip_if_fields_missing(["simulation"])
    def data_monitors_match_sim(cls, val, values):
        """Ensure each :class:`AbstractMonitorData` in ``.data`` corresponds to a monitor in
        ``.simulation``.
        """
        sim = values.get("simulation")

        for mnt_data in val:
            try:
                monitor_name = mnt_data.monitor.name
                sim.get_monitor_by_name(monitor_name)
            except Tidy3dKeyError as exc:
                raise DataError(
                    f"Data with monitor name '{monitor_name}' supplied "
                    f"but not found in the original '{sim.type}'."
                ) from exc
        return val

    @pd.validator("data", always=True)
    @skip_if_fields_missing(["simulation"])
    def validate_no_ambiguity(cls, val, values):
        """Ensure all :class:`AbstractMonitorData` entries in ``.data`` correspond to different
        monitors in ``.simulation``.
        """
        names = [mnt_data.monitor.name for mnt_data in val]

        if len(set(names)) != len(names):
            raise ValidationError("Some entries of '.data' provide data for same monitor(s).")

        return val

    @staticmethod
    def _field_component_value(
        field_component: Union[xr.DataArray, UnstructuredGridDatasetType], val: FieldVal
    ) -> xr.DataArray:
        """return the desired value of a field component.

        Parameter
        ----------
        field_component : Union[xarray.DataArray, UnstructuredGridDatasetType]
            Field component from which to calculate the value.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase']
            Which part of the field to return.

        Returns
        -------
        xarray.DataArray
            Value extracted from the field component.
        """
        if val in ("real", "re"):
            field_value = field_component.real
            field_value = field_value.rename(f"Re{{{field_component.name}}}")

        elif val in ("imag", "im"):
            field_value = field_component.imag
            field_value = field_value.rename(f"Im{{{field_component.name}}}")

        elif val == "abs":
            field_value = np.abs(field_component)
            field_value = field_value.rename(f"|{field_component.name}|")

        elif val == "abs^2":
            field_value = np.abs(field_component) ** 2
            field_value = field_value.rename(f"|{field_component.name}|²")

        elif val == "phase":
            field_value = np.arctan2(field_component.imag, field_component.real)
            field_value = field_value.rename(f"∠{field_component.name}")

        else:
            raise Tidy3dKeyError(
                f"Couldn't find 'val={val}'. Must be one of 'real', 're', 'imag', 'im', 'abs', 'abs^2', 'phase'."
            )

        return field_value
