"""Abstract base for simulation data structures."""

from __future__ import annotations

import pathlib
from abc import ABC
from os import PathLike
from typing import Any, Optional, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from tidy3d.components.base import Tidy3dBaseModel, skip_if_fields_missing
from tidy3d.components.base_sim.data.monitor_data import AbstractMonitorData
from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.data.utils import UnstructuredGridDatasetType
from tidy3d.components.file_util import replace_values
from tidy3d.components.monitor import AbstractMonitor
from tidy3d.components.types import FieldVal
from tidy3d.exceptions import DataError, FileError, Tidy3dKeyError, ValidationError


class AbstractSimulationData(Tidy3dBaseModel, ABC):
    """Stores data from a collection of :class:`AbstractMonitor` objects in
    a :class:`AbstractSimulation`.
    """

    simulation: AbstractSimulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`AbstractSimulation` associated with the data.",
    )

    data: tuple[AbstractMonitorData, ...] = pd.Field(
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
    def monitor_data(self) -> dict[str, AbstractMonitorData]:
        """Dictionary mapping monitor name to its associated :class:`AbstractMonitorData`."""
        return {monitor_data.monitor.name: monitor_data for monitor_data in self.data}

    @pd.root_validator(skip_on_failure=True)
    def data_monitors_match_sim(cls, values):
        """Ensure each :class:`AbstractMonitorData` in ``.data`` corresponds to a monitor in
        ``.simulation``.
        """
        sim = values.get("simulation")
        data = values.get("data")

        for mnt_data in data:
            try:
                monitor_name = mnt_data.monitor.name
                sim.get_monitor_by_name(monitor_name)
            except Tidy3dKeyError as exc:
                raise DataError(
                    f"Data with monitor name '{monitor_name}' supplied "
                    f"but not found in the original '{sim.type}'."
                ) from exc
        return values

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

    @staticmethod
    def _apply_log_scale(
        field_data: xr.DataArray,
        vmin: Optional[float] = None,
        db_factor: float = 1.0,
    ) -> xr.DataArray:
        """Prepare field data for log-scale plotting by handling zeros.

        Takes absolute value of the data, replaces zeros with a fill value
        (to prevent log10(0) warnings), and applies log10 scaling.

        Parameters
        ----------
        field_data : xr.DataArray
            The field data to prepare.
        vmin : float, optional
            The minimum value for the color scale. If provided, zeros are replaced
            with ``10 ** (vmin / db_factor)`` instead of NaN.
        db_factor : float
            Factor to multiply the log10 result by (e.g., 20 for dB scale of field,
            10 for dB scale of power). Default is 1 (pure log10 scale).

        Returns
        -------
        xr.DataArray
            The log-scaled field data.
        """
        fill_val = np.nan
        if vmin is not None:
            fill_val = 10 ** (vmin / db_factor)
        field_data = np.abs(field_data)
        field_data = field_data.where((field_data > 0) | np.isnan(field_data), fill_val)
        return db_factor * np.log10(field_data)

    def get_monitor_by_name(self, name: str) -> AbstractMonitor:
        """Return monitor named 'name'."""
        return self.simulation.get_monitor_by_name(name)

    def to_mat_file(self, fname: PathLike, **kwargs: Any) -> None:
        """Output the simulation data object as ``.mat`` MATLAB file.

        Parameters
        ----------
        fname : PathLike
            Full path to the output file. Should include ``.mat`` file extension.
        **kwargs : dict, optional
            Extra arguments to ``scipy.io.savemat``: see ``scipy`` documentation for more detail.

        Example
        -------
        >>> sim_data.to_mat_file('/path/to/file/data.mat') # doctest: +SKIP
        """
        # Check .mat file extension is given
        extension = pathlib.Path(fname).suffixes[0].lower()
        if len(extension) == 0:
            raise FileError(f"File '{fname}' missing extension.")
        if extension != ".mat":
            raise FileError(f"File '{fname}' should have a .mat extension.")

        # Handle m_dict in kwargs
        if "m_dict" in kwargs:
            raise ValueError(
                "'m_dict' is automatically determined by 'to_mat_file', can't pass to 'savemat'."
            )

        # Get SimData object as dictionary
        sim_dict = self.dict()

        # set long field names true by default, otherwise it wont save fields with > 31 characters
        if "long_field_names" not in kwargs:
            kwargs["long_field_names"] = True

        # Remove NoneType values from dict
        # Built from theory discussed in https://github.com/scipy/scipy/issues/3488
        modified_sim_dict = replace_values(sim_dict, None, [])

        try:
            from scipy.io import savemat

            savemat(fname, modified_sim_dict, **kwargs)
        except Exception as e:
            raise ValueError(
                "Could not save supplied simulation data to file. As this is an experimental "
                "feature, we may not be able to support the contents of your dataset. If you "
                "receive this error, please feel free to raise an issue on our front end "
                "repository so we can investigate."
            ) from e
