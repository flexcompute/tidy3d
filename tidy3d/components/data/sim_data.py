"""Simulation Level Data"""

from __future__ import annotations

import json
import pathlib
from abc import ABC
from collections import defaultdict
from typing import Callable, Tuple, Union

import h5py
import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ...constants import C_0, inf
from ...exceptions import DataError, FileError, Tidy3dKeyError
from ...log import log
from ..autograd.utils import split_list
from ..base import JSON_TAG, Tidy3dBaseModel
from ..base_sim.data.sim_data import AbstractSimulationData
from ..file_util import replace_values
from ..monitor import Monitor
from ..simulation import Simulation
from ..source import GaussianPulse, SourceType
from ..structure import Structure
from ..types import Ax, Axis, ColormapType, FieldVal, PlotScale, annotate_type
from ..viz import add_ax_if_none, equal_aspect
from .data_array import FreqDataArray
from .monitor_data import (
    AbstractFieldData,
    FieldTimeData,
    MonitorDataType,
    MonitorDataTypes,
)

DATA_TYPE_MAP = {data.__fields__["monitor"].type_: data for data in MonitorDataTypes}

# maps monitor type (string) to the class of the corresponding data
DATA_TYPE_NAME_MAP = {val.__fields__["monitor"].type_.__name__: val for val in MonitorDataTypes}

# residuals below this are considered good fits for broadband adjoint source creation
RESIDUAL_CUTOFF_ADJOINT = 1e-6


class AdjointSourceInfo(Tidy3dBaseModel):
    """Stores information about the adjoint sources to pass to autograd pipeline."""

    sources: Tuple[annotate_type(SourceType), ...] = pd.Field(
        ...,
        title="Adjoint Sources",
        description="Set of processed sources to include in the adjoint simulation.",
    )

    post_norm: Union[float, FreqDataArray] = pd.Field(
        ...,
        title="Post Normalization Values",
        description="Factor to multiply the adjoint fields by after running "
        "given the adjoint source pipeline used.",
    )

    normalize_sim: bool = pd.Field(
        ...,
        title="Normalize Adjoint Simulation",
        description="Whether the adjoint simulation needs to be normalized "
        "given the adjoint source pipeline used.",
    )


class AbstractYeeGridSimulationData(AbstractSimulationData, ABC):
    """Data from an :class:`.AbstractYeeGridSimulation` involving
    electromagnetic fields on a Yee grid.

    Notes
    -----

        The ``SimulationData`` objects store a copy of the original :class:`.Simulation`:, so it can be recovered if the
        ``SimulationData`` is loaded in a new session and the :class:`.Simulation` is no longer in memory.

        More importantly, the ``SimulationData`` contains a reference to the data for each of the monitors within the
        original :class:`.Simulation`. This data can be accessed directly using the name given to the monitors initially.
    """

    def load_field_monitor(self, monitor_name: str) -> AbstractFieldData:
        """Load monitor and raise exception if not a field monitor."""
        mon_data = self[monitor_name]
        if not isinstance(mon_data, AbstractFieldData):
            raise DataError(
                f"data for monitor '{monitor_name}' does not contain field data "
                f"as it is a '{type(mon_data)}'."
            )
        return mon_data

    def at_centers(self, field_monitor_name: str) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to center locations on
            the Yee grid.
        """

        monitor_data = self.load_field_monitor(field_monitor_name)
        return monitor_data.at_coords(monitor_data.colocation_centers)

    def _at_boundaries(self, monitor_data: xr.Dataset) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell
        boundaries.

        Parameters
        ----------
        monitor_data : xr.Dataset
            Monitor data to be co-located.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        if monitor_data.monitor.colocate:
            # TODO: this still errors if monitor_data.colocate is allowed to be ``True`` in the
            # adjoint plugin, and the monitor data is tracked in a gradient computation. It seems
            # interpolating does something to the arrays that makes the JAX chain work.
            return monitor_data.package_colocate_results(monitor_data.field_components)

        # colocate to monitor grid boundaries
        return monitor_data.at_coords(monitor_data.colocation_boundaries)

    def at_boundaries(self, field_monitor_name: str) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell
        boundaries.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        # colocate to monitor grid boundaries
        return self._at_boundaries(self.load_field_monitor(field_monitor_name))

    def _get_poynting_vector(self, field_monitor_data: AbstractFieldData) -> xr.Dataset:
        """return ``xarray.Dataset`` of the Poynting vector at Yee cell centers.

        Calculated values represent the instantaneous Poynting vector for time-domain fields and the
        complex vector for frequency-domain: ``S = 1/2 E × conj(H)``.

        Only the available components are returned, e.g., if the indicated monitor doesn't include
        field component `"Ex"`, then `"Sy"` and `"Sz"` will not be calculated.

        Parameters
        ----------
        field_monitor_data: AbstractFieldData
            Field monitor data from which to extract Poynting vector.

        Returns
        -------
        xarray.DataArray
            DataArray containing the Poynting vector calculated based on the field components
            colocated at the center locations of the Yee grid.
        """
        field_dataset = self._at_boundaries(field_monitor_data)

        time_domain = isinstance(field_monitor_data, FieldTimeData)

        poynting_components = {}

        dims = "xyz"
        for axis, dim in enumerate(dims):
            dim_1 = dims[axis - 2]
            dim_2 = dims[axis - 1]

            required_components = [f + c for f in "EH" for c in (dim_1, dim_2)]
            if not all(field_cmp in field_dataset for field_cmp in required_components):
                continue

            e_1 = field_dataset.data_vars["E" + dim_1]
            e_2 = field_dataset.data_vars["E" + dim_2]
            h_1 = field_dataset.data_vars["H" + dim_1]
            h_2 = field_dataset.data_vars["H" + dim_2]
            poynting_components["S" + dim] = (
                e_1 * h_2 - e_2 * h_1
                if time_domain
                else 0.5 * (e_1 * h_2.conj() - e_2 * h_1.conj())
            )

            # 2D monitors have grid correction factors that can be different from 1. For Poynting,
            # it is always the product of a primal-located field and dual-located field, so the
            # total grid correction factor is the product of the two
            grid_correction = (
                field_monitor_data.grid_dual_correction * field_monitor_data.grid_primal_correction
            )
            poynting_components["S" + dim] *= grid_correction

        return xr.Dataset(poynting_components)

    def get_poynting_vector(self, field_monitor_name: str) -> xr.Dataset:
        """return ``xarray.Dataset`` of the Poynting vector at Yee cell centers.

        Calculated values represent the instantaneous Poynting vector for time-domain fields and the
        complex vector for frequency-domain: ``S = 1/2 E × conj(H)``.

        Only the available components are returned, e.g., if the indicated monitor doesn't include
        field component `"Ex"`, then `"Sy"` and `"Sz"` will not be calculated.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.DataArray
            DataArray containing the Poynting vector calculated based on the field components
            colocated at the center locations of the Yee grid.
        """
        field_monitor_data = self.load_field_monitor(field_monitor_name)
        return self._get_poynting_vector(field_monitor_data=field_monitor_data)

    def _get_scalar_field(
        self,
        field_monitor_name: str,
        field_name: str,
        val: FieldVal,
        phase: float = 0.0,
    ):
        """return ``xarray.DataArray`` of the scalar field of a given monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.
        field_name : str
            Name of the derived field component: one of `('E', 'H', 'S', 'Sx', 'Sy', 'Sz')`.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        phase : float = 0.0
            Optional phase to apply to result

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """
        field_monitor_data = self.load_field_monitor(field_monitor_name)
        return self._get_scalar_field_from_data(
            field_monitor_data, field_name=field_name, val=val, phase=phase
        )

    def _get_scalar_field_from_data(
        self,
        field_monitor_data: AbstractFieldData,
        field_name: str,
        val: FieldVal,
        phase: float = 0.0,
    ):
        """return ``xarray.DataArray`` of the scalar field of a given monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_data : AbstractFieldData
            Field monitor data from which to extract scalar field.
        field_name : str
            Name of the derived field component: one of `('E', 'H', 'S', 'Sx', 'Sy', 'Sz')`.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        phase : float = 0.0
            Optional phase to apply to result

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """

        if field_name[0] == "S":
            dataset = self._get_poynting_vector(field_monitor_data)
            if len(field_name) > 1:
                if field_name in dataset:
                    derived_data = dataset[field_name]
                    derived_data.name = field_name
                    return self._field_component_value(derived_data, val)
                raise Tidy3dKeyError(f"Poynting component {field_name} not available")
        else:
            dataset = self._at_boundaries(field_monitor_data)

        dataset = self.apply_phase(data=dataset, phase=phase)

        if field_name in ("E", "H", "S"):
            # Gather vector components
            required_components = [field_name + c for c in "xyz"]
            if not all(field_cmp in dataset for field_cmp in required_components):
                raise DataError(
                    f"Field monitor must contain '{field_name}x', '{field_name}y', and "
                    f"'{field_name}z' fields to compute '{field_name}'."
                )
            field_components = (dataset[c] for c in required_components)

            # Apply the requested transformation
            val = val.lower()
            if val in ("real", "re"):
                derived_data = sum(f.real**2 for f in field_components) ** 0.5
                derived_data.name = f"|Re{{{field_name}}}|"

            elif val in ("imag", "im"):
                derived_data = sum(f.imag**2 for f in field_components) ** 0.5
                derived_data.name = f"|Im{{{field_name}}}|"

            elif val == "abs":
                derived_data = sum(abs(f) ** 2 for f in field_components) ** 0.5
                derived_data.name = f"|{field_name}|"

            elif val == "abs^2":
                derived_data = sum(abs(f) ** 2 for f in field_components)
                if hasattr(derived_data, "name"):
                    derived_data.name = f"|{field_name}|²"

            elif val == "phase":
                raise Tidy3dKeyError(f"Phase is not defined for complex vector {field_name}")

            else:
                raise Tidy3dKeyError(
                    f"'val' of {val} not supported. "
                    "Must be one of 'real', 'imag', 'abs', 'abs^2', or 'phase'."
                )

            return derived_data

        raise Tidy3dKeyError(
            f"Derived field name must be one of 'E', 'H', 'S', 'Sx', 'Sy', or 'Sz', received "
            f"'{field_name}'."
        )

    def get_intensity(self, field_monitor_name: str) -> xr.DataArray:
        """return `xarray.DataArray` of the intensity of a field monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """
        return self._get_scalar_field(
            field_monitor_name=field_monitor_name, field_name="E", val="abs^2"
        )

    @classmethod
    def mnt_data_from_file(cls, fname: str, mnt_name: str, **parse_obj_kwargs) -> MonitorDataType:
        """Loads data for a specific monitor from a .hdf5 file with data for a ``SimulationData``.

        Parameters
        ----------
        fname : str
            Full path to an hdf5 file containing :class:`.SimulationData` data.
        mnt_name : str, optional
            ``.name`` of the monitor to load the data from.
        **parse_obj_kwargs
            Keyword arguments passed to either pydantic's ``parse_obj`` function when loading model.

        Returns
        -------
        :class:`MonitorData`
            Monitor data corresponding to the ``mnt_name`` type.

        Example
        -------
        >>> field_data = your_simulation_data.from_file(fname='folder/data.hdf5', mnt_name="field") # doctest: +SKIP
        """

        if pathlib.Path(fname).suffix != ".hdf5":
            raise ValueError("'mnt_data_from_file' only works with '.hdf5' files.")

        # open file and ensure it has data
        with h5py.File(fname) as f_handle:
            if "data" not in f_handle:
                raise ValueError(f"could not find data in the supplied file {fname}")

            # get the monitor list from the json string
            json_string = f_handle[JSON_TAG][()]
            json_dict = json.loads(json_string)
            monitor_list = json_dict["simulation"]["monitors"]

            # loop through data
            for monitor_index_str, _mnt_data in f_handle["data"].items():
                # grab the monitor data for this data element
                monitor_dict = monitor_list[int(monitor_index_str)]

                # if a match on the monitor name
                if monitor_dict["name"] == mnt_name:
                    # try to grab the monitor data type
                    monitor_type_str = monitor_dict["type"]
                    if monitor_type_str not in DATA_TYPE_NAME_MAP:
                        raise ValueError(f"Could not find data type '{monitor_type_str}'.")
                    monitor_data_type = DATA_TYPE_NAME_MAP[monitor_type_str]

                    # load the monitor data from the file using the group_path
                    group_path = f"data/{monitor_index_str}"
                    return monitor_data_type.from_file(
                        fname, group_path=group_path, **parse_obj_kwargs
                    )

        raise ValueError(f"No monitor with name '{mnt_name}' found in data file.")

    @staticmethod
    def apply_phase(data: Union[xr.DataArray, xr.Dataset], phase: float = 0.0) -> xr.DataArray:
        """Apply a phase to xarray data."""
        if phase != 0.0:
            if np.any(np.iscomplex(data.values)):
                data *= np.exp(1j * phase)
            else:
                log.warning(
                    f"Non-zero phase of {phase} specified but the data being plotted is "
                    "real-valued. The phase will be ignored in the plot."
                )
        return data

    def plot_field_monitor_data(
        self,
        field_monitor_data: AbstractFieldData,
        field_name: str,
        val: FieldVal = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        phase: float = 0.0,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        shading: str = "flat",
        **sel_kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_data : AbstractFieldData
            Field monitor data to plot.
        field_name : str
            Name of ``field`` component to plot (eg. `'Ex'`).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        phase : float = 0.0
            Optional phase (radians) to apply to the fields.
            Only has an effect on frequency-domain fields.
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        shading: str = 'flat'
            Shading argument for Xarray plot method ('flat','nearest','goraud')
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or ``mode_index``, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        # get the DataArray corresponding to the monitor_name and field_name
        # deprecated intensity
        if field_name == "int":
            log.warning(
                "'int' field name is deprecated and will be removed in the future. Please use "
                "field_name='E' and val='abs^2' for the same effect."
            )
            field_name = "E"
            val = "abs^2"

        if field_name in ("E", "H") or field_name[0] == "S":
            # Derived fields
            field_data = self._get_scalar_field_from_data(
                field_monitor_data, field_name, val, phase=phase
            )
        else:
            # Direct field component (e.g. Ex)
            if field_name not in field_monitor_data.field_components:
                raise DataError(f"field_name '{field_name}' not found in data.")
            field_component = field_monitor_data.field_components[field_name]
            field_component.name = field_name
            field_component = self.apply_phase(data=field_component, phase=phase)
            field_data = self._field_component_value(field_component, val)

        if scale == "dB":
            if val == "phase":
                log.warning("Plotting phase component in log scale masks the phase sign.")
            db_factor = {
                ("S", "real"): 10,
                ("S", "imag"): 10,
                ("S", "abs"): 10,
                ("S", "abs^2"): 5,
                ("S", "phase"): 1,
                ("E", "abs^2"): 10,
                ("H", "abs^2"): 10,
            }.get((field_name[0], val), 20)
            field_data = db_factor * np.log10(np.abs(field_data))
            field_data.name += " (dB)"
            cmap_type = "sequential"
        else:
            cmap_type = (
                "cyclic"
                if val == "phase"
                else (
                    "divergent"
                    if len(field_name) == 2 and val in ("real", "imag", "re", "im")
                    else "sequential"
                )
            )

        # interp out any monitor.size==0 dimensions
        monitor = field_monitor_data.monitor
        thin_dims = {
            "xyz"[dim]: monitor.center[dim]
            for dim in range(3)
            if monitor.size[dim] == 0 and "xyz"[dim] not in sel_kwargs
        }
        for axis, pos in thin_dims.items():
            if axis not in field_data.coords:
                continue
            if field_data.coords[axis].size <= 1:
                field_data = field_data.sel(**{axis: pos}, method="nearest")
            else:
                field_data = field_data.interp(**{axis: pos}, kwargs=dict(bounds_error=True))

        # warn about new API changes and replace the values
        if "freq" in sel_kwargs:
            log.warning(
                "'freq' supplied to 'plot_field', frequency selection key renamed to 'f' and "
                "'freq' will error in future release, please update your local script to use "
                "'f=value'."
            )
            sel_kwargs["f"] = sel_kwargs.pop("freq")
        if "time" in sel_kwargs:
            log.warning(
                "'time' supplied to 'plot_field', frequency selection key renamed to 't' and "
                "'time' will error in future release, please update your local script to use "
                "'t=value'."
            )
            sel_kwargs["t"] = sel_kwargs.pop("time")

        # select the extra coordinates out of the data from user-specified kwargs
        for coord_name, coord_val in sel_kwargs.items():
            if (
                field_data.coords[coord_name].size <= 1
                or coord_name == "eme_port_index"
                or coord_name == "eme_cell_index"
                or coord_name == "sweep_index"
                or coord_name == "mode_index"
            ):
                field_data = field_data.sel(**{coord_name: coord_val}, method=None)
            else:
                field_data = field_data.interp(
                    **{coord_name: coord_val}, kwargs=dict(bounds_error=True)
                )

        # before dropping coordinates, check if a frequency can be derived from the data that can
        # be used to plot material permittivity
        if "f" in sel_kwargs:
            freq_eps_eval = sel_kwargs["f"]
        elif "f" in field_data.coords:
            freq_eps_eval = field_data.coords["f"].values[0]
        else:
            freq_eps_eval = None

        field_data = field_data.squeeze(drop=True)
        non_scalar_coords = {name: c for name, c in field_data.coords.items() if c.size > 1}

        # assert the data is valid for plotting
        if len(non_scalar_coords) != 2:
            raise DataError(
                f"Data after selection has {len(non_scalar_coords)} coordinates "
                f"({list(non_scalar_coords.keys())}), "
                "must be 2 spatial coordinates for plotting on plane. "
                "Please add keyword arguments to `plot_field()` to select out the other coords."
            )

        spatial_coords_in_data = {
            coord_name: (coord_name in non_scalar_coords) for coord_name in "xyz"
        }

        if sum(spatial_coords_in_data.values()) != 2:
            raise DataError(
                "All coordinates in the data after selection must be spatial (x, y, z), "
                f" given {non_scalar_coords.keys()}."
            )

        # get the spatial coordinate corresponding to the plane
        planar_coord = [name for name, c in spatial_coords_in_data.items() if c is False][0]
        axis = "xyz".index(planar_coord)
        if planar_coord in field_data.coords:
            position = float(field_data.coords[planar_coord])
        else:
            position = monitor.center[axis]

        return self.plot_scalar_array(
            field_data=field_data,
            axis=axis,
            position=position,
            freq=freq_eps_eval,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            cmap_type=cmap_type,
            ax=ax,
            shading=shading,
            infer_intervals=True if shading == "flat" else False,
        )

    def plot_field(
        self,
        field_monitor_name: str,
        field_name: str,
        val: FieldVal = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        phase: float = 0.0,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        shading: str = "flat",
        **sel_kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.FieldMonitor`, :class:`.FieldTimeData`, or :class:`.ModeSolverData`
            to plot.
        field_name : str
            Name of ``field`` component to plot (eg. `'Ex'`).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        phase : float = 0.0
            Optional phase (radians) to apply to the fields.
            Only has an effect on frequency-domain fields.
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        shading: str = 'flat'
            Shading argument for Xarray plot method ('flat','nearest','goraud')
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or ``mode_index``, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        field_monitor_data = self.load_field_monitor(field_monitor_name)

        return self.plot_field_monitor_data(
            field_monitor_data=field_monitor_data,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            phase=phase,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            shading=shading,
            **sel_kwargs,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_scalar_array(
        self,
        field_data: xr.DataArray,
        axis: Axis,
        position: float,
        freq: float = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        cmap_type: ColormapType = "divergent",
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_data: xr.DataArray
            DataArray with the field data to plot.
            Must be a scalar field.
        axis: Axis
            Axis normal to the plotting plane.
        position: float
            Position along the axis.
        freq: float = None
            Frequency at which the permittivity is evaluated at (if dispersive).
            By default, chooses permittivity as frequency goes to infinity.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        cmap_type : Literal["divergent", "sequential", "cyclic"] = "divergent"
            Type of color map to use for plotting.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        **kwargs : Extra arguments to ``DataArray.plot``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}

        if cmap_type == "divergent":
            cmap = "RdBu"
            center = 0.0
            eps_reverse = False
        elif cmap_type == "sequential":
            cmap = "magma"
            center = False
            eps_reverse = True
        elif cmap_type == "cyclic":
            cmap = "twilight"
            vmin = -np.pi
            vmax = np.pi
            center = False
            eps_reverse = False

        # plot the field
        xy_coord_labels = list("xyz")
        xy_coord_labels.pop(axis)
        x_coord_label, y_coord_label = xy_coord_labels[0], xy_coord_labels[1]
        field_data.plot(
            ax=ax,
            x=x_coord_label,
            y=y_coord_label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            center=center,
            cbar_kwargs={"label": field_data.name},
            **kwargs,
        )

        # plot the simulation epsilon
        ax = self.simulation.plot_structures_eps(
            freq=freq,
            cbar=False,
            alpha=eps_alpha,
            reverse=eps_reverse,
            ax=ax,
            **interp_kwarg,
        )

        # set the limits based on the xarray coordinates min and max
        x_coord_values = field_data.coords[x_coord_label]
        y_coord_values = field_data.coords[y_coord_label]
        ax.set_xlim(min(x_coord_values), max(x_coord_values))
        ax.set_ylim(min(y_coord_values), max(y_coord_values))

        return ax


class SimulationData(AbstractYeeGridSimulationData):
    """Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`.

    Notes
    -----

        The ``SimulationData`` objects store a copy of the original :class:`.Simulation`:, so it can be recovered if the
        ``SimulationData`` is loaded in a new session and the :class:`.Simulation` is no longer in memory.

        More importantly, the ``SimulationData`` contains a reference to the data for each of the monitors within the
        original :class:`.Simulation`. This data can be accessed directly using the name given to the monitors initially.

    Examples
    --------

    Standalone example:

    >>> import tidy3d as td
    >>> num_modes = 5
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = td.Grid(boundaries=td.Coords(x=x, y=y, z=z))
    >>> scalar_field = td.ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> field_monitor = td.FieldMonitor(
    ...     size=(2,4,6),
    ...     freqs=[2e14, 3e14],
    ...     name='field',
    ...     fields=['Ex'],
    ...     colocate=True,
    ... )
    >>> sim = td.Simulation(
    ...     size=(2, 4, 6),
    ...     grid_spec=td.GridSpec(wavelength=1.0),
    ...     monitors=[field_monitor],
    ...     run_time=2e-12,
    ...     sources=[
    ...         td.UniformCurrentSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             source_time=td.GaussianPulse(
    ...                 freq0=2e14,
    ...                 fwidth=4e13,
    ...             ),
    ...         )
    ...     ],
    ... )
    >>> field_data = td.FieldData(monitor=field_monitor, Ex=scalar_field, grid_expanded=grid)
    >>> sim_data = td.SimulationData(simulation=sim, data=(field_data,))

    To save and load the :class:`SimulationData` object.

    .. code-block:: python

        sim_data.to_file(fname='path/to/file.hdf5') # Save a SimulationData object to a HDF5 file
        sim_data = SimulationData.from_file(fname='path/to/file.hdf5') # Load a SimulationData object from a HDF5 file.

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Performing visualization of simulation data <../../notebooks/VizData.html>`_
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

    """

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`.Simulation` associated with the data.",
    )

    data: Tuple[annotate_type(MonitorDataType), ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

    diverged: bool = pd.Field(
        False,
        title="Diverged",
        description="A boolean flag denoting whether the simulation run diverged.",
    )

    @property
    def final_decay_value(self) -> float:
        """Returns value of the field decay at the final time step."""
        log_str = self.log
        if log_str is None:
            raise DataError(
                "No log string in the SimulationData object, can't find final decay value."
            )
        lines = log_str.split("\n")
        decay_lines = [line for line in lines if "field decay" in line]
        final_decay = 1.0
        if len(decay_lines) > 0:
            final_decay_line = decay_lines[-1]
            final_decay = float(final_decay_line.split("field decay: ")[-1])
        return final_decay

    def source_spectrum(self, source_index: int) -> Callable:
        """Get a spectrum normalization function for a given source index."""

        if source_index is None or len(self.simulation.sources) == 0:
            return np.ones_like

        source = self.simulation.sources[source_index]
        source_time = source.source_time
        times = self.simulation.tmesh
        dt = self.simulation.dt

        # plug in mornitor_data frequency domain information
        def source_spectrum_fn(freqs):
            """Source amplitude as function of frequency."""
            spectrum = source_time.spectrum(times, freqs, dt)

            # Remove user defined amplitude and phase from the normalization
            # such that they would still have an effect on the output fields.
            # In other words, we are only normalizing out the arbitrary part of the spectrum
            # that depends on things like freq0, fwidth and offset.
            return spectrum / source_time.amplitude / np.exp(1j * source_time.phase)

        return source_spectrum_fn

    def renormalize(self, normalize_index: int) -> SimulationData:
        """Return a copy of the :class:`.SimulationData` with a different source used for the
        normalization."""

        num_sources = len(self.simulation.sources)
        if normalize_index == self.simulation.normalize_index or num_sources == 0:
            # already normalized to that index
            return self.copy()

        if normalize_index and (normalize_index < 0 or normalize_index >= num_sources):
            # normalize index out of bounds for source list
            raise DataError(
                f"normalize_index {normalize_index} out of bounds for list of sources "
                f"of length {num_sources}"
            )

        def source_spectrum_fn(freqs):
            """Normalization function that also removes previous normalization if needed."""
            new_spectrum_fn = self.source_spectrum(normalize_index)
            old_spectrum_fn = self.source_spectrum(self.simulation.normalize_index)
            return new_spectrum_fn(freqs) / old_spectrum_fn(freqs)

        # Make a new monitor_data dictionary with renormalized data
        data_normalized = [mnt_data.normalize(source_spectrum_fn) for mnt_data in self.data]

        simulation = self.simulation.copy(update=dict(normalize_index=normalize_index))

        return self.copy(update=dict(simulation=simulation, data=data_normalized))

    def split_adjoint_data(self: SimulationData, num_mnts_original: int) -> tuple[list, list]:
        """Split data list into original, adjoint field, and adjoint permittivity."""

        data_all = list(self.data)
        num_mnts_adjoint = (len(data_all) - num_mnts_original) // 2

        log.info(
            f" -> {num_mnts_original} monitors, {num_mnts_adjoint} adjoint field monitors, {num_mnts_adjoint} adjoint eps monitors."
        )

        data_original, data_adjoint = split_list(data_all, index=num_mnts_original)

        return data_original, data_adjoint

    def split_original_fwd(self, num_mnts_original: int) -> Tuple[SimulationData, SimulationData]:
        """Split this simulation data into original and fwd data from number of original mnts."""

        # split the data and monitors into the original ones & adjoint gradient ones (for 'fwd')
        data_original, data_fwd = self.split_adjoint_data(num_mnts_original=num_mnts_original)
        monitors_orig, monitors_fwd = split_list(self.simulation.monitors, index=num_mnts_original)

        # reconstruct the simulation data for the user, using original sim, and data for original mnts
        sim_original = self.simulation.updated_copy(monitors=monitors_orig)
        sim_data_original = self.updated_copy(
            simulation=sim_original,
            data=data_original,
            deep=False,
        )

        # construct the 'forward' simulation and its data, which is only used for for gradient calc.
        sim_fwd = self.simulation.updated_copy(monitors=monitors_fwd)
        sim_data_fwd = self.updated_copy(
            simulation=sim_fwd,
            data=data_fwd,
            deep=False,
        )

        return sim_data_original, sim_data_fwd

    def make_adjoint_sim(
        self, data_vjp_paths: set[tuple], adjoint_monitors: list[Monitor]
    ) -> Simulation:
        """Make the adjoint simulation from the original simulation and the VJP-containing data."""

        sim_original = self.simulation

        # generate the adjoint sources {mnt_name : list[Source]}
        sources_adj_dict = self.make_adjoint_sources(data_vjp_paths=data_vjp_paths)
        adj_srcs = []
        for src_list in sources_adj_dict.values():
            adj_srcs += list(src_list)

        adjoint_source_info = self.process_adjoint_sources(adj_srcs=adj_srcs)

        # grab boundary conditions with flipped Bloch vectors (for adjoint)
        bc_adj = sim_original.boundary_spec.flipped_bloch_vecs

        # fields to update the 'fwd' simulation with to make it 'adj'
        sim_adj_update_dict = dict(
            sources=adjoint_source_info.sources,
            boundary_spec=bc_adj,
            monitors=adjoint_monitors,
            post_norm=adjoint_source_info.post_norm,
        )

        if not adjoint_source_info.normalize_sim:
            sim_adj_update_dict["normalize_index"] = None

        # set the ADJ grid spec wavelength to the original wavelength (for same meshing)
        grid_spec_original = sim_original.grid_spec
        if sim_original.sources and grid_spec_original.wavelength is None:
            wavelength_original = grid_spec_original.wavelength_from_sources(sim_original.sources)
            grid_spec_adj = grid_spec_original.updated_copy(wavelength=wavelength_original)
            sim_adj_update_dict["grid_spec"] = grid_spec_adj

        return sim_original.updated_copy(**sim_adj_update_dict)

    def make_adjoint_sources(self, data_vjp_paths: set[tuple]) -> dict[str, SourceType]:
        """Generate all of the non-zero sources for the adjoint simulation given the VJP data."""

        # map of index into 'self.data' to the list of datasets we need adjoint sources for
        adj_src_map = defaultdict(list)
        for _, index, dataset_name in data_vjp_paths:
            adj_src_map[index].append(dataset_name)

        # gather a dict of adjoint sources for every monitor data in the VJP that needs one
        sources_adj_all = defaultdict(list)
        for data_index, dataset_names in adj_src_map.items():
            mnt_data = self.data[data_index]
            sources_adj = mnt_data.make_adjoint_sources(
                dataset_names=dataset_names, fwidth=self.fwidth_adj
            )
            sources_adj_all[mnt_data.monitor.name] = sources_adj

        if not any(src for _, src in sources_adj_all.items()):
            raise ValueError(
                "No adjoint sources created for this simulation. "
                "This could indicate a bug in your setup, for example the objective function "
                "output depending on a monitor that is not supported. If you encounter this error, "
                "please examine your set up or contact customer support if you need more help."
            )

        return sources_adj_all

    @property
    def fwidth_adj(self) -> float:
        # fwidth of forward pass, try as default for adjoint
        normalize_index_fwd = self.simulation.normalize_index or 0
        return self.simulation.sources[normalize_index_fwd].source_time.fwidth

    def process_adjoint_sources(self, adj_srcs: list[SourceType]) -> AdjointSourceInfo:
        """Compute list of final sources along with a post run normalization for adj fields."""

        # dictionary mapping hash of sources with same freq dependence to list of time-dependencies
        hashes_to_sources = defaultdict(None)
        hashes_to_src_times = defaultdict(list)

        tmp_src_time = GaussianPulse(freq0=C_0, fwidth=inf)
        for src in adj_srcs:
            tmp_src = src.updated_copy(source_time=tmp_src_time)
            tmp_src_hash = tmp_src._hash_self()
            hashes_to_sources[tmp_src_hash] = src
            hashes_to_src_times[tmp_src_hash].append(src.source_time)

        num_ports = len(hashes_to_src_times)
        num_unique_freqs = len({src.source_time.freq0 for src in adj_srcs})

        # next, figure out which treatment / normalization to apply
        if num_unique_freqs == 1:
            log.info("Adjoint source creation: one unique frequency, no normalization.")
            return AdjointSourceInfo(sources=adj_srcs, post_norm=1.0, normalize_sim=True)

        if num_ports == 1 and len(adj_srcs) == num_unique_freqs:
            log.info("Adjoint source creation: one spatial port detected.")
            adj_srcs, post_norm = self.process_adjoint_sources_broadband(adj_srcs)
            return AdjointSourceInfo(sources=adj_srcs, post_norm=post_norm, normalize_sim=True)

        # if several spatial ports and several frequencies, try to fit
        log.info("Adjoint source creation: trying multifrequency fit.")
        adj_srcs, post_norm = self.process_adjoint_sources_fit(
            adj_srcs=adj_srcs,
            hashes_to_src_times=hashes_to_src_times,
            hashes_to_sources=hashes_to_sources,
        )
        return AdjointSourceInfo(sources=adj_srcs, post_norm=post_norm, normalize_sim=False)

    """ SIMPLE APPROACH """

    def process_adjoint_sources_broadband(
        self, adj_srcs: list[SourceType]
    ) -> tuple[list[SourceType], xr.DataArray]:
        """Process adjoint sources for the case of several sources at the same freq."""

        src_broadband = self._make_broadband_source(adj_srcs=adj_srcs)
        post_norm_amps = self._make_post_norm_amps(adj_srcs=adj_srcs)

        log.info(
            "Several adjoint sources, from one monitor. "
            "Only difference between them is the source time. "
            "Constructing broadband adjoint source and performing post-run normalization "
            f"of fields with {len(post_norm_amps)} frequencies."
        )

        return [src_broadband], post_norm_amps

    def _make_broadband_source(self, adj_srcs: list[SourceType]) -> SourceType:
        """Make a broadband source for a set of adjoint sources."""

        source_index = self.simulation.normalize_index or 0
        src_time_base = self.simulation.sources[source_index].source_time.copy()
        src_broadband = adj_srcs[0].updated_copy(source_time=src_time_base)

        return src_broadband

    @staticmethod
    def _make_post_norm_amps(adj_srcs: list[SourceType]) -> xr.DataArray:
        """Make a ``DataArray`` containing the complex amplitudes to multiply with adjoint field."""

        freqs = []
        amps_complex = []
        for src in adj_srcs:
            src_time = src.source_time
            freqs.append(src_time.freq0)
            amp_complex = src_time.amplitude * np.exp(1j * src_time.phase)
            amps_complex.append(amp_complex)

        coords = dict(f=freqs)
        amps_complex = np.array(amps_complex)
        return xr.DataArray(amps_complex, coords=coords)

    """ FITTING APPROACH """

    def process_adjoint_sources_fit(
        self,
        adj_srcs: list[SourceType],
        hashes_to_src_times: dict[str, GaussianPulse],
        hashes_to_sources: dict[str, list[SourceType]],
    ) -> tuple[list[SourceType], float]:
        """Process the adjoint sources using a least squared fit to the derivative data."""

        raise NotImplementedError(
            "Can't perform multi-frequency autograd with several adjoint sources yet. "
            "In the meantime, please construct a single 'Simulation' per output data "
            "(can be multi-frequency) and run in parallel using 'web.run_async'. For example, "
            "if your problem has 'P' outuput ports, e.g. waveguides, please make a 'Simulation' "
            "corresponding to the objective function contribution at each port."
        )

        # new adjoint sources
        new_adj_srcs = []
        for src_hash, source_times in hashes_to_src_times.items():
            src = hashes_to_sources[src_hash]
            new_sources = self.correct_adjoint_sources(
                src=src, fwidth=self.fwidth_adj, source_times=source_times
            )
            new_adj_srcs += new_sources

        # compute amplitudes of each adjoint source, and the norm
        adj_src_amps = []
        for src in new_adj_srcs:
            amp = src.source_time.amp_complex
            adj_src_amps.append(amp)
        norm_amps = np.linalg.norm(adj_src_amps)

        # normalize all of the adjoint sources by this and return the normalization term used
        adj_srcs_norm = []
        for src in new_adj_srcs:
            src_time = src.source_time
            amp = src_time.amp_complex
            src_time_norm = src_time.from_amp_complex(amp=amp / norm_amps)
            src_nrm = src.updated_copy(source_time=src_time_norm)
            adj_srcs_norm.append(src_nrm)

        return adj_srcs_norm, norm_amps

    def correct_adjoint_sources(
        self, src: SourceType, fwidth: float, source_times: list[GaussianPulse]
    ) -> [SourceType]:
        """Corret a set of spectrally overlapping adjoint sources to give correct E_adj."""

        freqs = [st.freq0 for st in source_times]
        times = self.simulation.tmesh
        dt = self.simulation.dt

        def get_spectrum(source_time: GaussianPulse, freqs: list[float]) -> complex:
            """Get the spectrum of a source time at a given frequency."""
            return source_time.spectrum(times=times, freqs=freqs, dt=dt)

        # compute matrix coupling the spectra of Gaussian pulses centered at each adjoint freq
        def get_coupling_matrix(fwidth: float) -> np.ndarray:
            """Matrix coupling the spectra of Gaussian pulses centered at each adjoint freq."""

            return np.array(
                [
                    get_spectrum(
                        source_time=GaussianPulse(freq0=source_time.freq0, fwidth=fwidth),
                        freqs=freqs,
                    )
                    for source_time in source_times
                ]
            ).T

        amps_adj = np.array([src_time.amp_complex for src_time in source_times])

        # compute the corrected set of amps to inject at each freq to take coupling into account
        def get_amps_corrected(fwidth: float) -> tuple[np.ndarray, float]:
            """New set of new adjoint source amps that generate the desired response at each f."""
            J_coupling = get_coupling_matrix(fwidth=fwidth)

            amps_adj_new, *info = np.linalg.lstsq(J_coupling, amps_adj, rcond=None)
            # amps_adj_new = np.linalg.solve(J_coupling, amps_adj)
            residual = J_coupling @ amps_adj_new - amps_adj
            residual_norm = np.linalg.norm(residual) / np.linalg.norm(amps_adj)
            return amps_adj_new, residual_norm

        # get the corrected amplitudes
        amps_corrected, res_norm = get_amps_corrected(self.fwidth_adj)

        if res_norm > RESIDUAL_CUTOFF_ADJOINT:
            raise ValueError(
                f"Residual of {res_norm:.5e} found when trying to fit adjoint source spectrum. "
                f"This is above our accuracy cutoff of {RESIDUAL_CUTOFF_ADJOINT:.5e} and therefore "
                "we are not able to process this adjoint simulation in a broadband way. "
                "To fix, split your simulation into a set of simulations, one for each port, and "
                "run parallel, broadband simulations using 'web.run_async'. "
            )

        # construct the new adjoint sources with the corrected amplitudes
        src_times_corrected = [
            src_time.from_amp_complex(amp=amp, fwidth=self.fwidth_adj)
            for src_time, amp in zip(source_times, amps_corrected)
        ]
        srcs_corrected = []
        for src_time in src_times_corrected:
            src_new = src.updated_copy(source_time=src_time)
            srcs_corrected.append(src_new)

        return srcs_corrected

    def get_adjoint_data(self, structure_index: int, data_type: str) -> MonitorDataType:
        """Grab the field or permittivity data for a given structure index."""

        monitor_name = Structure.get_monitor_name(index=structure_index, data_type=data_type)
        return self[monitor_name]

    def to_mat_file(self, fname: str, **kwargs):
        """Output the ``SimulationData`` object as ``.mat`` MATLAB file.

        Parameters
        ----------
        fname : str
            Full path to the output file. Should include ``.mat`` file extension.
        **kwargs : dict, optional
            Extra arguments to ``scipy.io.savemat``: see ``scipy`` documentation for more detail.

        Example
        -------
        >>> simData.to_mat_file('/path/to/file/data.mat') # doctest: +SKIP
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

        # Remove NoneType values from dict
        # Built from theory discussed in https://github.com/scipy/scipy/issues/3488
        modified_sim_dict = replace_values(sim_dict, None, [])

        try:
            from scipy.io import savemat

            savemat(fname, modified_sim_dict, **kwargs)
        except Exception as e:
            raise ValueError(
                "Could not save supplied 'SimulationData' to file. As this is an experimental feature, we may not be able to support the contents of your dataset. If you receive this error, please feel free to raise an issue on our front end repository so we can investigate."
            ) from e
