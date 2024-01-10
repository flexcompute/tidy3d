"""EME simulation data"""
from __future__ import annotations

from typing import Tuple, Union, Optional

import pydantic.v1 as pd
import numpy as np
import xarray as xr

from ...base_sim.data.sim_data import AbstractSimulationData
from ..simulation import EMESimulation
from .monitor_data import EMEMonitorDataType
from .dataset import EMESMatrixDataset
from ...data.data_array import EMESMatrixDataArray

from ....log import log
from ....exceptions import Tidy3dKeyError
from ...types import Ax, Axis, FieldVal, PlotScale, ColormapType
from ...viz import equal_aspect, add_ax_if_none
from ....exceptions import DataError, SetupError
from ...data.monitor_data import AbstractFieldData, ModeSolverData, FieldData, ModeData


class EMESimulationData(AbstractSimulationData):
    """Data associated with an EME simulation."""

    simulation: EMESimulation = pd.Field(
        ..., title="EME simulation", description="EME simulation associated with this data."
    )

    data: Tuple[EMEMonitorDataType, ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of EME monitor data "
        "associated with the monitors of the original :class:`.EMESimulation`.",
    )

    smatrix: EMESMatrixDataset = pd.Field(
        ..., title="S Matrix", description="Scattering matrix of the EME simulation."
    )

    port_modes: Optional[Tuple[ModeSolverData, ModeSolverData]] = pd.Field(
        ...,
        title="Port Modes",
        description="Modes associated with the two ports of the EME device. "
        "The scattering matrix is expressed in this basis.",
    )

    def smatrix_in_basis(
        self, modes1: Union[FieldData, ModeData], modes2: Union[FieldData, ModeData]
    ) -> EMESMatrixDataset:
        """Express the scattering matrix in the provided basis.
        Change of basis is done by computing overlaps between provided modes and port modes.

        Parameters
        ----------
        modes1: Union[FieldData, ModeData]
            New modal basis for port 1.
        modes2: Union[FieldData, ModeData]
            New modal basis for port 2.

        Returns
        -------
        :class:`.EMESMatrixDataset`
            The scattering matrix of the EME simulation, but expressed in the basis
            of the provided modes, rather than in the basis of ``port_modes`` used
            in computation.
        """

        if self.port_modes is None:
            raise SetupError(
                "Cannot convert the EME scattering matrix to the provided "
                "basis, because 'port_modes' is 'None'. Please set 'store_port_modes' "
                "to 'True' and re-run the simulation."
            )

        overlaps1 = modes1.outer_dot(self.port_modes[0])
        overlaps2 = modes2.outer_dot(self.port_modes[1])

        f = np.array(sorted(set(overlaps1.f.values).intersection(overlaps2.f.values)))
        isel1 = [list(overlaps1.f.values).index(freq) for freq in f]
        isel2 = [list(overlaps2.f.values).index(freq) for freq in f]
        overlaps1 = overlaps1.isel(f=isel1)
        overlaps2 = overlaps2.isel(f=isel2)

        modes_in_1 = "mode_index_0" in overlaps1.coords
        modes_in_2 = "mode_index_0" in overlaps2.coords

        if modes_in_1:
            mode_index_1 = overlaps1.mode_index_0.to_numpy()
        else:
            mode_index_1 = [0]
            overlaps1 = overlaps1.expand_dims(dim={"mode_index_0": mode_index_1}, axis=1)
        if modes_in_2:
            mode_index_2 = overlaps2.mode_index_0.to_numpy()
        else:
            mode_index_2 = [0]
            overlaps2 = overlaps2.expand_dims(dim={"mode_index_0": mode_index_2}, axis=1)

        S11s = []
        S12s = []
        S21s = []
        S22s = []

        for freq in f:
            O1 = overlaps1.sel(f=freq).to_numpy()
            O2 = overlaps2.sel(f=freq).to_numpy()
            S11 = self.smatrix.S11.sel(f=freq).to_numpy()
            S12 = self.smatrix.S12.sel(f=freq).to_numpy()
            S21 = self.smatrix.S21.sel(f=freq).to_numpy()
            S22 = self.smatrix.S22.sel(f=freq).to_numpy()

            S11s.append(O1 @ S11 @ O1.T)
            S12s.append(O1 @ S12 @ O2.T)
            S21s.append(O2 @ S21 @ O1.T)
            S22s.append(O2 @ S22 @ O2.T)

        coords11 = dict(
            f=f,
            mode_index_out=mode_index_1,
            mode_index_in=mode_index_1,
        )
        coords12 = dict(
            f=f,
            mode_index_out=mode_index_1,
            mode_index_in=mode_index_2,
        )
        coords21 = dict(
            f=f,
            mode_index_out=mode_index_2,
            mode_index_in=mode_index_1,
        )
        coords22 = dict(
            f=f,
            mode_index_out=mode_index_2,
            mode_index_in=mode_index_2,
        )
        xrS11 = EMESMatrixDataArray(
            S11s, coords=coords11, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS12 = EMESMatrixDataArray(
            S12s, coords=coords12, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS21 = EMESMatrixDataArray(
            S21s, coords=coords21, dims=("f", "mode_index_out", "mode_index_in")
        )
        xrS22 = EMESMatrixDataArray(
            S22s, coords=coords22, dims=("f", "mode_index_out", "mode_index_in")
        )

        if not modes_in_1:
            xrS11 = xrS11.isel(mode_index_out=0, mode_index_in=0, drop=True)
            xrS12 = xrS12.isel(mode_index_out=0, drop=True)
            xrS21 = xrS21.isel(mode_index_in=0, drop=True)
        if not modes_in_2:
            xrS12 = xrS12.isel(mode_index_in=0, drop=True)
            xrS21 = xrS21.isel(mode_index_out=0, drop=True)
            xrS22 = xrS22.isel(mode_index_out=0, mode_index_in=0, drop=True)

        smatrix = EMESMatrixDataset(S11=xrS11, S12=xrS12, S21=xrS21, S22=xrS22)
        return smatrix

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
        **sel_kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.FieldMonitor`, :class:`.FieldTimeData`, or :class:`.ModeSolverData`
            to plot.
        field_name : str
            Name of `field` component to plot (eg. `'Ex'`).
            Also accepts `'E'` and `'H'` to plot the vector magnitudes of the electric and
            magnetic fields, and `'S'` for the Poynting vector.
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
            The lower bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform `.sel()` selection in the monitor data.
            These kwargs can select over the spatial dimensions (`x`, `y`, `z`),
            frequency or time dimensions (`f`, `t`) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (`x`, `y`, or `z`).

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
            field_data = self._get_scalar_field(field_monitor_name, field_name, val, phase=phase)
        else:
            # Direct field component (e.g. Ex)
            field_monitor_data = self.load_field_monitor(field_monitor_name)
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
                    if len(field_name) == 2 and val in ("real", "imag")
                    else "sequential"
                )
            )

        # interp out any monitor.size==0 dimensions
        monitor = self.simulation.get_monitor_by_name(field_monitor_name)
        thin_dims = {
            "xyz"[dim]: monitor.center[dim]
            for dim in range(3)
            if monitor.size[dim] == 0 and "xyz"[dim] not in sel_kwargs
        }
        for axis, pos in thin_dims.items():
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
            if field_data.coords[coord_name].size <= 1:
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
        position = float(field_data.coords[planar_coord])

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

    def load_field_monitor(self, monitor_name: str) -> AbstractFieldData:
        """Load monitor and raise exception if not a field monitor."""
        mon_data = self[monitor_name]
        if not isinstance(mon_data, AbstractFieldData):
            raise DataError(
                f"data for monitor '{monitor_name}' does not contain field data "
                f"as it is a '{type(mon_data)}'."
            )
        return mon_data

    def _get_scalar_field(
        self, field_monitor_name: str, field_name: str, val: FieldVal, phase: float = 0.0
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

        if field_name[0] == "S":
            dataset = self.get_poynting_vector(field_monitor_name)
            if len(field_name) > 1:
                if field_name in dataset:
                    derived_data = dataset[field_name]
                    derived_data.name = field_name
                    return self._field_component_value(derived_data, val)
                raise Tidy3dKeyError(f"Poynting component {field_name} not available")
        else:
            dataset = self.at_boundaries(field_monitor_name)

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
            if val == "real":
                derived_data = sum(f.real**2 for f in field_components) ** 0.5
                derived_data.name = f"|Re{{{field_name}}}|"

            elif val == "imag":
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

            return derived_data

        raise Tidy3dKeyError(
            f"Derived field name must be one of 'E', 'H', 'S', 'Sx', 'Sy', or 'Sz', received "
            f"'{field_name}'."
        )
