""" Simulation Level Data """
from __future__ import annotations
from typing import Dict, Callable, Tuple

import xarray as xr
import pydantic as pd
import numpy as np

from .monitor_data import MonitorDataTypes, MonitorDataType, AbstractFieldData
from ..base import Tidy3dBaseModel
from ..simulation import Simulation
from ..boundary import BlochBoundary
from ..types import Ax, Axis, annotate_type, Literal
from ..viz import equal_aspect, add_ax_if_none
from ...log import DataError, log, Tidy3dKeyError, ValidationError

DATA_TYPE_MAP = {data.__fields__["monitor"].type_: data for data in MonitorDataTypes}


class SimulationData(Tidy3dBaseModel):
    """Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`.

    Example
    -------
    >>> from tidy3d import ModeSpec, GridSpec, ScalarFieldDataArray, FieldMonitor, FieldData
    >>> num_modes = 5
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> t = [0, 1e-12, 2e-12]
    >>> mode_index = np.arange(num_modes)
    >>> direction = ["+", "-"]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> field_monitor = FieldMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex'])
    >>> sim = Simulation(
    ...     size=(2, 4, 6),
    ...     grid_spec=GridSpec(wavelength=1.0),
    ...     monitors=[field_monitor],
    ...     run_time=2e-12,
    ... )
    >>> field_data = FieldData(monitor=field_monitor, Ex=scalar_field)
    >>> sim_data = SimulationData(simulation=sim, data=(field_data,))
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

    log: str = pd.Field(
        None,
        title="Solver Log",
        description="A string containing the log information from the simulation run.",
    )

    diverged: bool = pd.Field(
        False,
        title="Diverged",
        description="A boolean flag denoting whether the simulation run diverged.",
    )

    def __getitem__(self, monitor_name: str) -> MonitorDataType:
        """Get a :class:`.MonitorData` by name. Apply symmetry if applicable."""
        monitor_data = self.monitor_data[monitor_name]
        return monitor_data.symmetry_expanded_copy

    @property
    def monitor_data(self) -> Dict[str, MonitorDataType]:
        """Dictionary mapping monitor name to its associated :class:`.MonitorData`."""
        return {monitor_data.monitor.name: monitor_data for monitor_data in self.data}

    @pd.validator("data", always=True)
    def data_monitors_match_sim(cls, val, values):
        """Ensure each MonitorData in ``.data`` corresponds to a monitor in ``.simulation``."""
        sim = values.get("simulation")
        if sim is None:
            raise ValidationError("Simulation.simulation failed validation, can't validate data.")
        for mnt_data in val:
            try:
                monitor_name = mnt_data.monitor.name
                sim.get_monitor_by_name(monitor_name)
            except Tidy3dKeyError as exc:
                raise DataError(
                    f"Data with monitor name {monitor_name} supplied "
                    "but not found in the Simulation"
                ) from exc
        return val

    @property
    def final_decay_value(self) -> float:
        """Returns value of the field decay at the final time step."""
        log_str = self.log
        if log_str is None:
            raise DataError(
                "No log string in the SimulationData object, can't find final decay value."
            )
        lines = log_str.split("\n")
        decay_lines = [l for l in lines if "field decay" in l]
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
        user_defined_phase = np.exp(1j * source_time.phase)

        # get boundary information to determine whether to use complex fields
        boundaries = self.simulation.boundary_spec.to_list
        boundaries_1d = [boundary_1d for dim_boundary in boundaries for boundary_1d in dim_boundary]
        complex_fields = any(isinstance(boundary, BlochBoundary) for boundary in boundaries_1d)

        # plug in mornitor_data frequency domain information
        def source_spectrum_fn(freqs):
            """Source amplitude as function of frequency."""
            spectrum = source_time.spectrum(times, freqs, dt, complex_fields)

            # remove user defined phase from normalization so its effect is present in the result
            return spectrum * np.conj(user_defined_phase)

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

    def load_field_monitor(self, monitor_name: str) -> AbstractFieldData:
        """Load monitor and raise exception if not a field monitor."""
        mon_data = self[monitor_name]
        if not isinstance(mon_data, AbstractFieldData):
            raise DataError(
                f"data for monitor '{monitor_name}' does not contain field data "
                f"as it is a `{type(mon_data)}`."
            )
        return mon_data

    def at_centers(self, field_monitor_name: str) -> xr.Dataset:
        """return xarray.Dataset representation of field monitor data
        co-located at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data
            interpolated to center locations on Yee grid.
        """

        # get the data
        monitor_data = self.load_field_monitor(field_monitor_name)

        # discretize the monitor and get center locations
        sub_grid = self.simulation.discretize(monitor_data.monitor, extend=False)
        centers = sub_grid.centers

        # pass coords if each of the scalar field data have more than one coordinate along a dim
        xyz_kwargs = {}
        for dim, centers in zip("xyz", (centers.x, centers.y, centers.z)):
            scalar_data = list(monitor_data.field_components.values())
            coord_lens = [len(data.coords[dim]) for data in scalar_data]
            if all(ncoords > 1 for ncoords in coord_lens):
                xyz_kwargs[dim] = centers

        return monitor_data.colocate(**xyz_kwargs)

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

        field_dataset = self.at_centers(field_monitor_name)

        field_components = ("Ex", "Ey", "Ez")
        if not all(field_cmp in field_dataset for field_cmp in field_components):
            raise DataError(
                "Field monitor must contain 'Ex', 'Ey', and 'Ez' fields to compute intensity."
            )

        intensity_data = 0.0
        for field_cmp in field_components:
            field_cmp_data = field_dataset.data_vars[field_cmp]
            intensity_data += abs(field_cmp_data) ** 2
        intensity_data.name = "Intensity"
        return intensity_data

    def plot_field(  # pylint:disable=too-many-arguments, too-many-locals, too-many-branches
        self,
        field_monitor_name: str,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.FieldMonitor`, :class:`.FieldTimeData`, or :class:`.ModeSolverData`
            to plot.
        field_name : str
            Name of `field` component to plot (eg. `'Ex'`).
            Also accepts `'int'` to plot intensity.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
            If ``field_name='int'``, this has no effect.
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

        # intensity
        if field_name == "int":
            field_data = self.get_intensity(field_monitor_name)
            val = "abs"

        # normal case (eg. Ex)
        else:
            field_monitor_data = self.load_field_monitor(field_monitor_name)
            if field_name not in field_monitor_data.field_components:
                raise DataError(f"field_name '{field_name}' not found in data.")
            field_data = field_monitor_data.field_components[field_name]

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
                "'freq' suppled to 'plot_field', frequency selection key renamed to 'f' and 'freq' "
                "will error in future release, please update your local script to use 'f=value'."
            )
            sel_kwargs["f"] = sel_kwargs.pop("freq")
        if "time" in sel_kwargs:
            log.warning(
                "'time' suppled to 'plot_field', frequency selection key renamed to 't' and 'time' "
                "will error in future release, please update your local script to use 't=value'."
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
        field_data = field_data.squeeze(drop=True)
        non_scalar_coords = {name: val for name, val in field_data.coords.items() if val.size > 1}

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
        planar_coord = [name for name, val in spatial_coords_in_data.items() if val is False][0]
        axis = "xyz".index(planar_coord)
        position = float(field_data.coords[planar_coord])

        # the frequency at which to evaluate the permittivity with None signaling freq -> inf
        freq_eps_eval = sel_kwargs["f"] if "f" in sel_kwargs else None
        return self.plot_field_array(
            field_data=field_data,
            axis=axis,
            position=position,
            val=val,
            freq=freq_eps_eval,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

    @equal_aspect
    @add_ax_if_none
    # pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def plot_field_array(
        self,
        field_data: xr.DataArray,
        axis: Axis,
        position: float,
        val: Literal["real", "imag", "abs"] = "real",
        freq: float = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_data: xr.DataArray
            DataArray with the field data to plot.
        axis: Axis
            Axis normal to the plotting plane.
        position: float
            Position along the axis.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
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
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}

        # select the field value
        if val not in ("real", "imag", "abs"):
            raise DataError(f"`val` must be one of `{'real', 'imag', 'abs'}`, given {val}.")

        if val == "real":
            field_data = field_data.real
        elif val == "imag":
            field_data = field_data.imag
        elif val == "abs":
            field_data = abs(field_data)

        if val == "abs":
            cmap = "magma"
            eps_reverse = True
        else:
            cmap = "RdBu"
            eps_reverse = False

        # plot the field
        xy_coord_labels = list("xyz")
        xy_coord_labels.pop(axis)
        x_coord_label, y_coord_label = xy_coord_labels[0], xy_coord_labels[1]
        field_data.plot(
            ax=ax, x=x_coord_label, y=y_coord_label, cmap=cmap, vmin=vmin, vmax=vmax, robust=robust
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
