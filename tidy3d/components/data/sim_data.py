""" Simulation Level Data """
from typing import Dict, Optional, Literal

import xarray as xr
import pydantic as pd

from .base import Tidy3dData
from .monitor_data import MonitorDataType, AbstractFieldData
from ..base import cached_property
from ..simulation import Simulation
from ..types import Ax, Axis
from ..viz import equal_aspect, add_ax_if_none
from ...log import log, DataError

# TODO: final decay value
# TODO: saving and loading from hdf5 group or json file
# TODO: docstring examples?
# TODO: ModeSolverData


class SimulationData(Tidy3dData):
    """Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`.Simulation` associated with the data.",
    )

    monitor_data: Dict[str, MonitorDataType] = pd.Field(
        ...,
        title="Monitor Data",
        description="Mapping of monitor name to :class:`.MonitorData` instance.",
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

    normalize_index: Optional[pd.NonNegativeInt] = pd.Field(
        0,
        title="Normalization index",
        description="Index of the source in the simulation.sources to use to normalize the data.",
    )

    @pd.validator("normalize_index", always=True)
    def _check_normalize_index(cls, val, values):
        """Check validity of normalize index in context of simulation.sources."""

        # not normalizing
        if val is None:
            return val

        assert val >= 0, "normalize_index can't be negative."

        num_sources = len(values.get("simulation").sources)

        # no sources, just skip normalization
        if num_sources == 0:
            log.warning(f"normalize_index={val} supplied but no sources found, not normalizing.")
            return None  # TODO: do we want this behavior though?

        assert val < num_sources, f"{num_sources} sources greater than normalize_index of {val}"

        return val

    def __getitem__(self, monitor_name: str) -> MonitorDataType:
        """Get a :class:`.MonitorData` by name. Apply symmetry and normalize if applicable."""

        monitor_data = self.monitor_data[monitor_name]
        monitor_data = self.apply_symmetry(monitor_data)
        monitor_data = self.normalize_monitor_data(monitor_data)
        return monitor_data

    @property
    def final_decay_value(self) -> float:
        """Returns value of the field decay at the final time step."""
        log_str = self.log
        if log_str is None:
            raise DataError("No log string in the SimulationData object, "
                "can't find final decay value.")
        lines = log_str.split("\n")
        decay_lines = [l for l in lines if "field decay" in l]
        final_decay = 1.0
        if len(decay_lines) > 0:
            final_decay_line = decay_lines[-1]
            final_decay = float(final_decay_line.split("field decay: ")[-1])
        return final_decay

    def apply_symmetry(self, monitor_data: MonitorDataType) -> MonitorDataType:
        """Return copy of :class:`.MonitorData` object with symmetry values applied."""
        grid_expanded = self.simulation.discretize(monitor_data.monitor, extend=True)
        return monitor_data.apply_symmetry(
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=grid_expanded,
        )

    def normalize_monitor_data(self, monitor_data: MonitorDataType) -> MonitorDataType:
        """Return copy of :class:`.MonitorData` object with data normalized to source."""

        # if no normalize index, just return the new copy right away.
        if self.normalize_index is None:
            return monitor_data.copy()

        # get source time information
        source = self.simulation.sources[self.normalize_index]
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

        return monitor_data.normalize(source_spectrum_fn)

    def load_field_monitor(self, monitor_name: str) -> AbstractFieldData:
        """Load monitor and raise exception if not a field monitor."""
        mon_data = self[monitor_name]
        if not isinstance(mon_data, AbstractFieldData):
            raise DataError(
                f"data for monitor '{monitor_name}' does not contain field data as it is a `{type(mon_data)}`."
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

        # get the monitor, discretize, and get center locations
        monitor = monitor_data.monitor
        sub_grid = self.simulation.discretize(monitor, extend=True)
        centers = sub_grid.centers

        # pass coords if each of the scalar field data have more than one coordinate along a dim
        xyz_kwargs = {}
        for dim, centers in zip("xyz", (centers.x, centers.y, centers.z)):
            scalar_data = [data for _, (data, _, _) in monitor_data.field_components.items()]
            coord_lens = [len(data.coords[dim]) for data in scalar_data if data is not None]
            if all([ncoords > 1 for ncoords in coord_lens]):
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
                f"Field monitor must contain 'Ex', 'Ey', and 'Ez' fields to compute intensity."
            )

        intensity_data = 0.0
        for field_cmp in field_components:
            field_cmp_data = field_dataset.data_vars[field_cmp]
            intensity_data += abs(field_cmp_data) ** 2
        intensity_data.name = "Intensity"
        return intensity_data

    def plot_field(
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
            Name of :class:`.FieldMonitor`, :class:`.FieldTimeData`, or :class:`.ModeFieldData`
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
            field_data, _, _ = field_monitor_data.field_components[field_name]
            if field_data is None:
                raise DataError(
                    f"field_name '{field_name}' was not stored in data, must be specified in the monitor.fields"
                )

        # select the extra coordinates out of the data
        field_data = field_data.interp(**sel_kwargs)
        field_data = field_data.squeeze(drop=True)
        final_coords = {k: v for k, v in field_data.coords.items() if v.size > 1}

        # assert the data is valid for plotting
        if len(final_coords) != 2:
            raise DataError(
                f"Data after selection has {len(final_coords)} coordinates ({list(final_coords.keys())}), must be 2 spatial coordinates for plotting on plane. Please add keyword arguments to `plot_field()` to select out other"
            )

        spatial_coords_in_data = {coord_name: (coord_name in final_coords) for coord_name in "xyz"}

        if sum(spatial_coords_in_data.values()) != 2:
            raise DataError(
                f"All coordinates in the data after selection must be spatial (x, y, z), given {final_coords.keys()}."
            )

        # get the spatial coordinate corresponding to the plane
        planar_coord = {name: val for name, val in spatial_coords_in_data.items() if val is False}
        axis = "xyz".index(list(planar_coord.keys())[0])
        position = list(planar_coord.values())[0]

        # the frequency at which to evaluate the permittivity with None signaling freq -> inf
        freq_eps_eval = sel_kwargs["freq"] if "freq" in sel_kwargs else None

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
        x_coord_label, y_coord_label = xy_coord_labels  # pylint:disable=unbalanced-tuple-unpacking
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
