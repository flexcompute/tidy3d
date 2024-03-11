""" Simulation Level Data """
from __future__ import annotations
from typing import Callable, Tuple, Union

import pathlib
import xarray as xr
import pydantic.v1 as pd
import numpy as np
import h5py
import json

from .monitor_data import MonitorDataTypes, MonitorDataType, AbstractFieldData, FieldTimeData
from ..simulation import Simulation
from ..types import Ax, Axis, annotate_type, FieldVal, PlotScale, ColormapType
from ..viz import equal_aspect, add_ax_if_none
from ...exceptions import DataError, Tidy3dKeyError
from ...log import log
from ..base import JSON_TAG

from ..base_sim.data.sim_data import AbstractSimulationData


DATA_TYPE_MAP = {data.__fields__["monitor"].type_: data for data in MonitorDataTypes}

# maps monitor type (string) to the class of the corresponding data
DATA_TYPE_NAME_MAP = {val.__fields__["monitor"].type_.__name__: val for val in MonitorDataTypes}


class SimulationData(AbstractSimulationData):
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
        mon_data = self.load_field_monitor(field_monitor_name)
        field_dataset = self._at_boundaries(mon_data)

        time_domain = isinstance(self.monitor_data[field_monitor_name], FieldTimeData)

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
            grid_correction = mon_data.grid_dual_correction * mon_data.grid_primal_correction
            poynting_components["S" + dim] *= grid_correction

        return xr.Dataset(poynting_components)

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
                    if len(field_name) == 2 and val in ("real", "imag", "re", "im")
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
