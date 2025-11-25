"""Defines heat simulation data class"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.base_sim.data.sim_data import AbstractSimulationData
from tidy3d.components.data.data_array import (
    FreqVoltageDataArray,
    SpatialDataArray,
    SteadyVoltageDataArray,
)
from tidy3d.components.data.utils import (
    TetrahedralGridDataset,
    TriangularGridDataset,
    UnstructuredGridDataset,
)
from tidy3d.components.tcad.data.monitor_data.mesh import VolumeMeshData
from tidy3d.components.tcad.data.types import (
    SteadyPotentialData,
    TCADMonitorDataType,
    TemperatureData,
)
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.monitors.mesh import VolumeMeshMonitor
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types import Ax, RealFieldVal, annotate_type
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.exceptions import DataError, Tidy3dKeyError
from tidy3d.log import log

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


class DeviceCharacteristics(Tidy3dBaseModel):
    """Stores device characteristics. For example, in steady-state it stores
    the steady DC capacitance (provided an array of voltages has been defined
    in the simulation).

    Example
    -------

    >>> import tidy3d as td
    >>> C = [0, 1, 4]
    >>> V = [-1, -0.5, 0.5]
    >>> intensities = [0.1, 1.5, 3.6]
    >>> capacitance = SteadyVoltageDataArray(data=C, coords={"v": V})
    >>> current_voltage = SteadyVoltageDataArray(data=intensities, coords={"v": V})
    >>> device_characteristics = DeviceCharacteristics(
    ...     steady_dc_hole_capacitance=capacitance,
    ...     steady_dc_current_voltage=current_voltage,
    ... )

    """

    steady_dc_hole_capacitance: Optional[SteadyVoltageDataArray] = pd.Field(
        None,
        title="Steady DC hole capacitance",
        description="Device steady DC capacitance data based on holes. If the simulation "
        "has converged, these result should be close to that of electrons.",
    )

    steady_dc_electron_capacitance: Optional[SteadyVoltageDataArray] = pd.Field(
        None,
        title="Steady DC electron capacitance",
        description="Device steady DC capacitance data based on electrons. If the simulation "
        "has converged, these result should be close to that of holes.",
    )

    steady_dc_current_voltage: Optional[SteadyVoltageDataArray] = pd.Field(
        None,
        title="Steady DC current-voltage",
        description="Device steady DC current-voltage relation for the device.",
    )

    steady_dc_resistance_voltage: Optional[SteadyVoltageDataArray] = pd.Field(
        None,
        title="Small signal resistance",
        description="Steady DC computation of the small signal resistance. This is computed "
        "as the derivative of the current-voltage relation :math:`\\frac{\\Delta V}{\\Delta I}`, and the result "
        "is given in Ohms. Note that in 2D the resistance is given in :math:`\\Omega \\mu`.",
    )

    ac_current_voltage: Optional[FreqVoltageDataArray] = pd.Field(
        None,
        title="Small-signal AC current-voltage",
        description="Small-signal AC current as a function of DC bias voltage and frequency. "
        "This complex-valued data :math:`I(v, f)` is computed from small-signal analysis and "
        "can be used to determine frequency-dependent device parameters like admittance. "
        "For 2D simulations, the units are :math:`A/{\\mu m}`, so scale by device width.",
    )


class AbstractHeatChargeSimulationData(AbstractSimulationData, ABC):
    """Abstract class for HeatChargeSimulation results, or VolumeMesher results."""

    simulation: HeatChargeSimulation = pd.Field(
        title="Heat-Charge Simulation",
        description="Original :class:`.HeatChargeSimulation` associated with the data.",
    )

    @staticmethod
    def _get_field_by_name(monitor_data: TCADMonitorDataType, field_name: Optional[str] = None):
        """Return a field data based on a monitor dataset and a specified field name."""
        if field_name is None:
            if len(monitor_data.field_components) > 1:
                raise DataError(
                    "'field_name' must be specified for datasets that store more than one field."
                )
            field_name = next(iter(monitor_data.field_components))

        if field_name not in monitor_data.field_components.keys():
            raise DataError(f"field_name '{field_name}' not found in data.")

        field = monitor_data.field_components[field_name]
        if field is None:
            raise DataError(f"Field {field_name} is empty.")

        return field

    @equal_aspect
    @add_ax_if_none
    def plot_mesh(
        self,
        monitor_name: str,
        field_name: Optional[str] = None,
        structures_fill: bool = True,
        ax: Ax = None,
        **sel_kwargs: Any,
    ) -> Ax:
        """Plot the simulation mesh in a monitor region with structures overlaid.

        Parameters
        ----------
        monitor_name : str
            Name of :class:`.HeatChargeMonitor` to plot. Must be a monitor with the `unstructured=True` setting.
        field_name : Optional[str] = "mesh"
            Name of ``field`` component whose associated grid to plot. Not required if monitor data contains only one field.
        structures_fill : bool = True
            Whether to overlay the mesh on structures filled with color or only show structure outlines.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            or time dimension (``t``) if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Note
        ----
            For 3D simulations, the 2D mesh shown here would be the result of slicing the underlying unstructured tetrahedral grid with the selected plane.
            If however the monitor sets `conformal=True`, the simulation mesh has been made to conform to the monitor plane, in which case the visualized mesh is exact.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        monitor_data = self[monitor_name]

        if not monitor_data.monitor.unstructured:
            raise DataError("'plot_mesh' can only be used with unstructured-grid monitors.")

        field_data = self._get_field_by_name(monitor_data=monitor_data, field_name=field_name)

        # do sel on unstructured data
        if len(sel_kwargs) > 0:
            field_data = field_data.sel(**sel_kwargs)

        if isinstance(field_data, TetrahedralGridDataset):
            raise DataError(
                "Must select a two-dimensional slice of unstructured dataset for plotting"
                " on a plane."
            )

        # compute parameters for structures plot
        axis = field_data.normal_axis
        position = field_data.normal_pos

        # compute plot bounds
        field_data_bounds = field_data.bounds
        min_bounds = list(field_data_bounds[0])
        max_bounds = list(field_data_bounds[1])
        min_bounds.pop(axis)
        max_bounds.pop(axis)

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}
        # plot the simulation structures first, because we don't use alpha
        ax = self.simulation.scene.plot_structures(
            ax=ax,
            fill=structures_fill,
            hlim=(min_bounds[0], max_bounds[0]),
            vlim=(min_bounds[1], max_bounds[1]),
            **interp_kwarg,
        )

        # only then overlay the mesh plot
        field_data.plot(ax=ax, cmap=False, field=False, grid=True)

        # set the limits based on the xarray coordinates min and max
        ax.set_xlim(min_bounds[0], max_bounds[0])
        ax.set_ylim(min_bounds[1], max_bounds[1])

        return ax


class HeatChargeSimulationData(AbstractHeatChargeSimulationData):
    """Stores results of a :class:`HeatChargeSimulation`.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> temp_mnt = td.TemperatureMonitor(size=(1, 2, 3), name="sample")
    >>> heat_sim = HeatChargeSimulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         td.Structure(
    ...             geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=td.Medium(
    ...                 permittivity=2.0, heat_spec=td.SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=td.Medium(permittivity=3.0, heat_spec=td.FluidSpec()),
    ...     grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    ...     sources=[td.HeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         td.HeatChargeBoundarySpec(
    ...             placement=td.StructureBoundary(structure="box"),
    ...             condition=td.TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[temp_mnt],
    ... )
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> temp_array = td.SpatialDataArray(300 * np.abs(np.random.random((2,3,4))), coords=coords)
    >>> temp_mnt_data = td.TemperatureData(monitor=temp_mnt, temperature=temp_array)
    >>> heat_sim_data = td.HeatChargeSimulationData(
    ...     simulation=heat_sim, data=[temp_mnt_data],
    ... )
    """

    data: tuple[annotate_type(TCADMonitorDataType), ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

    device_characteristics: Optional[DeviceCharacteristics] = pd.Field(
        None,
        title="Device characteristics",
        description="Data characterizing the device :class:`DeviceCharacteristics`.",
    )

    @equal_aspect
    @add_ax_if_none
    def plot_field(
        self,
        monitor_name: str,
        field_name: Optional[Literal["temperature", "potential"]] = None,
        val: RealFieldVal = "real",
        scale: Literal["lin", "log"] = "lin",
        structures_alpha: float = 0.2,
        robust: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Ax = None,
        cmap: Optional[Union[str, Colormap]] = None,
        **sel_kwargs: Any,
    ) -> Ax:
        """Plot the data for a monitor with simulation structures overlaid.

        Parameters
        ----------
        monitor_name : str
            Name of :class:`.HeatChargeMonitor` to plot.
        field_name : Optional[Literal["temperature", "potential"]] = None
            Name of ``field`` component to plot (eg. `'temperature'`). Not required if monitor data contains only one field.
        val : Literal['real', 'abs', 'abs^2'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'log']
            Plot in linear or logarithmic scale.
        structures_alpha : float = 0.2
            Opacity of the structure property to plot (heat conductivity or electric conductivity
            depending on the type of monitor). Must be between 0 and 1 (inclusive).
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
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values. ``None`` uses the default which infers it from the data.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            or time dimension (``t``) if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        monitor_data = self[monitor_name]
        property_to_plot = None

        field = self._get_field_by_name(monitor_data=monitor_data, field_name=field_name)

        # forward field name to actual data so it gets displayed
        # field.name = field_name
        field_data = self._field_component_value(field, val)

        if isinstance(monitor_data, (TemperatureData, VolumeMeshData)):
            property_to_plot = "heat_conductivity"
        elif isinstance(monitor_data, SteadyPotentialData):
            property_to_plot = "electric_conductivity"
        else:
            raise DataError(
                f"Monitor '{monitor_name}' (type '{monitor_data.monitor.type}') is not a "
                f"supported monitor. Supported monitors are 'TemperatureData', 'SteadyPotentialData'."
            )

        if scale == "log":
            field_data = np.log10(np.abs(field_data))

        cmap_to_use = "coolwarm" if cmap is None else cmap

        # do sel on unstructured data
        # it could produce either SpatialDataArray or UnstructuredGridDatasetType
        if isinstance(field_data, UnstructuredGridDataset) and len(sel_kwargs) > 0:
            field_data = field_data.sel(**sel_kwargs)

        if isinstance(field_data, TetrahedralGridDataset):
            raise DataError(
                "Must select a two-dimensional slice of unstructured dataset for plotting"
                " on a plane."
            )

        if isinstance(field_data, TriangularGridDataset):
            field_data.plot(
                ax=ax,
                cmap=cmap_to_use,
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={"label": field_name},
                grid=False,
            )

            # compute parameters for structures overlay plot
            axis = field_data.normal_axis
            position = field_data.normal_pos

            # compute plot bounds
            field_data_bounds = field_data.bounds
            min_bounds = list(field_data_bounds[0])
            max_bounds = list(field_data_bounds[1])
            min_bounds.pop(axis)
            max_bounds.pop(axis)

        if isinstance(field_data, SpatialDataArray):
            # interp out any monitor.size==0 dimensions
            monitor = self.simulation.get_monitor_by_name(monitor_name)
            thin_dims = {
                "xyz"[dim]: monitor.center[dim]
                for dim in range(3)
                if monitor.size[dim] == 0 and "xyz"[dim] not in sel_kwargs
            }
            for axis, pos in thin_dims.items():
                if field_data.coords[axis].size <= 1:
                    field_data = field_data.sel(**{axis: pos}, method="nearest")
                else:
                    field_data = field_data.interp(**{axis: pos}, kwargs={"bounds_error": True})

            # select the extra coordinates out of the data from user-specified kwargs
            for coord_name, coord_val in sel_kwargs.items():
                if field_data.coords[coord_name].size <= 1:
                    field_data = field_data.sel(**{coord_name: coord_val}, method=None)
                else:
                    field_data = field_data.interp(
                        **{coord_name: coord_val}, kwargs={"bounds_error": True}
                    )

            field_data = field_data.squeeze(drop=True)
            non_scalar_coords = {name: c for name, c in field_data.coords.items() if c.size > 1}

            # assert the data is valid for plotting
            if len(non_scalar_coords) != 2:
                raise DataError(
                    f"Data after selection has {len(non_scalar_coords)} coordinates "
                    f"({list(non_scalar_coords.keys())}), "
                    "must be 2 spatial coordinates for plotting on plane. "
                    "Please add keyword arguments to 'plot_monitor_data()' to select out the other coords."
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

            xy_coord_labels = list("xyz")
            xy_coord_labels.pop(axis)
            x_coord_label, y_coord_label = xy_coord_labels[0], xy_coord_labels[1]
            field_data.plot(
                ax=ax,
                x=x_coord_label,
                y=y_coord_label,
                cmap=cmap_to_use,
                vmin=vmin,
                vmax=vmax,
                robust=robust,
                cbar_kwargs={"label": field_name},
            )

            # compute plot bounds
            x_coord_values = field_data.coords[x_coord_label]
            y_coord_values = field_data.coords[y_coord_label]
            min_bounds = (min(x_coord_values), min(y_coord_values))
            max_bounds = (max(x_coord_values), max(y_coord_values))

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}
        # plot the simulation heat/electric conductivity
        if property_to_plot is not None:
            ax = self.simulation.scene.plot_heat_charge_property(
                cbar=False,
                alpha=structures_alpha,
                ax=ax,
                property=property_to_plot,
                **interp_kwarg,
            )

        # set the limits based on the xarray coordinates min and max
        ax.set_xlim(min_bounds[0], max_bounds[0])
        ax.set_ylim(min_bounds[1], max_bounds[1])

        return ax


class HeatSimulationData(HeatChargeSimulationData):
    """DEPRECATED: Wrapper for Heat Simulation data.

    Warning
    -------
        :class:`HeatSimulationData` is DEPRECATED.
        Consider using :class:`HeatChargeSimulationData` instead.
    """

    simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="Original :class:`HeatSimulation` associated with the data.",
    )

    @pd.root_validator(skip_on_failure=True)
    def issue_warning_deprecated(cls, values):
        """Issue warning for 'HeatSimulations'."""
        log.warning(
            "'HeatSimulationData' is deprecated and will be discontinued. You can use "
            "'HeatChargeSimulationData' instead"
        )
        return values


class VolumeMesherData(AbstractHeatChargeSimulationData):
    """Stores results of a :class:`VolumeMesher`.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> mesh_mnt = td.VolumeMeshMonitor(size=(1, 2, 3), name="mesh")
    >>> temp_mnt = td.TemperatureMonitor(size=(1, 2, 3), name="sample")
    >>> heat_sim = td.HeatChargeSimulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         td.Structure(
    ...             geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=td.Medium(
    ...                 permittivity=2.0, heat_spec=td.SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=td.Medium(permittivity=3.0, heat_spec=td.FluidSpec()),
    ...     grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    ...     sources=[td.HeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         td.HeatChargeBoundarySpec(
    ...             placement=td.StructureBoundary(structure="box"),
    ...             condition=td.TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[temp_mnt],
    ... )
    >>> tet_grid_points = td.PointDataArray(
    ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     dims=("index", "axis"),
    ... )
    >>> tet_grid_cells = td.CellDataArray(
    ...     [[0, 1, 2, 4], [1, 2, 3, 4]],
    ...     dims=("cell_index", "vertex_index"),
    ... )
    >>> tet_grid_values = td.IndexedDataArray(
    ...     np.zeros((tet_grid_points.shape[0],)),
    ...     dims=("index",),
    ...     name="Mesh",
    ... )
    >>> tet_grid = td.TetrahedralGridDataset(
    ...     points=tet_grid_points,
    ...     cells=tet_grid_cells,
    ...     values=tet_grid_values,
    ... )
    >>> mesh_mnt_data = td.VolumeMeshData(monitor=mesh_mnt, mesh=tet_grid) # doctest: +SKIP
    >>> mesh_data = td.VolumeMesherData(simulation=heat_sim, data=[mesh_mnt_data], monitors=[mesh_mnt]) # doctest: +SKIP
    """

    monitors: tuple[VolumeMeshMonitor, ...] = pd.Field(
        ...,
        title="Monitors",
        description="List of monitors to be used for the mesher.",
    )

    data: tuple[VolumeMeshData, ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.VolumeMesher`.",
    )

    @property
    def mesher(self) -> VolumeMesher:
        """Get the mesher associated with this mesher data."""
        return VolumeMesher(
            simulation=self.simulation,
            monitors=self.monitors,
        )

    @pd.root_validator(skip_on_failure=True)
    def data_monitors_match_sim(cls, values):
        """Ensure each :class:`AbstractMonitorData` in ``.data`` corresponds to a monitor in
        ``.simulation``.
        """
        monitors = values.get("monitors")
        data = values.get("data")
        mnt_names = {mnt.name for mnt in monitors}

        for mnt_data in data:
            monitor_name = mnt_data.monitor.name
            if monitor_name not in mnt_names:
                raise DataError(
                    f"Data with monitor name '{monitor_name}' supplied "
                    f"but not found in the list of monitors."
                )
        return values

    def get_monitor_by_name(self, name: str) -> VolumeMeshMonitor:
        """Return monitor named 'name'."""
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise Tidy3dKeyError(f"No monitor named '{name}'")
