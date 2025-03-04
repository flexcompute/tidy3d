"""Defines heat simulation data class"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base_sim.data.sim_data import AbstractSimulationData
from tidy3d.components.data.data_array import (
    SpatialDataArray,
    SteadyVoltageDataArray,
)
from tidy3d.components.data.utils import (
    TetrahedralGridDataset,
    TriangularGridDataset,
    UnstructuredGridDataset,
)
from tidy3d.components.tcad.data.types import (
    SteadyPotentialData,
    TCADMonitorDataType,
    TemperatureData,
)
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types import Ax, Literal, RealFieldVal, annotate_type
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.exceptions import DataError
from tidy3d.log import log

from ...base import Tidy3dBaseModel


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


class HeatChargeSimulationData(AbstractSimulationData):
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

    simulation: HeatChargeSimulation = pd.Field(
        title="Heat-Charge Simulation",
        description="Original :class:`.HeatChargeSimulation` associated with the data.",
    )

    data: Tuple[annotate_type(TCADMonitorDataType), ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

    device_characteristics: Optional[DeviceCharacteristics] = pd.Field(
        None,
        title="Device characteristics",
        description="Data characterizing the device. Current characteristics include: "
        "'steady_dc_hole_capacitance', 'steady_dc_electron_capacitance', and 'steady_dc_current_voltage'",
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
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.TemperatureMonitorData` to plot.
        field_name : Optional[Literal["temperature", "potential"]] = None
            Name of ``field`` component to plot (eg. `'temperature'`). Not required if monitor data contains only one field.
        val : Literal['real', 'abs', 'abs^2'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'log']
            Plot in linear or logarithmic scale.
        structures_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
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

        if field_name is None:
            if isinstance(monitor_data, TemperatureData):
                field_name = "temperature"
            elif isinstance(monitor_data, SteadyPotentialData):
                field_name = "potential"

        if field_name not in monitor_data.field_components.keys():
            raise DataError(f"field_name '{field_name}' not found in data.")

        field = monitor_data.field_components[field_name]
        if field is None:
            raise DataError(f"Field {field_name} is empty and cannot be plotted.")
        # forward field name to actual data so it gets displayed
        # field.name = field_name
        field_data = self._field_component_value(field, val)

        if isinstance(monitor_data, TemperatureData):
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

        cmap = "coolwarm"

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
                cmap=cmap,
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
                    field_data = field_data.interp(**{axis: pos}, kwargs=dict(bounds_error=True))

            # select the extra coordinates out of the data from user-specified kwargs
            for coord_name, coord_val in sel_kwargs.items():
                if field_data.coords[coord_name].size <= 1:
                    field_data = field_data.sel(**{coord_name: coord_val}, method=None)
                else:
                    field_data = field_data.interp(
                        **{coord_name: coord_val}, kwargs=dict(bounds_error=True)
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
                cmap=cmap,
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
        :class`HeatSimulationData` is DEPRECATED.
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
