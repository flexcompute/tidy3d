"""Defines heat simulation data class"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pydantic.v1 as pd

from ...exceptions import DataError
from ...log import log
from ..base_sim.data.sim_data import AbstractSimulationData
from ..data.data_array import DCCapacitanceDataArray, DCIVCurveDataArray, SpatialDataArray
from ..data.dataset import TetrahedralGridDataset, TriangularGridDataset, UnstructuredGridDataset
from ..types import Ax, Literal, RealFieldVal
from ..viz import add_ax_if_none, equal_aspect
from .heat.simulation import HeatSimulation
from .monitor_data import HeatChargeMonitorDataType, StaticVoltageData, TemperatureData
from .simulation import HeatChargeSimulation


class HeatChargeSimulationData(AbstractSimulationData):
    """Stores results of a heat-charge simulation.

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

    data: Tuple[HeatChargeMonitorDataType, ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

    device_characteristics: Optional[Dict] = pd.Field(
        None,
        title="Device characteristics",
        description="Data characterizing the device. Current characteristics include: "
        "'iv_curve' for and I-V curve and 'cv_curve' for a capacitance curve.",
    )

    @pd.validator("device_characteristics", pre=True)
    def validate_device_characteristics(cls, val):
        if val is None:
            return val

        validated_dict = {}
        for key, dc in val.items():
            if isinstance(dc, DCCapacitanceDataArray):
                validated_dict[key] = DCCapacitanceDataArray(
                    data=dc.data,
                    dims=["Voltage (V)"],
                    coords=dc.coords,
                    attrs={"long_name": "Capacitance (fF)"},
                )
            elif isinstance(dc, DCIVCurveDataArray):
                validated_dict[key] = DCIVCurveDataArray(
                    data=dc.data,
                    dims=["Voltage (V)"],
                    coords=dc.coords,
                    attrs={"long_name": "Current (A)"},
                )
        return validated_dict

    @equal_aspect
    @add_ax_if_none
    def plot_field(
        self,
        monitor_name: str,
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

        if isinstance(monitor_data, TemperatureData):
            if monitor_data.temperature is None:
                raise DataError(f"No data to plot for monitor '{monitor_name}'.")
            field_data = self._field_component_value(monitor_data.temperature, val)
            property_to_plot = "heat_conductivity"

        elif isinstance(monitor_data, StaticVoltageData):
            if monitor_data.voltage is None:
                raise DataError(f"No data to plot for monitor '{monitor_name}'.")
            field_data = self._field_component_value(monitor_data.voltage, val)
            property_to_plot = "electric_conductivity"

        else:
            raise DataError(
                f"Monitor '{monitor_name}' (type '{monitor_data.monitor.type}') is not a "
                f"supported monitor. Supported monitors are 'TemperatureData', 'StaticVoltageData'."
            )

        field_name = monitor_data.field_name(val)

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
    """Wrapper for Heat simulation data. 'HeatSimulationData' is deprecated.
    Consider using 'HeatChargeSimulationData' instead."""

    simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="Original :class:`.HeatSimulation` associated with the data.",
    )

    @pd.root_validator(skip_on_failure=True)
    def issue_warning_deprecated(cls, values):
        """Issue warning for 'HeatSimulations'."""
        log.warning(
            "'HeatSimulationData' is deprecated and will be discontinued. You can use "
            "'HeatChargeSimulationData' instead"
        )
        return values
