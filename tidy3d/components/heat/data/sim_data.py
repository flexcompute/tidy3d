"""Defines heat simulation data class"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pydantic.v1 as pd

from .monitor_data import HeatMonitorDataType, TemperatureData
from ..simulation import HeatSimulation

from ...base_sim.data.sim_data import AbstractSimulationData
from ...types import Ax, RealFieldVal, Literal
from ...viz import equal_aspect, add_ax_if_none
from ....exceptions import DataError


class HeatSimulationData(AbstractSimulationData):
    """Stores results of a heat simulation.

    Example
    -------
    >>> from tidy3d import Medium, SolidSpec, FluidSpec, UniformUnstructuredGrid, SpatialDataArray
    >>> from tidy3d import Structure, Box, UniformUnstructuredGrid, UniformHeatSource
    >>> from tidy3d import StructureBoundary, TemperatureBC, TemperatureMonitor, TemperatureData
    >>> from tidy3d import HeatBoundarySpec
    >>> import numpy as np
    >>> temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="sample")
    >>> heat_sim = HeatSimulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(
    ...                 permittivity=2.0, heat_spec=SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=Medium(permittivity=3.0, heat_spec=FluidSpec()),
    ...     grid_spec=UniformUnstructuredGrid(dl=0.1),
    ...     sources=[UniformHeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         HeatBoundarySpec(
    ...             placement=StructureBoundary(structure="box"),
    ...             condition=TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[temp_mnt],
    ... )
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> temp_array = SpatialDataArray(300 * np.abs(np.random.random((2,3,4))), coords=coords)
    >>> temp_mnt_data = TemperatureData(monitor=temp_mnt, temperature=temp_array)
    >>> heat_sim_data = HeatSimulationData(
    ...     simulation=heat_sim, data=[temp_mnt_data],
    ... )
    """

    simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="Original :class:`.HeatSimulation` associated with the data.",
    )

    data: Tuple[HeatMonitorDataType, ...] = pd.Field(
        ...,
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

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
        """Plot the data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.TemperatureMonitorData` to plot.
        val : Literal['real', 'abs', 'abs^2'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        structures_alpha : float = 0.2
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
            or time dimension (`t`) if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (`x`, `y`, or `z`).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        monitor_data = self[monitor_name]

        if not isinstance(monitor_data, TemperatureData):
            raise DataError(
                f"Monitor '{monitor_name}' (type '{monitor_data.monitor.type}') is not a "
                f"'TemperatureMonitor'."
            )

        field_data = self._field_component_value(monitor_data.temperature, val)

        if scale == "log":
            field_data = np.log10(np.abs(field_data))
            cmap_type = "sequential"
        elif val == "real":
            cmap_type = "divergent"
        else:
            cmap_type = "sequential"

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
                "Please add keyword arguments to `plot_monitor_data()` to select out the other coords."
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

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}

        if cmap_type == "divergent":
            cmap = "RdBu_r"
        elif cmap_type == "sequential":
            cmap = "magma"

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
            cbar_kwargs={"label": "temperature"},
        )

        # plot the simulation heat conductivity
        ax = self.simulation.scene.plot_structures_heat_conductivity(
            cbar=False,
            alpha=structures_alpha,
            ax=ax,
            **interp_kwarg,
        )

        # set the limits based on the xarray coordinates min and max
        x_coord_values = field_data.coords[x_coord_label]
        y_coord_values = field_data.coords[y_coord_label]
        ax.set_xlim(min(x_coord_values), max(x_coord_values))
        ax.set_ylim(min(y_coord_values), max(y_coord_values))

        return ax
