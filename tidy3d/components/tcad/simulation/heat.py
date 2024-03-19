"""Defines heat simulation class
NOTE: Keeping this class for backward compatibility only"""

from __future__ import annotations

from typing import Tuple

import pydantic.v1 as pd

from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types import Ax
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.log import log


class HeatSimulation(HeatChargeSimulation):
    """
    Contains all information about heat simulation.

    Example
    -------
    >>> import tidy3d as td
    >>> heat_sim = td.HeatSimulation( # doctest: +SKIP
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
    ...     monitors=[td.TemperatureMonitor(size=(1, 2, 3), name="sample")],
    ... )
    """

    @pd.root_validator(skip_on_failure=True)
    def issue_warning_deprecated(cls, values):
        """Issue warning for 'HeatSimulations'."""
        log.warning(
            "Setting up deprecated 'HeatSimulation'. "
            "Consider defining 'HeatChargeSimulation' instead."
        )
        return values

    @equal_aspect
    @add_ax_if_none
    def plot_heat_conductivity(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        colorbar: str = "conductivity",
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        colorbar: str = "conductivity"
            Display colorbar for thermal conductivity ("conductivity") or heat source rate
            ("source").
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        plot_type = None
        if colorbar == "conductivity":
            plot_type = "heat_conductivity"
        elif colorbar == "source":
            plot_type = "source"

        return self.plot_property(
            x=x,
            y=y,
            z=z,
            ax=ax,
            alpha=alpha,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            property=plot_type,
            hlim=hlim,
            vlim=vlim,
        )
