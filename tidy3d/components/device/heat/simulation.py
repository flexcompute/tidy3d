"""Defines heat simulation class
NOTE: Keeping this class for backward compatibility only"""
from __future__ import annotations

from typing import Tuple

from ...types import Ax
from ...viz import add_ax_if_none, equal_aspect
from ...scene import Scene
from ..simulation import DeviceSimulation
from ....log import log


class HeatSimulation(DeviceSimulation):
    """Contains all information about heat simulation.

    Example
    -------
    >>> from tidy3d import Medium, SolidSpec, FluidSpec, UniformUnstructuredGrid, TemperatureMonitor
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
    ...     monitors=[TemperatureMonitor(size=(1, 2, 3), name="sample")],
    ... )
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warning(
            "Setting up deprecated ``HeatSimulation``. "
            "Do consider defining ``DeviceSimulation`` instead."
        )

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
            plot_type = "heat_source"

        return self.plot_scene_specs(
            x=x,
            y=y,
            z=z,
            ax=ax,
            alpha=alpha,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            plot_type=plot_type,
            hlim=hlim,
            vlim=vlim,
        )

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> HeatSimulation:
        """Create a simulation from a :class:.`Scene` instance. Must provide additional parameters
        to define a valid simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:.`Scene`
            Scene containing structures information.
        **kwargs
            Other arguments

        Example
        -------
        >>> from tidy3d import Scene, Medium, Box, Structure, UniformUnstructuredGrid
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> scene = Scene(
        ...     structures=[box],
        ...     medium=Medium(permittivity=3),
        ... )
        >>> sim = HeatSimulation.from_scene(
        ...     scene=scene,
        ...     center=(0, 0, 0),
        ...     size=(5, 6, 7),
        ...     grid_spec=UniformUnstructuredGrid(dl=0.4),
        ... )
        """

        return cls(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )
