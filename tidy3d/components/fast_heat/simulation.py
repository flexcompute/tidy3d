"""Defines the Cartesian fast-heat simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.fast_heat.grid import FastHeatGridSpec
from tidy3d.components.fast_heat.monitor import FastHeatTemperatureMonitor
from tidy3d.components.geometry.base import Box
from tidy3d.components.material.types import StructureMediumType
from tidy3d.components.medium import Medium
from tidy3d.components.scene import Scene
from tidy3d.components.tcad.analysis.heat_simulation_type import UnsteadyHeatAnalysis
from tidy3d.components.tcad.boundary.specification import HeatChargeBoundarySpec
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.tcad.types import (
    ConvectionBC,
    HeatFluxBC,
    HeatSource,
    RadiationBC,
    TemperatureBC,
    ThermalContactResistance,
)
from tidy3d.components.tcad.viz import (
    HEAT_BC_COLOR_CONVECTION,
    HEAT_BC_COLOR_FLUX,
    HEAT_BC_COLOR_TEMPERATURE,
    HEAT_SOURCE_CMAP,
    plot_params_heat_bc,
    plot_params_heat_source,
)
from tidy3d.components.types import ScalarSymmetry
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import inf

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.types import Ax
    from tidy3d.components.viz import PlotParams


class FastHeatSimulation(AbstractSimulation):
    """Defines heat conduction simulations on a Cartesian grid, either shared by the whole domain
    or specified per stacked layer.

    Notes
    -----
    A steady simulation solves for the temperature :math:`T` with thermal conductivity :math:`k`
    and volumetric heat source :math:`q`:

    .. math::

        -\\nabla \\cdot (k \\nabla T) = q

    Setting ``analysis_spec`` to an :class:`.UnsteadyHeatAnalysis` adds the temporal term, with
    :math:`\\rho` the density and :math:`c_p` the specific heat capacity:

    .. math::

        \\frac{\\partial \\rho c_p T}{\\partial t} - \\nabla \\cdot (k \\nabla T) = q

    The temperature is computed in the solid media only. Boundary conditions, heat sources and
    temperature monitors complete the setup; without ``analysis_spec`` the simulation is steady.
    :attr:`grids` holds the Cartesian grid of each layer. A simulation with zero size along one
    axis is 2D.

    Example
    -------
    >>> from tidy3d import (
    ...     Box, HeatChargeBoundarySpec, HeatSource, Medium, SimulationBoundary, SolidSpec,
    ...     Structure, TemperatureBC, UniformGrid,
    ... )
    >>> from tidy3d.components.fast_heat.monitor import FastHeatTemperatureMonitor
    >>> silicon = Medium(name="Si", heat_spec=SolidSpec(conductivity=148e-6))
    >>> sim = FastHeatSimulation(
    ...     size=(2, 2, 1),
    ...     structures=[Structure(geometry=Box(size=(1, 1, 1)), medium=silicon, name="box")],
    ...     boundary_spec=[
    ...         HeatChargeBoundarySpec(
    ...             placement=SimulationBoundary(surfaces=["z-"]),
    ...             condition=TemperatureBC(temperature=300),
    ...         )
    ...     ],
    ...     sources=[HeatSource(structures=["box"], rate=1e-3)],
    ...     monitors=[FastHeatTemperatureMonitor(size=(2, 2, 0), name="mid")],
    ...     grid_spec=FastHeatGridSpec(
    ...         grid_x=UniformGrid(dl=0.1),
    ...         grid_y=UniformGrid(dl=0.1),
    ...         grid_z=UniformGrid(dl=0.1),
    ...     ),
    ... )
    """

    medium: StructureMediumType = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium of the simulation: fills every part of the domain that no "
        ":class:`.Structure` covers. Defaults to :class:`.Medium`, which has no solid "
        "``heat_spec``, so no temperature is computed there.",
    )

    grid_spec: FastHeatGridSpec = Field(
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )

    symmetry: tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = Field(
        default=(0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (symmetry off) or ``1`` (symmetry on). "
        "Only ``(0, 0, 0)`` is supported.",
    )

    boundary_spec: tuple[HeatChargeBoundarySpec, ...] = Field(
        default=(),
        title="Boundary Condition Specifications",
        description="List of boundary condition specifications. The supported conditions are "
        ":class:`.TemperatureBC`, :class:`.HeatFluxBC` and :class:`.ConvectionBC` with a "
        "constant ``transfer_coeff`` and no ``emissivity``. Surfaces without a boundary "
        "condition are insulated. A steady simulation needs a :class:`.TemperatureBC` or a "
        ":class:`.ConvectionBC` with a positive ``transfer_coeff``. A condition applies on "
        "the simulation boundary of solid media and on interfaces between a solid and a "
        "non-solid medium; a placement on an interface between two solids is an error when the "
        "simulation is solved.",
    )

    sources: tuple[HeatSource, ...] = Field(
        default=(),
        title="Heat Sources",
        description="List of heat sources, each applying its ``rate`` to the structures it names. "
        "These structures must be made of a solid medium. Rates of several sources on one "
        "structure add.",
    )

    monitors: tuple[FastHeatTemperatureMonitor, ...] = Field(
        default=(),
        title="Monitors",
        description="Temperature monitors in the simulation.",
    )

    analysis_spec: UnsteadyHeatAnalysis | None = Field(
        default=None,
        title="Analysis Specification",
        description="Transient analysis settings. If ``None``, the steady-state temperature is "
        "computed.",
    )

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the simulation boundary conditions on a plane defined by one nonzero x,y,z
        coordinate.

        The drawing follows :meth:`.HeatChargeSimulation.plot_boundaries`: it does not draw
        :class:`.SimulationBoundary` placements and ignores ``surfaces``. The faces each
        condition applies to are described in ``boundary_spec``.

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
        **kwargs
            Unused.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        plane = Box(center=center, size=Box.unpop_axis(0, (inf, inf), axis=axis))
        # Nothing to draw without conditions, on a plane that misses the domain, or in 2D on a
        # plane across the zero-size axis (the boundaries are lines in the simulation plane).
        lower, upper = self.bounds
        across_zero_axis = any(size == 0 and other != axis for other, size in enumerate(self.size))
        boundaries = []
        if self.boundary_spec and lower[axis] <= position <= upper[axis] and not across_zero_axis:
            boundaries = HeatChargeSimulation._construct_heat_charge_boundaries(
                structures=[self.simulation_structure, *self.scene.sorted_structures],
                plane=plane,
                boundary_spec=self.boundary_spec,
            )
        for boundary_spec, shape in boundaries:
            plot_params = self._get_bc_plot_params(boundary_spec=boundary_spec)
            ax = self.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        ax = self.add_ax_lims(axis=axis, ax=ax)
        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)
        return Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )

    @staticmethod
    def _get_bc_plot_params(boundary_spec: HeatChargeBoundarySpec) -> PlotParams:
        """Plot parameters of a boundary condition, coloured as in
        :class:`.HeatChargeSimulation`."""
        condition = boundary_spec.condition
        if isinstance(condition, TemperatureBC):
            return plot_params_heat_bc.updated_copy(facecolor=HEAT_BC_COLOR_TEMPERATURE)
        if isinstance(condition, HeatFluxBC | ThermalContactResistance):
            return plot_params_heat_bc.updated_copy(facecolor=HEAT_BC_COLOR_FLUX)
        if isinstance(condition, ConvectionBC | RadiationBC):
            return plot_params_heat_bc.updated_copy(facecolor=HEAT_BC_COLOR_CONVECTION)
        return plot_params_heat_bc

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        alpha: float | None = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the structures each heat source heats on a plane defined by one nonzero x,y,z
        coordinate, coloured by the mean rate of the source. A structure heated by several
        sources is coloured by the last one.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : tuple[float, float] = None
            Unused; the plot spans the simulation domain.
        vlim : tuple[float, float] = None
            Unused; the plot spans the simulation domain.
        alpha : float = None
            Opacity of the sources. If ``None``, the sources are opaque.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        from matplotlib import colormaps

        alpha = 1 if alpha is None else alpha
        if alpha <= 0:
            return ax

        # As in HeatChargeSimulation, a structure named by several sources shows the last one.
        structures = self.scene.sorted_structures
        source_of = {name: source for source in self.sources for name in source.structures}
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        plane = Box(center=center, size=Box.unpop_axis(0, (inf, inf), axis=axis))
        shapes = self.scene._filter_structures_plane(
            structures=structures,
            plane=plane,
            property_list=[source_of.get(structure.name) for structure in structures],
            section_tolerance_2d=True,
        )

        rates = [float(np.mean(source.rate)) for source in self.sources]
        rate_min, rate_max = min([*rates, 0]), max([*rates, 0])
        cmap = colormaps[HEAT_SOURCE_CMAP]
        for source, shape in shapes:
            if source is None:
                continue
            fraction = (float(np.mean(source.rate)) - rate_min) / (rate_max - rate_min + 1e-5)
            plot_params = plot_params_heat_source.updated_copy(
                alpha=alpha, edgecolor=cmap(fraction)
            )
            ax = self.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        ax = self.add_ax_lims(axis=axis, ax=ax)
        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)
        return Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )

    @cached_property
    def grids(self) -> tuple[Grid, ...]:
        """Cartesian grid of each layer.

        Returns
        -------
        tuple[:class:`.Grid`, ...]
            Grid of each layer, from the lower to the upper simulation bound along the stacking
            axis. A simulation without layers has one grid. Along a zero-size axis, a 2D
            simulation has one cell of 1 µm depth.
        """
        from tidy3d.em.translate.fast_heat import task

        return task.grids(self)

    def plot_grid(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the structures and the cell boundaries of :attr:`grids` on a plane
        defined by one nonzero x, y, z coordinate. Each layer cut by the plane draws its own
        cell boundaries over its extent.

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
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        ax = self.plot(x=x, y=y, z=z, ax=ax)
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis((0, 1, 2), axis=axis)
        for grid in self.grids:
            boundaries = grid.boundaries.to_list
            if not boundaries[axis][0] <= position <= boundaries[axis][-1]:
                continue
            boundaries_x, boundaries_y = boundaries[axis_x], boundaries[axis_y]
            # A zero-size axis has one artificial cell; like Simulation.plot_grid, no lines.
            if self.size[axis_x] > 0:
                ax.vlines(boundaries_x, boundaries_y[0], boundaries_y[-1], **kwargs)
            if self.size[axis_y] > 0:
                ax.hlines(boundaries_y, boundaries_x[0], boundaries_x[-1], **kwargs)
        return ax
