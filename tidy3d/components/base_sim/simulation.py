"""Abstract base for defining simulation classes of different solvers"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import pydantic.v1 as pd

from .monitor import AbstractSurfaceIntegrationMonitor

from ..base import Tidy3dBaseModel, cached_property
from ..validators import assert_unique_names, assert_objects_in_scene_bounds
from ..geometry.base import Box
from ..geometry.utils import GeometryType
from ..types import Ax, Bound, Axis, Symmetry, TYPE_TAG_STR
from ..structure import Structure
from ..viz import add_ax_if_none, equal_aspect
from ..scene import Scene

from ..viz import PlotParams, plot_params_symmetry

from ...version import __version__
from ...exceptions import Tidy3dKeyError, SetupError, ValidationError


class AbstractSimulation(Tidy3dBaseModel, ABC):
    """Base class for simulation classes of different solvers."""

    scene: Scene = pd.Field(
        ...,
        title="Scene",
        description="Scene describing simulation geometry.",
    )

    domain: Optional[GeometryType] = pd.Field(
        None,
        title="Simulation Domain",
        description="Domain where simulation is performed. "
        "If ``None``, the whole domain of ``scene`` is used.",
        discriminator=TYPE_TAG_STR,
    )

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or "
        "``-1`` (odd, i.e. 'PEC' symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )

    sources: Tuple[None, ...] = pd.Field(
        (),
        title="Sources",
        description="Sources in the simulation.",
    )

    boundary_spec: None = pd.Field(
        None,
        title="Boundaries",
        description="Specification of boundary conditions.",
    )

    monitors: Tuple[None, ...] = pd.Field(
        (),
        title="Monitors",
        description="Monitors in the simulation. ",
    )

    grid_spec: None = pd.Field(
        None,
        title="Grid Specification",
        description="Specifications for the simulation grid.",
    )

    version: str = pd.Field(
        __version__,
        title="Version",
        description="String specifying the front end version number.",
    )

    """ Validating setup """

    # make sure all names are unique
    _unique_monitor_names = assert_unique_names("monitors")
    _monitors_in_bounds = assert_objects_in_scene_bounds("monitors")

    @pd.validator("monitors", always=True)
    def _integration_surfaces_in_bounds(cls, val, values):
        """Error if any of the integration surfaces are outside of the simulation domain."""

        if val is None:
            return val

        if "domain" not in values:
            raise ValidationError(
                "Cannot validate 'monitors' because `.domain` has failed validation"
            )

        sim_domain = values["domain"]

        if sim_domain is None:
            scene = values["scene"]
            sim_box = scene.bounding_box
        else:
            sim_box = sim_domain.bounding_box

        for mnt in (mnt for mnt in val if isinstance(mnt, AbstractSurfaceIntegrationMonitor)):
            if not any(sim_box.intersects(surf) for surf in mnt.integration_surfaces):
                raise SetupError(
                    f"All integration surfaces of monitor '{mnt.name}' are outside of the "
                    "simulation bounds."
                )

        return val

    """ Accounting """

    def get_monitor_by_name(self, name: str):  # -> Monitor:
        """Return monitor named 'name'."""
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise Tidy3dKeyError(f"No monitor named '{name}'")

    @cached_property
    def domain_geometry(self) -> GeometryType:
        """Returns geometry representing the domain of the simulation."""
        if self.domain is not None:
            return self.domain
        return self.scene.bounding_box

    @cached_property
    def domain_structure(self) -> Structure:
        """Returns structure representing the domain of the simulation. This differs from
        `Scene.background_structure` in that it has finite extent."""
        return Structure(geometry=self.domain_geometry, medium=self.scene.medium)

    @cached_property
    def simulation_bounds(self) -> Bound:
        """Returns Box representing the domain of the simulation."""
        # in this default implementation we just take self.domain or scene.bounding_box
        # this should be changed in different solvers depending on whether automatic extensions
        # (like pml) are present
        return self.domain_geometry.bounds

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **patch_kwargs,
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
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.scene.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.simulation_bounds
        for source in self.sources:
            ax = source.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_monitors(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's monitors on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.simulation_bounds
        for monitor in self.monitors:
            ax = monitor.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_symmetries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's symmetries on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        normal_axis, _ = Box.parse_xyz_kwargs(x=x, y=y, z=z)

        for sym_axis, sym_value in enumerate(self.symmetry):
            if sym_value == 0 or sym_axis == normal_axis:
                continue
            sym_box = self._make_symmetry_box(sym_axis=sym_axis)
            plot_params = self._make_symmetry_plot_params(sym_value=sym_value)
            ax = sym_box.plot(x=x, y=y, z=z, ax=ax, **plot_params.to_kwargs())
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    def _make_symmetry_plot_params(self, sym_value: Symmetry) -> PlotParams:
        """Make PlotParams for symmetry."""

        plot_params = plot_params_symmetry.copy()

        if sym_value == 1:
            plot_params = plot_params.copy(
                update={"facecolor": "lightsteelblue", "edgecolor": "lightsteelblue", "hatch": "++"}
            )
        elif sym_value == -1:
            plot_params = plot_params.copy(
                update={"facecolor": "goldenrod", "edgecolor": "goldenrod", "hatch": "--"}
            )

        return plot_params

    def _make_symmetry_box(self, sym_axis: Axis) -> Box:
        """Construct a :class:`.Box` representing the symmetry to be plotted."""
        rmin, rmax = (list(bound) for bound in self.simulation_bounds)
        rmax[sym_axis] = (rmin[sym_axis] + rmax[sym_axis]) / 2

        return Box.from_bounds(rmin, rmax)

    def _set_plot_bounds(
        self,
        ax: Ax,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Sets the xy limits of the simulation at a plane, useful after plotting.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to set bounds on.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The axes after setting the boundaries.
        """
        return Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

    @abstractmethod
    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the simulation boundary conditions as lines on a plane
           defined by one nonzero x,y,z coordinate.

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
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
