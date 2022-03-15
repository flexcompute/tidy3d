"""Plotly wrapper for tidy3d."""
# pylint:disable=too-many-arguments
from typing import Tuple

import plotly.graph_objects as go
from shapely.geometry.base import BaseGeometry as ShapelyGeo
import numpy as np

from ...components.types import PlotlyFig, Axis
from ...components.simulation import Simulation
from ...components.structure import Structure
from ...components.geometry import Geometry, Box
from ...components.medium import Medium
from ...components.viz import add_fig_if_none, equal_aspect_plotly
from ...components.viz import plot_params_sim_boundary, PlotParams
from ...components.base import Tidy3dBaseModel


class GeometryPlotly(Tidy3dBaseModel):
    """Geometry that adds plotly-based implementations of its standard plotting functions."""

    geometry: Geometry

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
        name: str = None,
    ) -> PlotlyFig:
        """Plot cross sections on plane using plotly."""

        # for each intersection, plot the shape
        for shape in self.geometry.intersections(x=x, y=y, z=z):
            fig = self.plotly_shape(
                shape=shape,
                plot_params=self.geometry.plot_params,
                fig=fig,
                row=row,
                col=col,
                name=name,
            )

        return fig

    def plotly_shape(
        self,
        shape: ShapelyGeo,
        plot_params: PlotParams,
        fig: PlotlyFig,
        row: int = None,
        col: int = None,
        name: str = None,
    ) -> PlotlyFig:
        """Plot a shape to a figure."""
        _shape = self.geometry.evaluate_inf_shape(shape)
        xs, ys = self.geometry._get_shape_coords(shape=shape)
        plotly_trace = go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            fillcolor=plot_params.facecolor,
            line=dict(width=plot_params.linewidth, color=plot_params.facecolor),
            marker=dict(size=0.0001, line=dict(width=0)),
            name=name,
            opacity=plot_params.alpha,
        )
        fig.add_trace(plotly_trace, row=row, col=col)
        return fig


class StructurePlotly(Tidy3dBaseModel):
    """Structure that adds plotly-based implementations of its standard plotting functions."""

    structure: Structure

    @add_fig_if_none
    def plotly(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
        name: str = None,
    ) -> PlotlyFig:
        """Use plotly to plot structure's geometric cross section at single (x,y,z) coordinate."""
        geometry = GeometryPlotly(geometry=self.geometry)
        return geometry.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col)


class SimulationPlotly(Tidy3dBaseModel):
    """Simulation that adds plotly-based implementations of its standard plotting functions."""

    simulation: Simulation

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        Uses plotly.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        fig = self._plotly_bounding_box(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_structures(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_sources(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_monitors(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_symmetries(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_pml(x=x, y=y, z=z, fig=fig, row=row, col=col)
        if row is None and col is None:
            fig = self._plotly_cleanup(x=x, y=y, z=z, fig=fig)

        return fig

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.
        Uses plotly.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        fig = self._plotly_bounding_box(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_structures_eps(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_sources(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_monitors(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_symmetries(x=x, y=y, z=z, fig=fig, row=row, col=col)
        fig = self.plotly_pml(x=x, y=y, z=z, fig=fig, row=row, col=col)
        if row is None and col is None:
            fig = self._plotly_cleanup(x=x, y=y, z=z, fig=fig)

        return fig

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly_structures(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z .

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        medium_shapes = self.simulation._filter_structures_plane(
            self.simulation.structures, x=x, y=y, z=z
        )
        for (medium, shape) in medium_shapes:
            fig = self._plotly_shape_structure(
                medium=medium, shape=shape, fig=fig, row=row, col=col
            )
        return fig

    def _plotly_shape_structure(
        self, medium: Medium, shape: ShapelyGeo, fig: PlotlyFig, row: int = None, col: int = None
    ) -> PlotlyFig:
        """Plot a structure's cross section shape for a given medium."""
        plot_params_struct = self.simulation._get_structure_plot_params(medium=medium)
        mat_index = self.simulation.medium_map[medium]
        name = medium.name if medium.name else f"medium[{mat_index}]"
        geometry = GeometryPlotly(geometry=self.simulation.geometry)
        fig = geometry.plotly_shape(
            shape=shape, plot_params=plot_params_struct, fig=fig, row=row, col=col, name=name
        )
        return fig

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly_structures_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        medium_shapes = self.simulation._filter_structures_plane(
            self.simulation.structures, x=x, y=y, z=z
        )
        for (medium, shape) in medium_shapes:
            fig = self._plotly_shape_structure_eps(
                freq=freq, medium=medium, shape=shape, fig=fig, row=row, col=col
            )
        return fig

    def _plotly_shape_structure_eps(
        self,
        freq: float,
        medium: Medium,
        shape: ShapelyGeo,
        fig: PlotlyFig,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot a structure's cross section shape for a given medium, grayscale for permittivity."""
        plot_params = self.simulation._get_structure_eps_plot_params(medium=medium, freq=freq)
        plot_params.facecolor = f"rgb{tuple(3*[float(plot_params.facecolor)*255])}"
        fig = self.plotly_shape(shape=shape, plot_params=plot_params, fig=fig, row=row, col=col)
        return fig

    @add_fig_if_none
    def plotly_sources(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        for source in self.simulation.sources:
            source_plotly = GeometryPlotly(geometry=source)
            fig = source_plotly.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col, name="sources")
        return fig

    @add_fig_if_none
    def plotly_monitors(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's monitors on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        for monitor in self.simulation.monitors:
            monitor_plotly = GeometryPlotly(geometry=monitor)
            fig = monitor_plotly.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col, name="monitors")
        return fig

    @add_fig_if_none
    def plotly_pml(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's absorbing boundaries
        on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """
        normal_axis, _ = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)
        pml_boxes = self.simulation._make_pml_boxes(normal_axis=normal_axis)
        for pml_box in pml_boxes:
            pml_box_plotly = GeometryPlotly(geometry=pml_box)
            fig = pml_box_plotly.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col)
        return fig

    @add_fig_if_none
    def plotly_symmetries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Plot each of simulation's symmetries on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        fig : plotly.graph_objects.Figure = None
            plotly ``Figure`` to plot on, if not specified, one is created.
        row : int = None
            Index (from 1) of the subplot row to plot on.
        col : int = None
            Index (from 1) of the subplot column to plot on.

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        normal_axis, _ = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)
        sym_boxes = self.simulation._make_symmetry_boxes(normal_axis=normal_axis)
        for sym_box in sym_boxes:
            sym_box_plotly = GeometryPlotly(geometry=sym_box)
            fig = sym_box_plotly.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col)
        return fig

    @add_fig_if_none
    def _plotly_bounding_box(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
        row: int = None,
        col: int = None,
    ) -> PlotlyFig:
        """Add simulation bounding box."""
        rmin, rmax = self.simulation.bounds_pml
        sim_box_pml = Box.from_bounds(rmin=rmin, rmax=rmax)
        sim_box_pml.plot_params = plot_params_sim_boundary
        sim_box_pml_plotly = GeometryPlotly(geometry=sim_box_pml)
        fig = sim_box_pml_plotly.plotly(x=x, y=y, z=z, fig=fig, row=row, col=col)
        return fig

    def _plotly_cleanup(
        self,
        fig: PlotlyFig,
        x: float = None,
        y: float = None,
        z: float = None,
    ) -> PlotlyFig:
        """Finish plotting simulation cross section using plotly."""

        normal_axis, pos = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)

        fig = self._plotly_resize(fig=fig, normal_axis=normal_axis)
        _, (xlabel, ylabel) = self.simulation.pop_axis("xyz", axis=normal_axis)

        fig.update_layout(
            title=f'{"xyz"[normal_axis]} = {pos:.2f}',
            xaxis_title=rf"{xlabel} ($\mu m$)",
            yaxis_title=rf"{ylabel} ($\mu m$)",
            legend_title="Contents",
        )

        fig = self._plotly_clean_labels(fig=fig)

        return fig

    def _plotly_bounds(self, normal_axis: Axis) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """get X, Y limits for plotly figure."""

        rmin, rmax = self.simulation.bounds_pml
        rmin = np.array(rmin)
        rmax = np.array(rmax)
        _, (xmin, ymin) = self.simulation.pop_axis(rmin, axis=normal_axis)
        _, (xmax, ymax) = self.simulation.pop_axis(rmax, axis=normal_axis)
        return (xmin, xmax), (ymin, ymax)

    def _plotly_resize(
        self,
        fig: PlotlyFig,
        normal_axis: Axis,
        width_pixels: float = 700,
    ) -> PlotlyFig:
        """Set the lmits and make equal aspect."""

        (xmin, xmax), (ymin, ymax) = self._plotly_bounds(normal_axis=normal_axis)

        width = xmax - xmin
        height = ymax - ymin

        fig.update_xaxes(range=[xmin - width / 10, xmax + width / 10])
        fig.update_yaxes(range=[ymin - height / 10, ymax + height / 10])

        fig.update_layout(width=float(width_pixels), height=float(width_pixels) * height / width)
        return fig

    @staticmethod
    def _plotly_clean_labels(fig: PlotlyFig) -> PlotlyFig:
        """Remove label entries that show up more than once."""
        seen = []
        for trace in fig["data"]:
            name = trace["name"]
            if name not in seen:
                seen.append(name)
            else:
                trace["showlegend"] = False
        return fig


# from tidy3d.plugins.plotly import SimulationPlotly;from tests.utils import SIM_FULL as sim; sim_plotly = SimulationPlotly(simulation=sim); fig = sim_plotly(x=0);fig.show()
