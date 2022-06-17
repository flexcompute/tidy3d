"""Simulation and Geometry plotting with plotly."""
# pylint:disable=too-many-arguments, protected-access
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from shapely.geometry.base import BaseGeometry as ShapelyGeo
import pydantic as pd

from .component import UIComponent
from .utils import PlotlyFig, add_fig_if_none, equal_aspect_plotly, plot_params_sim_boundary
from ...components.base import Tidy3dBaseModel
from ...components.geometry import Geometry, Box
from ...components.medium import Medium
from ...components.simulation import Simulation
from ...components.structure import Structure
from ...components.types import Axis
from ...components.viz import PlotParams, plot_params_pml


def plotly_shape(
    shape: ShapelyGeo,
    plot_params: PlotParams,
    fig: PlotlyFig,
    name: str = None,
) -> PlotlyFig:
    """Plot a shape to a figure."""
    _shape = Geometry.evaluate_inf_shape(shape)
    exterior_coords, _ = Geometry.strip_coords(shape=_shape)
    xs, ys = list(zip(*exterior_coords))
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
    fig.add_trace(plotly_trace)
    return fig


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
        name: str = None,
        **patch_kwargs,
    ) -> PlotlyFig:
        """Plot cross sections on plane using plotly."""

        plot_params = self.geometry.plot_params.copy(deep=True)
        plot_params_struct = plot_params.include_kwargs(**patch_kwargs)

        # for each intersection, plot the shape
        for shape in self.geometry.intersections(x=x, y=y, z=z):
            fig = plotly_shape(
                shape=shape,
                plot_params=plot_params_struct,
                fig=fig,
                name=name,
            )

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
        name: str = None,
        **patch_kwargs,
    ) -> PlotlyFig:
        """Use plotly to plot structure's geometric cross section at single (x,y,z) coordinate."""
        geometry = GeometryPlotly(geometry=self.structure.geometry)
        return geometry.plotly(x=x, y=y, z=z, fig=fig, name=name, **patch_kwargs)


class SimulationPlotly(UIComponent):
    """Simulation that adds plotly-based implementations of its standard plotting functions."""

    simulation: Simulation = pd.Field(
        ..., title="Simulation", description="The Simulation instance to plot."
    )

    cs_axis: Axis = pd.Field(
        0,
        title="Cross-section axis",
        description="The axis (0,1,2) representing the plotting plane normal direction.",
    )

    cs_val: float = pd.Field(
        None,
        title="Cross-section value",
        description="The position along the plotting plane axis normal.",
    )

    @property
    def xyz_label_bounds(self):
        """Get the plot normal axis label and the min and max bounds."""

        xyz_label = "xyz"[self.cs_axis]
        bmin, bmax = self.simulation.bounds
        xyz_min = bmin[self.cs_axis]
        xyz_max = bmax[self.cs_axis]
        return xyz_label, (xyz_min, xyz_max)

    def make_figure(self):
        """Generate plotly figure from the current state of self."""

        xyz_label, (xyz_min, xyz_max) = self.xyz_label_bounds
        if self.cs_val is None:
            self.cs_val = (xyz_min + xyz_max) / 2.0
        plotly_kwargs = {xyz_label: self.cs_val}
        return self.plotly(**plotly_kwargs)

    def make_component(self):  # pylint: disable=too-many-locals
        """Creates the dash component."""

        xyz_label, (xyz_min, xyz_max) = self.xyz_label_bounds
        figure = self.make_figure()

        graph = html.Div(
            [dcc.Graph(figure=figure, id="simulation_plot")], style={"padding": 10, "flex": 1}
        )

        xyz_header = html.H2("Cross-section axis and position.")

        xyz_dropdown = dcc.Dropdown(
            options=["x", "y", "z"],
            value=xyz_label,
            id="simulation_cs_axis_dropdown",
        )

        xyz_slider = dcc.Slider(
            min=xyz_min,
            max=xyz_max,
            value=self.cs_val,
            id="simulation_cs_slider",
        )

        xyz_selection = html.Div(
            [xyz_header, xyz_dropdown, xyz_slider],
            style={"padding": 50, "flex": 1},
        )

        return dcc.Tab(
            [
                html.H1("Viewing Simulation."),
                html.Div([graph, xyz_selection], style={"display": "flex", "flexDirection": "row"}),
            ],
            label="Simulation",
        )

    @equal_aspect_plotly
    @add_fig_if_none
    def plotly(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
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

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        # fig = self._plotly_bounding_box(x=x, y=y, z=z, fig=fig)
        fig = self.plotly_structures(x=x, y=y, z=z, fig=fig)
        fig = self.plotly_sources(x=x, y=y, z=z, fig=fig)
        fig = self.plotly_monitors(x=x, y=y, z=z, fig=fig)
        fig = self.plotly_symmetries(x=x, y=y, z=z, fig=fig)
        fig = self.plotly_pml(x=x, y=y, z=z, fig=fig)
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

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        structures = self.simulation.structures
        medium_shapes = self.simulation._get_structures_plane(structures=structures, x=x, y=y, z=z)
        for (medium, shape) in medium_shapes:
            fig = self._plotly_shape_structure(medium=medium, shape=shape, fig=fig)
        return fig

    def _plotly_shape_structure(
        self, medium: Medium, shape: ShapelyGeo, fig: PlotlyFig
    ) -> PlotlyFig:
        """Plot a structure's cross section shape for a given medium."""
        mat_index = self.simulation.medium_map[medium]
        plot_params_struct = self.simulation._get_structure_plot_params(
            medium=medium, mat_index=mat_index
        )
        name = medium.name or f"medium[{mat_index}]"
        fig = plotly_shape(shape=shape, plot_params=plot_params_struct, fig=fig, name=name)
        return fig

    @add_fig_if_none
    def plotly_sources(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
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

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        for source in self.simulation.sources:
            source_plotly = GeometryPlotly(geometry=source)
            fig = source_plotly.plotly(x=x, y=y, z=z, fig=fig, name="sources")
        return fig

    @add_fig_if_none
    def plotly_monitors(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
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

        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        for monitor in self.simulation.monitors:
            monitor_plotly = GeometryPlotly(geometry=monitor)
            fig = monitor_plotly.plotly(x=x, y=y, z=z, fig=fig, name="monitors")
        return fig

    @add_fig_if_none
    def plotly_pml(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
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


        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """
        normal_axis, _ = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)
        pml_boxes = self.simulation._make_pml_boxes(normal_axis=normal_axis)
        for pml_box in pml_boxes:
            pml_box_plotly = GeometryPlotly(geometry=pml_box)
            fig = pml_box_plotly.plotly(x=x, y=y, z=z, fig=fig, **plot_params_pml.to_kwargs())
        return fig

    @add_fig_if_none
    def plotly_symmetries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
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


        Returns
        -------
        plotly.graph_objects.Figure
            The supplied or created plotly ``Figure``.
        """

        normal_axis, _ = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)

        for sym_axis, sym_value in enumerate(self.simulation.symmetry):
            if sym_value == 0 or sym_axis == normal_axis:
                continue
            sym_box = self.simulation._make_symmetry_box(sym_axis=sym_axis)
            plot_params = self.simulation._make_symmetry_plot_params(sym_value=sym_value)
            sym_box_plotly = GeometryPlotly(geometry=sym_box)
            fig = sym_box_plotly.plotly(x=x, y=y, z=z, fig=fig, **plot_params.to_kwargs())

        return fig

    @add_fig_if_none
    def _plotly_bounding_box(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        fig: PlotlyFig = None,
    ) -> PlotlyFig:
        """Add simulation bounding box."""
        rmin, rmax = self.simulation.bounds_pml
        sim_box_pml = Box.from_bounds(rmin=rmin, rmax=rmax)
        sim_box_pml_plotly = GeometryPlotly(geometry=sim_box_pml)
        fig = sim_box_pml_plotly.plotly(
            x=x, y=y, z=z, fig=fig, **plot_params_sim_boundary.to_kwargs()
        )
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
            title_text=f'Simulation plotted on {"xyz"[normal_axis]} = {pos:.2f} cross-section',
            title_x=0.5,
            xaxis_title=f"{xlabel} (um)",
            yaxis_title=f"{ylabel} (um)",
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
    ) -> PlotlyFig:
        """Set the lmits and make equal aspect."""

        (xmin, xmax), (ymin, ymax) = self._plotly_bounds(normal_axis=normal_axis)

        fig.update_xaxes(range=[xmin, xmax])
        fig.update_yaxes(range=[ymin, ymax])

        return fig

    @staticmethod
    def _plotly_clean_labels(fig: PlotlyFig) -> PlotlyFig:
        """Remove label entries that show up more than once."""
        seen = []
        for trace in fig["data"]:
            name = trace["name"]
            if name is not None and name not in seen:
                seen.append(name)
            else:
                trace["showlegend"] = False
        return fig
