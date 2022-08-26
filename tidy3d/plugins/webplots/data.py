"""Defines how the specific data objects render as UI components."""
from abc import ABC
from typing import Union, Tuple, List, Dict

import numpy as np
import plotly.graph_objects as go
import pydantic as pd
from dash import dcc, html, Input
from typing_extensions import Literal

from .component import UIComponent
from .utils import PlotlyFig
from ...components.data import FluxData, FluxTimeData, FieldData, FieldTimeData
from ...components.data import ModeSolverData, ModeData, ScalarFieldDataArray
from ...components.geometry import Geometry
from ...components.types import Axis, Direction
from ...log import Tidy3dKeyError, log

# divide an SI unit by these to put in the unit in the name
#    eg.  1e-12 (sec) / PICOSECOND = 1 (ps)
# multiply a value in the specified unit by these to put in SI
#    eg.  1 (THz) * TERAHERTZ = 1e12 (Hz)
PICOSECOND = 1e-12
TERAHERTZ = 1e12

# supported data types
Tidy3dBaseModelType = Union[
    FluxData, FluxTimeData, FieldData, FieldTimeData, ModeData, ModeSolverData
]


class DataPlotly(UIComponent, ABC):
    """Base class for anything that generates dash components from tidy3d data objects."""

    monitor_name: str = pd.Field(..., tite="monitor name", description="Name of the monitor.")
    data: Tidy3dBaseModelType = pd.Field(
        ..., tite="data", description="The Tidy3dBaseModel object wrapped by this UI component."
    )

    @property
    def label(self) -> str:
        """Get tab label for this component."""
        return f"monitor: '{self.monitor_name}'"

    # @property
    # def unique_id(self) -> str:
    #     """Get unique id for this component."""
    #     return f"monitor_{self.monitor_name}"

    @staticmethod
    def sel_by_val(data, val: str) -> Tidy3dBaseModelType:
        """Select the correct data type based on the `val` selection."""
        if val.lower() == "real":
            return data.real
        if val.lower() == "imag":
            return data.imag
        if val.lower() == "abs":
            return abs(data)
        if val.lower() == "abs^2":
            return abs(data) ** 2
        raise ValueError(f"Could not find the right function to apply with {val}.")

    def append_monitor_name(self, value: str) -> Dict[str, str]:
        """Adds the monitor name to a value, used to make the ids unique across all monitors."""
        return {"type": f"{type(self.data).__name__}_{value}", "name": self.monitor_name}

    @classmethod
    def from_monitor_data(
        cls, monitor_name: str, monitor_data: Tidy3dBaseModelType, **kwargs
    ) -> Tidy3dBaseModelType:
        """Load a PlotlyData UI component from the monitor name and its data."""

        # maps the supplied ``monitor_data`` argument to the corresponding plotly wrapper.
        data_plotly_map = (
            (FluxData, FluxDataPlotly),
            (FluxTimeData, FluxTimeDataPlotly),
            (FieldData, FieldDataPlotly),
            (FieldTimeData, FieldTimeDataPlotly),
            (ModeSolverData, ModeSolverDataPlotly),
            (ModeData, ModeDataPlotly),
        )

        # get the type of the supplied ``monitor_data``.
        monitor_data_type = type(monitor_data)

        for mnt_data, mnt_data_plotly in data_plotly_map:
            if isinstance(monitor_data, mnt_data):
                return mnt_data_plotly(data=monitor_data, monitor_name=monitor_name, **kwargs)

        log.warning(
            f"could not find a plotly wrapper for monitor {monitor_name}"
            f"of type {monitor_data_type.__name__}"
        )
        return None


class AbstractFluxDataPlotly(DataPlotly, ABC):
    """Flux data in frequency or time domain."""

    data: Union[FluxData, FluxTimeData] = pd.Field(
        ...,
        title="data",
        description="A flux data object in freq or time domain.",
    )

    @property
    def ft_label_coords_units(self) -> Tuple[str, List[float], str]:
        """Get the ``freq`` or ``time`` label, coords, and units depending on the data contents."""

        if "f" in self.data.data.coords:
            ft_label = "freq"
            ft_coords = self.data.data.coords["f"].values / TERAHERTZ
            ft_units = "THz"
        elif "t" in self.data.data.coords:
            ft_label = "time"
            ft_coords = self.data.data.coords["t"].values / PICOSECOND
            ft_units = "ps"
        else:
            raise Tidy3dKeyError("Neither frequency nor time data found in this data object.")

        return ft_label, ft_coords, ft_units

    def make_figure(self) -> PlotlyFig:
        """Generate plotly figure from the current state of self."""
        return self.plotly()

    def make_component(self) -> dcc.Tab:
        """Creates the dash component for this montor data."""

        # initital setup
        fig = self.make_figure()

        # individual components
        flux_plot = html.Div(
            [
                dcc.Graph(
                    id=self.append_monitor_name("figure"),
                    figure=fig,
                )
            ],
            style={"padding": 10, "flex": 1},
        )

        # return the layout of the component so the app can insert it
        return dcc.Tab(
            [
                html.H1(f"Viewing data for {type(self.data).__name__}: '{self.monitor_name}'"),
                flux_plot,
            ],
            label=self.label,
        )

    def plotly(self) -> PlotlyFig:
        """Generate the plotly figure for this component."""

        ft_label, ft_coords, ft_units = self.ft_label_coords_units
        ft_units = "THz" if "f" in ft_label else "ps"

        fig = go.Figure(go.Scatter(x=ft_coords, y=self.data.data.values))

        fig.update_layout(
            xaxis_title=f"{ft_label} ({ft_units})",
            yaxis_title="Flux (normalized)",
        )

        fig.update_layout(yaxis=dict(showexponent="all", exponentformat="e"))

        return fig


class FluxDataPlotly(AbstractFluxDataPlotly):
    """Flux in frequency domain."""

    data: FluxData = pd.Field(
        ...,
        title="data",
        description="A flux data object in the frequency domain.",
    )


class FluxTimeDataPlotly(AbstractFluxDataPlotly):
    """Flux in time domain."""

    data: FluxTimeData = pd.Field(
        ...,
        title="data",
        description="A flux data object in the time domain.",
    )


class ModeDataPlotly(DataPlotly):
    """Flux data in frequency or time domain."""

    data: ModeData = pd.Field(
        ...,
        title="data",
        description="A mode amplitude data object",
    )

    amps_or_neff: Literal["amps", "neff"] = pd.Field(
        "amps",
        title="Amps or effective index value",
        description="The state of the component's 'amplitude or neff' value.",
    )

    val: str = pd.Field(
        "abs^2",
        title="Plotting value value",
        description="The state of the component's plotting value value.",
    )

    dir_val: Direction = pd.Field(
        None, title="Direction value", description="The state of the component's direction value."
    )

    mode_ind_val: int = pd.Field(
        None, title="Mode index value", description="The state of the component's mode index value."
    )

    @property
    def mode_ind_coords(self) -> List[float]:
        """Get the mode indices."""
        return self.data.amps.coords["mode_index"].values

    @property
    def direction_coords(self) -> List[float]:
        """Get the mode indices."""
        return self.data.amps.coords["direction"].values

    @property
    def ft_label_coords_units(self) -> Tuple[str, List[float], str]:
        """Get the `freq` or `time` label and coords."""

        ft_label = "freq"
        ft_coords = self.data.amps.coords["f"].values / TERAHERTZ
        ft_units = "THz"
        return ft_label, ft_coords, ft_units

    @property
    def dir_dropdown_hidden(self) -> bool:
        """Should the dropdown be hidden?"""
        return self.amps_or_neff == "neff"

    def make_figure(self) -> PlotlyFig:
        """Generate plotly figure from the current state of self."""

        if self.dir_val is None:
            self.dir_val = self.direction_coords[0]

        if self.mode_ind_val is None:
            self.mode_ind_val = self.mode_ind_coords[0]

        if self.amps_or_neff == "amps":
            return self.plotly_amps(mode_index=self.mode_ind_val, dir_val=self.dir_val)

        return self.plotly_neff(mode_index=self.mode_ind_val)

    def make_component(self) -> dcc.Tab:
        """Creates the dash component for this montor data."""

        # initital setup
        fig = self.make_figure()

        # individual components

        # amplitude plot
        amp_plot = html.Div(
            [
                dcc.Graph(
                    id=self.append_monitor_name("figure"),
                    figure=fig,
                )
            ],
            style={"padding": 10, "flex": 1},
        )

        # select amps or neff
        amps_or_neff_dropdown = dcc.Dropdown(
            options=["amps", "neff"],
            value=self.amps_or_neff,
            id=self.append_monitor_name("amps_or_neff_dropdown"),
        )

        # select real, abs, imag, power
        field_value_dropdown = dcc.Dropdown(
            options=["real", "imag", "abs^2"],
            value=self.val,
            id=self.append_monitor_name("val_dropdown"),
        )

        # header for direction dropdown
        dir_dropdown_header = html.Div(
            [html.H2("Direction: forward (+) or backward (-).")],
            hidden=self.dir_dropdown_hidden,
            id=self.append_monitor_name("dir_dropdown_header"),
        )

        # select direction
        dir_value_dropdown = html.Div(
            [
                dcc.Dropdown(
                    options=list(self.direction_coords),
                    value=self.dir_val,
                    id=self.append_monitor_name("dir_dropdown"),
                )
            ],
            hidden=self.dir_dropdown_hidden,
            id=self.append_monitor_name("dir_dropdown_div"),
        )

        # mode index selector
        mode_ind_dropdown = html.Div(
            [
                dcc.Dropdown(
                    options=list(self.mode_ind_coords),
                    value=self.mode_ind_val,
                    id=self.append_monitor_name("mode_index_selector"),
                ),
            ]
        )

        # data control panel
        panel_children = [
            html.H2("Amplitude or effective index."),
            amps_or_neff_dropdown,
            html.H2("Value to plot."),
            field_value_dropdown,
            dir_dropdown_header,
            dir_value_dropdown,
            html.H2("Mode Index."),
            mode_ind_dropdown,
        ]

        plot_selections = html.Div(panel_children, style={"padding": 10, "flex": 1})

        # define layout
        component = dcc.Tab(
            [
                html.H1(f"Viewing data for {type(self.data).__name__}: '{self.monitor_name}'"),
                html.Div(
                    [
                        # left hand side
                        amp_plot,
                        # right hand side
                        plot_selections,
                    ],
                    # make elements in above list stack row-wise
                    style={"display": "flex", "flexDirection": "row"},
                ),
            ],
            label=self.label,
        )

        return component

    def plotly_amps(self, mode_index: int, dir_val: str):
        """Make a line chart for the mode amplitudes."""

        ft_label, ft_coords, ft_units = self.ft_label_coords_units
        ft_units = "THz" if "f" in ft_label else "ps"

        amp_val = self.sel_by_val(self.data.amps, val=self.val)
        amp_val = amp_val.sel(direction=dir_val, mode_index=mode_index)
        fig = go.Figure(go.Scatter(x=ft_coords, y=amp_val))

        fig.update_layout(
            title_text=f"amplitudes of mode w/ index {self.mode_ind_val} in {dir_val} direction.",
            title_x=0.5,
            xaxis_title=f"{ft_label} ({ft_units})",
            yaxis_title="Amplitude (normalized)",
        )
        return fig

    def plotly_neff(self, mode_index: int):
        """Make a line chart for the mode amplitudes."""

        ft_label, ft_coords, ft_units = self.ft_label_coords_units
        ft_units = "THz" if "f" in ft_label else "ps"

        neff_val = self.sel_by_val(self.data.n_complex, val=self.val)
        neff_val = neff_val.sel(mode_index=mode_index)
        fig = go.Figure(go.Scatter(x=ft_coords, y=neff_val))

        fig.update_layout(
            title_text=f"effective index of mode w/ index {self.mode_ind_val}",
            title_x=0.5,
            xaxis_title=f"{ft_label} ({ft_units})",
            yaxis_title="Effective index",
        )
        return fig


class AbstractFieldDataPlotly(DataPlotly, ABC):
    """Some kind of field-like data plotted in the app."""

    data: Union[FieldData, FieldTimeData] = pd.Field(
        ...,
        title="data",
        description="A Field-like data object.",
    )

    field_val: str = pd.Field(
        None, title="Field value", description="The component's field component value."
    )

    val: str = pd.Field(
        "abs", title="Plot value value", description="The component's plotting value value."
    )

    cs_axis: Axis = pd.Field(
        None,
        title="Cross section axis value",
        description="The component's cross section axis value.",
    )

    cs_val: float = pd.Field(
        None,
        title="Cross section position value",
        description="The component's cross section position value.",
    )

    ft_val: float = pd.Field(
        None, title="Freq or time value", description="The component's frequency or time value."
    )

    mode_ind_val: int = pd.Field(
        None, title="Mode index value", description="The component's mode index value value."
    )

    @property
    def ft_label_coords_units(self) -> Tuple[str, List[float], str]:
        """Get the `freq` or `time` label and coords."""

        scalar_field_data = self.data.data_dict[self.field_val]

        if "f" in self.scalar_field_data.data.coords:
            ft_label = "freq"
            ft_coords = scalar_field_data.data.coords["f"].values / TERAHERTZ
            ft_units = "THz"
        elif "t" in self.scalar_field_data.data.coords:
            ft_label = "time"
            ft_coords = scalar_field_data.data.coords["t"].values / PICOSECOND
            ft_units = "ps"
        else:
            raise Tidy3dKeyError("neither frequency nor time data found in this data object.")

        return ft_label, ft_coords, ft_units

    @property
    def inital_field_val(self):
        """The starting field value."""
        field_vals = list(self.data.data_dict.keys())
        if not field_vals:
            raise ValueError("Data doesn't have any field components stored.")
        return field_vals[0]

    @property
    def scalar_field_data(self) -> ScalarFieldDataArray:
        """The current scalar field monitor data."""
        if self.field_val is None:
            self.field_val = self.inital_field_val

        return self.data.data_dict[self.field_val]

    @property
    def inital_cs_axis(self) -> List[int]:
        """Returns the cross section axis that plots the 2D view."""
        coords = self.scalar_field_data.data.coords
        coords_xyz = [coords[xyz_label].values for xyz_label in "xyz"]
        has_volume = [len(coord) > 3 for coord in coords_xyz]

        # if a 2D view
        if sum(has_volume) == 2:
            # initialize with the cross section axis set up to display the 2D plot
            return has_volume.index(False)

        # otherwise, just initialize with x as cross section axis.
        return 0

    @property
    def xyz_label_coords(self) -> Tuple[str, List[float]]:
        """Get the plane normal direction label and coords."""
        if self.cs_axis is None:
            self.cs_axis = self.inital_cs_axis
        xyz_label = "xyz"[self.cs_axis]
        xyz_coords = self.scalar_field_data.data.coords[xyz_label].values
        return xyz_label, xyz_coords

    @property
    def mode_ind_coords(self) -> List[int]:
        """Get the mode indices."""
        if "mode_index" in self.scalar_field_data.data.coords:
            return self.scalar_field_data.data.coords["mode_index"].values
        return None

    def make_figure(self) -> PlotlyFig:
        """Generate plotly figure from the current state of self."""

        # if no field specified, use the first one in the fields list
        if self.field_val is None:
            self.field_val = self.inital_field_val

        if self.cs_axis is None:
            self.cs_axis = self.inital_cs_axis

        # if no mode_ind_val specified, use the first of the coords (or None)
        if self.mode_ind_val is None:
            mode_indices = self.mode_ind_coords
            self.mode_ind_val = mode_indices[0] if mode_indices is not None else None

        # if cross section value, use the average of the coordinates of the current axis
        xyz_label, xyz_coords = self.xyz_label_coords
        if self.cs_val is None:
            self.cs_val = np.mean(xyz_coords)

        # if no freq or time value, use the average of the coordinates
        ft_label, ft_coords, _ = self.ft_label_coords_units
        if self.ft_val is None:
            self.ft_val = np.mean(ft_coords)

        plotly_kwargs = {
            xyz_label: self.cs_val,
            "field": self.field_val,
            ft_label: self.ft_val,
            "val": self.val,
            "mode_index": self.mode_ind_val,
        }

        return self.plotly(**plotly_kwargs)

    def make_component(self) -> dcc.Tab:  # pylint:disable=too-many-locals
        """Creates the dash component."""

        # initial setup
        xyz_label, xyz_coords = self.xyz_label_coords
        ft_label, ft_coords, ft_units = self.ft_label_coords_units
        fig = self.make_figure()

        # individual components

        # plot of the fields
        field_plot = html.Div(
            [
                dcc.Graph(
                    id=self.append_monitor_name("figure"),
                    figure=fig,
                )
            ],
            style={"padding": 10, "flex": 1},
        )

        # pick the field component
        field_dropdown = dcc.Dropdown(
            options=list(self.data.data_dict.keys()),
            value=self.field_val,
            id=self.append_monitor_name("field_dropdown"),
        )

        # pick the real / imag / abs to plot
        field_value_dropdown = dcc.Dropdown(
            options=["real", "imag", "abs"],
            value=self.val,
            id=self.append_monitor_name("val_dropdown"),
        )

        # pick the cross section axis
        xyz_dropdown = dcc.Dropdown(
            options=["x", "y", "z"],
            value=xyz_label,
            id=self.append_monitor_name("cs_axis_dropdown"),
        )

        # pick the cross section position
        xyz_slider = dcc.Slider(
            min=xyz_coords[0],
            max=xyz_coords[-1],
            value=self.cs_val,
            id=self.append_monitor_name("cs_slider"),
        )

        # combine the cross section selection into one component
        xyz_selection = html.Div([xyz_dropdown, xyz_slider])

        # pick the frequency or time
        freq_time_slider = html.Div(
            [
                dcc.Slider(
                    min=ft_coords[0],
                    max=ft_coords[-1],
                    value=self.ft_val,
                    id=self.append_monitor_name("ft_slider"),
                ),
            ]
        )

        # all the controls for adjusting plotted data
        plot_selections = html.Div(
            [
                html.H2("Field component."),
                field_dropdown,
                html.H2("Value to plot."),
                field_value_dropdown,
                html.H2("Cross-section axis and position."),
                xyz_selection,
                html.H2(f"{ft_label} value ({ft_units})."),
                freq_time_slider,
            ],
            style={"padding": 10, "flex": 1},
        )

        # add a mode index dropdown to right hand side, if applicable
        if self.mode_ind_val is not None:
            # make a mode index label and dropdown
            mode_ind_label = html.H2("Mode Index component.")
            mode_ind_dropdown = html.Div(
                [
                    dcc.Dropdown(
                        options=list(self.mode_ind_coords),
                        value=self.mode_ind_val,
                        id=self.append_monitor_name("mode_index_selector"),
                    ),
                ]
            )

            # add these to the plot selections panel component
            plot_selections.children.append(mode_ind_label)
            plot_selections.children.append(mode_ind_dropdown)

        # full layout
        component = dcc.Tab(
            [
                # title
                html.H1(f"Viewing data for {type(self.data).__name__}: '{self.monitor_name}'"),
                # below title
                html.Div(
                    [
                        # left hand side
                        field_plot,
                        # right hand side
                        plot_selections,
                    ],
                    # make elements in above list stack row-wise
                    style={"display": "flex", "flexDirection": "row"},
                ),
            ],
            # label for the tab
            label=self.label,
        )

        # these are the inputs to the callback function which links the buttons to the figure
        app_inputs = [
            Input(self.append_monitor_name("field_dropdown"), "value"),
            Input(self.append_monitor_name("val_dropdown"), "value"),
            Input(self.append_monitor_name("cs_axis_dropdown"), "value"),
            Input(self.append_monitor_name("cs_slider"), "value"),
            Input(self.append_monitor_name("ft_slider"), "value"),
        ]

        # add the mode index dropdown to the app inputs, if defined
        if self.mode_ind_val is not None:
            app_inputs.append(Input(self.append_monitor_name("mode_index_selector"), "value"))

        # link what happens in the app_inputs to what gets displayed in the figure

        return component

    def plotly(  # pylint:disable=too-many-arguments, too-many-locals
        self,
        field: str,
        val: Literal["real", "imag", "abs"],
        freq: float = None,
        time: float = None,
        x: float = None,
        y: float = None,
        z: float = None,
        mode_index: int = None,
    ) -> PlotlyFig:
        """Creates the plotly figure given some parameters."""

        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)

        # grab by field name
        scalar_field_data = self.scalar_field_data.data

        # select mode_index, if given
        if mode_index is not None and "mode_index" in scalar_field_data.coords:
            scalar_field_data = scalar_field_data.sel(mode_index=mode_index)

        # select by frequency, if given
        if freq is not None:
            freq *= TERAHERTZ
            sel_ft = scalar_field_data.sel(f=freq, method="nearest")

        # select by time, if given
        if time is not None:
            time *= PICOSECOND
            sel_ft = scalar_field_data.sel(t=time, method="nearest")

        # select the cross sectional plane data
        xyz_labels = ["x", "y", "z"]
        xyz_kwargs = {xyz_labels.pop(axis): position}
        sel_xyz = sel_ft.sel(**xyz_kwargs, method="nearest")

        # get the correct field value (real, imaginary, abs)
        sel_val = self.sel_by_val(data=sel_xyz, val=val)

        # get the correct x and y labels
        coords_plot_x = sel_val.coords[xyz_labels[0]]
        coords_plot_y = sel_val.coords[xyz_labels[1]]

        # construct the field plot
        fig = go.Figure(
            data=go.Heatmap(
                x=coords_plot_x,
                y=coords_plot_y,
                z=sel_val.values,
                transpose=True,
                type="heatmap",
                colorscale="magma" if val in "abs" in val else "RdBu",
            )
        )

        # update title and x and y labels.
        ft_text = f"f={freq:.2e}" if freq is not None else f"t={time:.2e}"
        _, (xlabel, ylabel) = Geometry.pop_axis("xyz", axis=axis)
        fig.update_layout(
            title_text=f'{val}[{field}({"xyz"[axis]}={position:.2e}, {ft_text})]',
            title_x=0.5,
            xaxis_title=f"{xlabel} (um)",
            yaxis_title=f"{ylabel} (um)",
        )

        return fig


class FieldDataPlotly(AbstractFieldDataPlotly):
    """Plot :class:`.FieldData` in app."""

    data: FieldData = pd.Field(
        ...,
        title="data",
        description="A field data object in the frequency domain",
    )


class FieldTimeDataPlotly(AbstractFieldDataPlotly):
    """Plot :class:`.FieldTimeData` in app."""

    data: FieldTimeData = pd.Field(
        ...,
        title="data",
        description="A field data object in the time domain.",
    )


class ModeSolverDataPlotly(AbstractFieldDataPlotly):
    """Plot :class:`.ModeSolverData` in app."""

    data: ModeSolverData = pd.Field(
        ...,
        title="data",
        description="A mode field object.",
    )
