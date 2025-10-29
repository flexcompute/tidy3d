"""Module containing specifications for voltage path integrals."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pydantic.v1 as pd
from typing_extensions import Self

from tidy3d.components.geometry.base import Geometry
from tidy3d.components.microwave.path_integrals.specs.base import (
    AxisAlignedPathIntegralSpec,
    Custom2DPathIntegralSpec,
)
from tidy3d.components.microwave.path_integrals.viz import (
    plot_params_voltage_minus,
    plot_params_voltage_path,
    plot_params_voltage_plus,
)
from tidy3d.components.types import Ax
from tidy3d.components.types.base import Direction
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import fp_eps


class AxisAlignedVoltageIntegralSpec(AxisAlignedPathIntegralSpec):
    """Class for specifying the voltage calculation between two points defined by an axis-aligned line."""

    sign: Direction = pd.Field(
        ...,
        title="Direction of Path Integral",
        description="Positive indicates V=Vb-Va where position b has a larger coordinate along the axis of integration.",
    )

    @classmethod
    def from_terminal_positions(
        cls,
        plus_terminal: float,
        minus_terminal: float,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        extrapolate_to_endpoints: bool = True,
        snap_path_to_grid: bool = True,
    ) -> Self:
        """Helper to create a :class:`AxisAlignedVoltageIntegralSpec` from two coordinates that
        define a line and two positions indicating the endpoints of the path integral.

        Parameters
        ----------
        plus_terminal : float
            Position along the voltage axis of the positive terminal.
        minus_terminal : float
            Position along the voltage axis of the negative terminal.
        x : float = None
            Position in x direction, only two of x,y,z can be specified to define line.
        y : float = None
            Position in y direction, only two of x,y,z can be specified to define line.
        z : float = None
            Position in z direction, only two of x,y,z can be specified to define line.
        extrapolate_to_endpoints: bool = True
            Passed directly to :class:`AxisAlignedVoltageIntegralSpec`
        snap_path_to_grid: bool = True
            Passed directly to :class:`AxisAlignedVoltageIntegralSpec`

        Returns
        -------
        AxisAlignedVoltageIntegralSpec
            The created path integral for computing voltage between the two terminals.
        """
        axis_positions = Geometry.parse_two_xyz_kwargs(x=x, y=y, z=z)
        # Calculate center and size of the future box
        midpoint = (plus_terminal + minus_terminal) / 2
        length = np.abs(plus_terminal - minus_terminal)
        center = [midpoint, midpoint, midpoint]
        size = [length, length, length]
        for axis, position in axis_positions:
            size[axis] = 0
            center[axis] = position

        direction = "+"
        if plus_terminal < minus_terminal:
            direction = "-"

        return cls(
            center=center,
            size=size,
            extrapolate_to_endpoints=extrapolate_to_endpoints,
            snap_path_to_grid=snap_path_to_grid,
            sign=direction,
        )

    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **path_kwargs: Any,
    ) -> Ax:
        """Plot path integral at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **path_kwargs
            Optional keyword arguments passed to the matplotlib plotting of the line.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/36marrat>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis == self.main_axis or not np.isclose(position, self.center[axis], rtol=fp_eps):
            return ax

        (xs, ys) = self._vertices_2D(axis)
        # Plot the path
        plot_params = plot_params_voltage_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        ax.plot(xs, ys, markevery=[0, -1], **plot_kwargs)

        # Plot special end points
        end_kwargs = plot_params_voltage_plus.include_kwargs(**path_kwargs).to_kwargs()
        start_kwargs = plot_params_voltage_minus.include_kwargs(**path_kwargs).to_kwargs()

        if self.sign == "-":
            start_kwargs, end_kwargs = end_kwargs, start_kwargs

        ax.plot(xs[0], ys[0], **start_kwargs)
        ax.plot(xs[1], ys[1], **end_kwargs)
        return ax


class Custom2DVoltageIntegralSpec(Custom2DPathIntegralSpec):
    """Class for specifying the computation of voltage between two points defined by a custom path.
    Computed voltage is :math:`V=V_b-V_a`, where position b is the final vertex in the supplied path.

    Notes
    -----

    Use :class:`.AxisAlignedVoltageIntegralSpec` if possible, since interpolation
    near conductors will not be accurate.

    .. TODO Improve by including extrapolate_to_endpoints field, non-trivial extension."""

    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **path_kwargs: Any,
    ) -> Ax:
        """Plot path integral at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **path_kwargs
            Optional keyword arguments passed to the matplotlib plotting of the line.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/36marrat>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        axis, position = Geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        if axis != self.main_axis or not np.isclose(position, self.position, rtol=fp_eps):
            return ax

        plot_params = plot_params_voltage_path.include_kwargs(**path_kwargs)
        plot_kwargs = plot_params.to_kwargs()
        xs = self.vertices[:, 0]
        ys = self.vertices[:, 1]
        ax.plot(xs, ys, markevery=[0, -1], **plot_kwargs)

        # Plot special end points
        end_kwargs = plot_params_voltage_plus.include_kwargs(**path_kwargs).to_kwargs()
        start_kwargs = plot_params_voltage_minus.include_kwargs(**path_kwargs).to_kwargs()
        ax.plot(xs[0], ys[0], **start_kwargs)
        ax.plot(xs[-1], ys[-1], **end_kwargs)

        return ax
