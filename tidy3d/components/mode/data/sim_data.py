"""Mode solver simulation data"""

from __future__ import annotations

from typing import Literal, Tuple

import pydantic.v1 as pd

from ...base import cached_property
from ...data.monitor_data import ModeSolverData, PermittivityData
from ...data.sim_data import AbstractYeeGridSimulationData
from ...types import (
    Ax,
    PlotScale,
)
from ..simulation import ModeSimulation

ModeSimulationMonitorDataType = PermittivityData


class ModeSimulationData(AbstractYeeGridSimulationData):
    """Data associated with a mode solver simulation."""

    simulation: ModeSimulation = pd.Field(
        ..., title="Mode simulation", description="Mode simulation associated with this data."
    )

    modes_raw: ModeSolverData = pd.Field(
        ...,
        title="Raw Modes",
        description=":class:`.ModeSolverData` containing the field and effective index on unexpanded grid.",
    )

    data: Tuple[ModeSimulationMonitorDataType, ...] = pd.Field(
        (),
        title="Monitor Data",
        description="List of monitor data "
        "associated with the monitors of the original :class:`.ModeSimulation`.",
    )

    @cached_property
    def modes(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data."""
        return self.modes_raw.symmetry_expanded_copy

    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlaid.

        Parameters
        ----------
        field_name : str
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        eps_alpha : float = 0.2
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
            frequency or time dimensions (``f``, ``t``) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.plot_field_monitor_data(
            field_monitor_data=self.modes,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )
