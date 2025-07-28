"""Base class for generating an S matrix automatically from tidy3d simulations and port definitions."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Union, get_args

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import DataArray
from tidy3d.components.simulation import Simulation
from tidy3d.components.types import FreqArray
from tidy3d.constants import HERTZ
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from tidy3d.log import log
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.types import TerminalPortType
from tidy3d.plugins.smatrix.ports.wave import WavePort

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "."


class AbstractComponentModeler(ABC, Tidy3dBaseModel):
    """Tool for modeling devices and computing port parameters."""

    name: str = pd.Field(
        "",
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )
    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )

    ports: tuple[Union[Port, TerminalPortType], ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )

    freqs: FreqArray = pd.Field(
        ...,
        title="Frequencies",
        description="Array or list of frequencies at which to compute port parameters.",
        units=HERTZ,
    )

    remove_dc_component: bool = pd.Field(
        True,
        title="Remove DC Component",
        description="Whether to remove the DC component in the Gaussian pulse spectrum. "
        "If ``True``, the Gaussian pulse is modified at low frequencies to zero out the "
        "DC component, which is usually desirable so that the fields will decay. However, "
        "for broadband simulations, it may be better to have non-vanishing source power "
        "near zero frequency. Setting this to ``False`` results in an unmodified Gaussian "
        "pulse spectrum which can have a nonzero DC component.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("'AbstractComponentModeler.simulation' must not have any sources.")
        return val

    @pd.validator("ports", always=True)
    def _warn_rf_license(cls, val):
        """Warn about new licensing requirements for RF ports."""
        rf_port = False
        TerminalPortTypeTuple = get_args(TerminalPortType)
        for port in val:
            if type(port) in TerminalPortTypeTuple:
                rf_port = True
                break
        if rf_port:
            log.warning(
                "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
                log_once=True,
            )
        return val

    @staticmethod
    def get_task_name(port: Port, mode_index: Optional[int] = None) -> str:
        """The name of a task, determined by the port of the source and mode index, if given."""
        if mode_index is not None:
            return f"smatrix_{port.name}_{mode_index}"
        return f"smatrix_{port.name}"

    def get_port_by_name(self, port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in self.ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @staticmethod
    def inv(matrix: DataArray):
        """Helper to invert a port matrix."""
        return np.linalg.inv(matrix)

    def _shift_value_signed(self, port: Union[Port, WavePort]) -> float:
        """How far (signed) to shift the source from the monitor."""

        # get the grid boundaries and sizes along port normal from the simulation
        normal_axis = port.size.index(0.0)
        grid = self.simulation.grid
        grid_boundaries = grid.boundaries.to_list[normal_axis]
        grid_centers = grid.centers.to_list[normal_axis]

        # get the index of the grid cell where the port lies
        port_position = port.center[normal_axis]
        port_pos_gt_grid_bounds = np.argwhere(port_position > grid_boundaries)

        # no port index can be determined
        if len(port_pos_gt_grid_bounds) == 0:
            raise SetupError(f"Port position '{port_position}' outside of simulation bounds.")
        port_index = port_pos_gt_grid_bounds[-1]

        # shift the port to the left
        if port.direction == "+":
            shifted_index = port_index - 2
            if (
                shifted_index < 0
                or grid_centers[shifted_index] <= self.simulation.bounds[0][normal_axis]
            ):
                raise SetupError(
                    f"Port {port.name} normal is less than 2 cells to the boundary "
                    f"on -{'xyz'[normal_axis]} side. "
                    "Please either increase the mesh resolution near the port or "
                    "move the port away from the boundary."
                )

        # shift the port to the right
        else:
            shifted_index = port_index + 2
            if (
                shifted_index >= len(grid_centers)
                or grid_centers[shifted_index] >= self.simulation.bounds[1][normal_axis]
            ):
                raise SetupError(
                    f"Port {port.name} normal is tless than 2 cells to the boundary "
                    f"on +{'xyz'[normal_axis]} side."
                    "Please either increase the mesh resolution near the port or "
                    "move the port away from the boundary."
                )

        new_pos = grid_centers[shifted_index]
        return new_pos - port_position


AbstractComponentModeler.update_forward_refs()
