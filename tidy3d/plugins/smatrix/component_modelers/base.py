"""Base class for generating an S matrix automatically from tidy3d simulations and port definitions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, get_args

import numpy as np
import pydantic.v1 as pd

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.data.data_array import DataArray
from ....components.data.sim_data import SimulationData
from ....components.simulation import Simulation
from ....components.types import FreqArray
from ....config import config
from ....constants import HERTZ
from ....exceptions import SetupError, Tidy3dKeyError
from ....log import log
from ....web.api.container import Batch, BatchData
from ..ports.coaxial_lumped import CoaxialLumpedPort
from ..ports.modal import Port
from ..ports.rectangular_lumped import LumpedPort
from ..ports.wave import WavePort

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "."

LumpedPortType = Union[LumpedPort, CoaxialLumpedPort]
TerminalPortType = Union[LumpedPortType, WavePort]


class AbstractComponentModeler(ABC, Tidy3dBaseModel):
    """Tool for modeling devices and computing port parameters."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )

    ports: Tuple[Union[Port, TerminalPortType], ...] = pd.Field(
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

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Name of the folder for the tasks on web.",
    )

    verbose: bool = pd.Field(
        False,
        title="Verbosity",
        description="Whether the :class:`.AbstractComponentModeler` should print status and progressbars.",
    )

    callback_url: str = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    path_dir: str = pd.Field(
        DEFAULT_DATA_DIR,
        title="Directory Path",
        description="Base directory where data and batch will be downloaded.",
    )

    solver_version: str = pd.Field(
        None,
        title="Solver Version",
        description_str="Custom solver version to use. "
        "If not supplied, uses default for the current front end version.",
    )

    batch_cached: Batch = pd.Field(
        None,
        title="Batch (Cached)",
        description="Optional field to specify ``batch``. Only used as a workaround internally "
        "so that ``batch`` is written when ``.to_file()`` and then the proper batch is loaded "
        "from ``.from_file()``. We recommend leaving unset as setting this field along with "
        "fields that were not used to create the task will cause errors.",
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
    def _task_name(port: Port, mode_index: int = None) -> str:
        """The name of a task, determined by the port of the source and mode index, if given."""
        if mode_index is not None:
            return f"smatrix_{port.name}_{mode_index}"
        return f"smatrix_{port.name}"

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`.Simulation` objects for the S matrix calculation."""

    def to_file(self, fname: str) -> None:
        """Exports :class:`AbstractComponentModeler` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`AbstractComponentModeler` to.

        Example
        -------
        >>> modeler.to_file(fname='folder/sim.json') # doctest: +SKIP
        """

        batch_cached = self._cached_properties.get("batch")
        if batch_cached is not None:
            jobs_cached = batch_cached._cached_properties.get("jobs")
            if jobs_cached is not None:
                jobs = {}
                for key, job in jobs_cached.items():
                    task_id = job._cached_properties.get("task_id")
                    jobs[key] = job.updated_copy(task_id_cached=task_id)
                batch_cached = batch_cached.updated_copy(jobs_cached=jobs)
            self = self.updated_copy(batch_cached=batch_cached)
        super(AbstractComponentModeler, self).to_file(fname=fname)  # noqa: UP008

    @cached_property
    def batch(self) -> Batch:
        """:class:`.Batch` associated with this component modeler."""

        if self.batch_cached is not None:
            return self.batch_cached

        # first try loading the batch from file, if it exists
        batch_path = self._batch_path

        if os.path.exists(batch_path):
            return Batch.from_file(fname=batch_path)

        return Batch(
            simulations=self.sim_dict,
            folder_name=self.folder_name,
            callback_url=self.callback_url,
            verbose=self.verbose,
            solver_version=self.solver_version,
        )

    @cached_property
    def batch_path(self) -> str:
        """Path to the batch saved to file."""
        return self.batch._batch_path(path_dir=self.path_dir)

    @cached_property
    def batch_data(self) -> BatchData:
        """The :class:`.BatchData` associated with the simulations run for this component modeler."""
        return self.batch.run(path_dir=self.path_dir)

    def get_path_dir(self, path_dir: str) -> None:
        """Check whether the supplied 'path_dir' matches the internal field value."""

        if path_dir != self.path_dir and path_dir != DEFAULT_DATA_DIR:
            raise ValueError(
                f"'path_dir' of '{path_dir}' passed, but 'ComponentModeler.path_dir' is "
                f"{self.path_dir}. Moving forward, only the 'ComponentModeler.path_dir' will be "
                "used internally, please update your scripts accordingly to avoid passing this "
                "value to methods. "
            )

        return self.path_dir

    @cached_property
    def _batch_path(self) -> str:
        """Where we store the batch for this :class:`AbstractComponentModeler` instance after the run."""
        return os.path.join(self.path_dir, "batch" + str(hash(self)) + ".hdf5")

    def _run_sims(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Run :class:`.Simulation` for each port and return the batch after saving."""
        _ = self.get_path_dir(path_dir)
        self.batch.to_file(self._batch_path)
        batch_data = self.batch_data
        return batch_data

    def get_port_by_name(self, port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in self.ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @abstractmethod
    def _construct_smatrix(self, batch_data: BatchData) -> DataArray:
        """Post process :class:`.BatchData` to generate scattering matrix."""

    @abstractmethod
    def _internal_construct_smatrix(self, batch_data: BatchData) -> DataArray:
        """Post process :class:`.BatchData` to generate scattering matrix, for internal use only."""

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Solves for the scattering matrix of the system."""
        _ = self.get_path_dir(path_dir)
        return self._construct_smatrix()

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Load a scattering matrix from saved :class:`.BatchData` object."""
        return self.run(path_dir=path_dir)

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

    def sim_data_by_task_name(self, task_name: str) -> SimulationData:
        """Get the simulation data by task name, avoids emitting warnings from the ``Simulation``."""
        log_level_cache = config.logging_level
        config.logging_level = "ERROR"
        sim_data = self.batch_data[task_name]
        config.logging_level = log_level_cache
        return sim_data
