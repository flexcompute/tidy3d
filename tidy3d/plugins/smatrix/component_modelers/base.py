"""Base class for generating an S matrix automatically from tidy3d simulations and port definitions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.data.data_array import DataArray
from ....components.simulation import Simulation
from ....components.types import Complex, FreqArray
from ....constants import HERTZ
from ....exceptions import SetupError, Tidy3dKeyError
from ....log import log
from ....web.api.container import Batch, BatchData
from ..ports.modal import Port
from ..ports.rectangular_lumped import LumpedPort

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "."


class AbstractComponentModeler(ABC, Tidy3dBaseModel):
    """Tool for modeling devices and computing port parameters."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources present.",
    )

    ports: Tuple[Union[Port, LumpedPort], ...] = pd.Field(
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

    @staticmethod
    def _task_name(port: Port, mode_index: int = None) -> str:
        """The name of a task, determined by the port of the source and mode index, if given."""
        if mode_index is not None:
            return f"smatrix_{port.name}_{mode_index}"
        return f"smatrix_{port.name}"

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the S matrix calculation."""

    def to_file(self, fname: str) -> None:
        """Exports :class:`Tidy3dBaseModel` instance to .yaml, .json, or .hdf5 file

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to save the :class:`Tidy3dBaseModel` to.

        Example
        -------
        >>> simulation.to_file(fname='folder/sim.json') # doctest: +SKIP
        """

        batch_cached = self._cached_properties.get("batch")
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
        """Batch associated with this component modeler."""

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

        return self.batch._batch_path(path_dir=DEFAULT_DATA_DIR)

    def get_path_dir(self, path_dir: str) -> None:
        """Check whether the supplied 'path_dir' matches the internal field value."""

        if path_dir not in (DEFAULT_DATA_DIR, self.path_dir):
            log.warning(
                f"'ComponentModeler' method was supplied a 'path_dir' of '{path_dir}' "
                f"when its internal 'path_dir' field was set to '{self.path_dir}'. "
                "The passed value will be deprecated in later versions. "
                "Please set the internal 'path_dir' field to the desired value and "
                "remove the 'path_dir' from the method argument. "
                f"Using supplied '{path_dir}'."
            )
            return path_dir

        return self.path_dir

    @cached_property
    def _batch_path(self) -> str:
        """Where we store the batch for this ComponentModeler instance after the run."""
        return os.path.join(self.path_dir, "batch" + str(hash(self)) + ".hdf5")

    def _run_sims(self, path_dir: str = DEFAULT_DATA_DIR) -> BatchData:
        """Run :class:`Simulations` for each port and return the batch after saving."""
        batch = self.batch

        batch_data = batch.run(path_dir=path_dir)
        batch.to_file(self._batch_path)
        return batch_data

    def get_port_by_name(self, port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in self.ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @abstractmethod
    def _construct_smatrix(self, batch_data: BatchData) -> DataArray:
        """Post process :class:`BatchData` to generate scattering matrix."""

    def run(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Solves for the scattering matrix of the system."""
        path_dir = self.get_path_dir(path_dir)
        batch_data = self._run_sims(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    def load(self, path_dir: str = DEFAULT_DATA_DIR) -> DataArray:
        """Load a scattering matrix from saved :class:`BatchData` object."""
        path_dir = self.get_path_dir(path_dir)

        batch_data = BatchData.load(path_dir=path_dir)
        return self._construct_smatrix(batch_data=batch_data)

    @staticmethod
    def s_to_z(s_matrix: DataArray, reference: Complex) -> DataArray:
        """Get the impedance matrix given the scattering matrix and a reference impedance."""

        # move the input and output port dimensions to the end, for ease of matrix operations
        dims = list(s_matrix.dims)
        dims.append(dims.pop(dims.index("port_out")))
        dims.append(dims.pop(dims.index("port_in")))
        z_matrix = s_matrix.copy(deep=True).transpose(*dims)
        s_vals = z_matrix.values

        eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))
        z_vals = np.matmul(AbstractComponentModeler.inv(eye - s_vals), (eye + s_vals)) * reference

        z_matrix.data = z_vals
        return z_matrix

    @staticmethod
    def ab_to_s(a_matrix: DataArray, b_matrix: DataArray) -> DataArray:
        """Get the scattering matrix given the power wave matrices."""

        # move the input and output port dimensions to the end, for ease of matrix operations
        if a_matrix.dims != b_matrix.dims:
            raise ValueError(
                "'a_matrix' and 'b_matrix' must have the same number of dimensions, "
                f"got {a_matrix.dims=} and {b_matrix.dims=}"
            )
        dims = list(a_matrix.dims)
        dims.append(dims.pop(dims.index("port_out")))
        dims.append(dims.pop(dims.index("port_in")))

        s_matrix = a_matrix.copy(deep=True).transpose(*dims)
        a_vals = s_matrix.copy(deep=True).transpose(*dims).values
        b_vals = b_matrix.copy(deep=True).transpose(*dims).values

        s_vals = np.matmul(b_vals, AbstractComponentModeler.inv(a_vals))

        s_matrix.data = s_vals
        return s_matrix

    @staticmethod
    def inv(matrix: DataArray):
        """Helper to invert a port matrix."""
        return np.linalg.inv(matrix)
