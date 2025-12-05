"""Base class for generating an S matrix automatically from tidy3d simulations and port definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from tidy3d.components.geometry.utils import _shift_value_signed
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.time import SourceTimeType
from tidy3d.components.types import Complex, FreqArray
from tidy3d.components.validators import (
    assert_unique_names,
    validate_freqs_min,
    validate_freqs_not_empty,
    validate_freqs_unique,
)
from tidy3d.config import config
from tidy3d.constants import HERTZ
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from tidy3d.log import log
from tidy3d.plugins.smatrix.ports.modal import Port
from tidy3d.plugins.smatrix.ports.types import LumpedPortType, PortType, TerminalPortType
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.types import Element, MatrixIndex, NetworkElement, NetworkIndex

if TYPE_CHECKING:
    from tidy3d.web.core.types import PayType

# fwidth of gaussian pulse in units of central frequency
FWIDTH_FRAC = 1.0 / 10
DEFAULT_DATA_DIR = "."

IndexType = Union[MatrixIndex, NetworkIndex]
ElementType = Union[Element, NetworkElement]
TaskNameFormat = Literal["RF", "PF"]


class AbstractComponentModeler(ABC, Tidy3dBaseModel):
    """Tool for modeling devices and computing port parameters."""

    name: str = pd.Field(
        "",
        title="Name",
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

    run_only: Optional[tuple[IndexType, ...]] = pd.Field(
        None,
        title="Run Only",
        description="Set of matrix indices that define the simulations to run. "
        "If ``None``, simulations will be run for all indices in the scattering matrix. "
        "If a tuple is given, simulations will be run only for the given matrix indices.",
    )

    element_mappings: tuple[tuple[ElementType, ElementType, Complex], ...] = pd.Field(
        (),
        title="Element Mappings",
        description="Tuple of S matrix element mappings, each described by a tuple of "
        "(input_element, output_element, coefficient), where the coefficient is the "
        "element_mapping coefficient describing the relationship between the input and output "
        "matrix element. If all elements of a given column of the scattering matrix are defined "
        "by ``element_mappings``, the simulation corresponding to this column is skipped automatically.",
    )
    custom_source_time: Optional[SourceTimeType] = pd.Field(
        None,
        title="Custom Source Time",
        description="If provided, this will be used as specification of the source time-dependence in simulations. "
        "Otherwise, a default source time will be constructed.",
    )

    @pd.root_validator(pre=False)
    def _warn_refactor_2_10(cls, values):
        log.warning(
            f"'{cls.__name__}' was refactored (tidy3d 'v2.10.0'). Existing functionality is available differently. Please consult the migration documentation: https://docs.flexcompute.com/projects/tidy3d/en/latest/api/microwave/microwave_migration.html",
            log_once=True,
        )
        return values

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError(f"'{cls.__name__}.simulation' must not have any sources.")
        return val

    @pd.validator("element_mappings", always=True)
    def _validate_element_mappings(cls, element_mappings, values):
        """
        Validate that each source index referenced in element_mappings is included in run_only.
        """
        run_only = values.get("run_only")
        if run_only is None:
            return element_mappings

        valid_set = set(run_only)
        invalid_indices = set()
        for mapping in element_mappings:
            input_element = mapping[0]
            output_element = mapping[1]
            for source_index in [input_element[1], output_element[1]]:
                if source_index not in valid_set:
                    invalid_indices.add(source_index)
        if invalid_indices:
            raise SetupError(
                f"'element_mappings' references source index(es) {invalid_indices} "
                f"that are not present in run_only: {run_only}."
            )
        return element_mappings

    @pd.validator("run_only", always=True)
    @skip_if_fields_missing(["ports"])
    def _validate_run_only(cls, val, values):
        """Validate that run_only entries are unique and exist in matrix_indices_monitor."""
        if val is None:
            return val

        # Check uniqueness
        if len(val) != len(set(val)):
            duplicates = [idx for idx in set(val) if val.count(idx) > 1]
            raise SetupError(
                f"'run_only' contains duplicate entries: {duplicates}. "
                "Each index must appear only once."
            )

        # Check membership - use the helper method to get valid indices
        ports = values["ports"]

        valid_indices = set(cls._construct_matrix_indices_monitor(ports))
        invalid_indices = [idx for idx in val if idx not in valid_indices]

        if invalid_indices:
            raise SetupError(
                f"'run_only' contains indices {invalid_indices} that are not present in "
                f"'matrix_indices_monitor'. Valid indices are: {sorted(valid_indices)}"
            )

        return val

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()
    _freqs_unique = validate_freqs_unique()

    @pd.validator("custom_source_time", always=True)
    @skip_if_fields_missing(["freqs"])
    def _freqs_in_custom_source_time(cls, val, values):
        """Make sure freqs is in the range of the custom source time."""
        if val is None:
            return val
        freq_range = val._frequency_range_sigma_cached
        freqs = values["freqs"]

        if freq_range[0] > min(freqs) or max(freqs) > freq_range[1]:
            log.warning(
                "Custom source time does not cover all 'freqs'.",
            )
        return val

    @staticmethod
    def get_task_name(port: PortType, mode_index: Optional[int] = None) -> str:
        """Generates a standardized task name from a port object.

        This method creates a unique string identifier for a simulation task based on
        a port and, if applicable, a specified mode index.

        Parameters
        ----------
        port : PortType
            The port object from which to derive the base name.
        mode_index : Optional[int], optional
            If provided, this index is appended
            to the port name (e.g., 'port_1@1'). Defaults to `None`, in which case the first
            mode is chosen by default.

        Returns
        -------
        str
            The formatted task name string.

        Raises
        ------
        ValueError
            If `mode_index` is specified for a lumped port.
        """

        if isinstance(port, LumpedPortType):
            if mode_index is not None:
                raise ValueError(
                    "'mode_index' should not be specified for a lumped port, "
                    f"but was passed with value '{mode_index}'."
                )
            return f"{port.name}"
        elif isinstance(port, WavePort):
            # WavePorts default to first mode index
            if mode_index is not None:
                return f"{port.name}@{mode_index}"
            return f"{port.name}@{port._mode_indices[0]}"
        else:
            # Modal ports default to 0
            if mode_index is not None:
                return f"{port.name}@{mode_index}"
            return f"{port.name}@0"

    def get_port_by_name(self, port_name: str) -> Port:
        """Get the port from the name."""
        ports = [port for port in self.ports if port.name == port_name]
        if len(ports) == 0:
            raise Tidy3dKeyError(f'Port "{port_name}" not found.')
        return ports[0]

    @staticmethod
    @abstractmethod
    def _construct_matrix_indices_monitor(ports: tuple) -> tuple[IndexType, ...]:
        """Construct matrix indices for monitoring from ports.

        This helper method is used by both the matrix_indices_monitor property
        and the run_only validator to ensure consistency.

        Parameters
        ----------
        ports : tuple
            Tuple of port objects.

        Returns
        -------
        tuple[IndexType, ...]
            Tuple of matrix indices for monitoring.
        """

    @property
    @abstractmethod
    def matrix_indices_monitor(self) -> tuple[IndexType, ...]:
        """Abstract property for all matrix indices that will be used to collect data."""

    @cached_property
    def matrix_indices_source(self) -> tuple[IndexType, ...]:
        """Tuple of all the source matrix indices, which may be less than the total number of
        ports."""
        if self.run_only is not None:
            return self.run_only
        return self.matrix_indices_monitor

    @cached_property
    def matrix_indices_run_sim(self) -> tuple[IndexType, ...]:
        """Tuple of all the matrix indices that will be used to run simulations."""

        if not self.element_mappings:
            return self.matrix_indices_source

        # all the (i, j) pairs in `S_ij` that are tagged as covered by `element_mappings`
        elements_determined_by_map = [element_out for (_, element_out, _) in self.element_mappings]

        # loop through rows of the full s matrix and record rows that still need running.
        source_indices_needed = []
        for col_index in self.matrix_indices_source:
            # loop through columns and keep track of whether each element is covered by mapping.
            matrix_elements_covered = []
            for row_index in self.matrix_indices_monitor:
                element = (row_index, col_index)
                element_covered_by_map = element in elements_determined_by_map
                matrix_elements_covered.append(element_covered_by_map)

            # if any matrix elements in row still not covered by map, a source is needed for row.
            if not all(matrix_elements_covered):
                source_indices_needed.append(col_index)

        return source_indices_needed

    def _shift_value_signed(self, port: Union[Port, WavePort]) -> float:
        """How far (signed) to shift the source from the monitor."""

        return _shift_value_signed(
            obj=port,
            grid=self.simulation.grid,
            bounds=self.simulation.bounds,
            direction=port.direction,
            shift=-2,
            name=f"Port {port.name}",
        )

    unique_port_names = assert_unique_names("ports")

    def run(
        self,
        path_dir: str = DEFAULT_DATA_DIR,
        *,
        folder_name: str = "default",
        callback_url: Optional[str] = None,
        verbose: bool = True,
        solver_version: Optional[str] = None,
        pay_type: Union[PayType, str] = "AUTO",
        priority: Optional[int] = None,
        local_gradient: bool = False,
        max_num_adjoint_per_fwd: Optional[int] = None,
    ):
        log.warning(
            "'ComponentModeler.run()' is deprecated and will be removed in a future release. "
            "Use web.run(modeler) instead. 'web.run' returns a 'ComponentModelerData' object; "
            "get the scattering matrix via 'data.smatrix()'.",
            log_once=True,
        )
        from tidy3d.plugins.smatrix.run import _run_local

        if max_num_adjoint_per_fwd is None:
            max_num_adjoint_per_fwd = config.adjoint.max_adjoint_per_fwd

        data = _run_local(
            self,
            path_dir=path_dir,
            folder_name=folder_name,
            callback_url=callback_url,
            verbose=verbose,
            solver_version=solver_version,
            pay_type=pay_type,
            priority=priority,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
        )
        return data.smatrix()

    def validate_pre_upload(self):
        """Validate the modeler before upload."""
        self.base_sim.validate_pre_upload(source_required=False)


AbstractComponentModeler.update_forward_refs()
