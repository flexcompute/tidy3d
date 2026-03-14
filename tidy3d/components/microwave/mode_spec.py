"""Specification for modes associated with transmission lines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import Field, PositiveInt, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import ModePlaneAnalyzer
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    CustomImpedanceSpec,
    ImpedanceSpecType,
)
from tidy3d.components.mode_spec import AbstractModeSpec
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.microwave.types import ImpedanceDef
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Coordinate, Size, Symmetry

TEM_POLARIZATION_THRESHOLD = 0.995
QTEM_POLARIZATION_THRESHOLD = 0.95
MONITOR_COLOCATE = False


class MicrowaveModeSpec(AbstractModeSpec, MicrowaveBaseModel):
    """Specification for transmission line modes and microwave waveguides.

    Notes
    -----
        The :class:`~tidy3d.rf.MicrowaveModeSpec` class specifies how quantities related to transmission line
        modes and microwave waveguides are computed. For example, it defines the paths for line integrals, which are used to
        compute voltage, current, and characteristic impedance of the transmission line.

    Example
    -------
    >>> import tidy3d as td
    >>> # Using automatic impedance calculation (single spec, will be duplicated for all modes)
    >>> mode_spec_auto = td.MicrowaveModeSpec(
    ...     num_modes=2,
    ...     impedance_specs=td.AutoImpedanceSpec()
    ... )
    >>> # Using custom impedance specification for multiple modes
    >>> voltage_spec = td.AxisAlignedVoltageIntegralSpec(
    ...     center=(0, 0, 0), size=(0, 0, 1), sign="+"
    ... )
    >>> current_spec = td.AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(2, 1, 0), sign="+"
    ... )
    >>> custom_impedance = td.CustomImpedanceSpec(
    ...     voltage_spec=voltage_spec, current_spec=current_spec
    ... )
    >>> mode_spec_custom = td.MicrowaveModeSpec(
    ...     num_modes=1,
    ...     impedance_specs=custom_impedance
    ... )
    >>> # Using num_modes='auto' with a tuple of specs
    >>> mode_spec_auto_tuple = td.MicrowaveModeSpec(
    ...     num_modes="auto",
    ...     impedance_specs=(custom_impedance, custom_impedance)
    ... )
    """

    num_modes: Union[PositiveInt, Literal["auto"]] = Field(
        1,
        title="Number of modes",
        description="Number of modes returned by mode solver. "
        "Use 'auto' to infer from impedance_specs length (if tuple) or detected "
        "from the number of isolated conductors "
        "assuming quasi-TEM modes (if AutoImpedanceSpec).",
    )

    impedance_specs: Union[
        ImpedanceSpecType,
        tuple[Optional[ImpedanceSpecType], ...],
    ] = Field(
        default_factory=AutoImpedanceSpec._default_without_license_warning,
        title="Impedance Specifications",
        description="Field controls how the impedance is calculated for each mode calculated by the mode solver. "
        "Can be a single impedance specification (which will be applied to all modes) or a tuple of specifications "
        "(one per mode). The number of impedance specifications should match the number of modes field. "
        "When an impedance specification of ``None`` is used, the impedance calculation will be "
        "ignored for the associated mode.",
    )

    tem_polarization_threshold: float = Field(
        TEM_POLARIZATION_THRESHOLD,
        gt=0.0,
        le=1.0,
        title="TEM Polarization Threshold",
        description="Threshold for classifying modes as TEM, TE, or TM based on mean TE/TM fraction "
        "across frequencies. A mode is classified as TEM if both mean TE and TM fractions are greater "
        "than or equal to this threshold. Similarly, a mode is classified as TE (or TM) if the mean TE "
        "(or TM) fraction is greater than or equal to this threshold.",
    )

    qtem_polarization_threshold: float = Field(
        QTEM_POLARIZATION_THRESHOLD,
        gt=0.0,
        le=1.0,
        title="Quasi-TEM Polarization Threshold",
        description="Threshold for classifying modes as quasi-TEM based on TE/TM fraction at the lowest "
        "frequency. A mode is classified as quasi-TEM if both TE and TM fractions at the lowest frequency "
        "are greater than or equal to this threshold.",
    )

    @cached_property
    def _impedance_specs_as_tuple(self) -> tuple[Optional[ImpedanceSpecType], ...]:
        """Gets the impedance_specs field converted to a tuple."""
        if isinstance(self.impedance_specs, (tuple, list)):
            return tuple(self.impedance_specs)
        return (self.impedance_specs,)

    @cached_property
    def _using_auto_current_spec(self) -> bool:
        """Checks whether at least one of the modes will require an auto setup of the current path specification."""
        return any(
            isinstance(impedance_spec, AutoImpedanceSpec)
            for impedance_spec in self._impedance_specs_as_tuple
        )

    @model_validator(mode="after")
    def check_impedance_specs_consistent_with_num_modes(self) -> Self:
        """Check impedance specs consistency with num_modes.

        Validates that:
        1. num_modes='auto' with tuple
        2. num_modes='auto' with single spec: must be AutoImpedanceSpec
        3. num_modes=int with tuple: length must match
        4. num_modes=int with single spec: allowed (applied to all modes)
        """
        val = self.impedance_specs
        num_modes = self.num_modes

        if num_modes == "auto":
            if not isinstance(val, (tuple, list)):
                if not isinstance(val, AutoImpedanceSpec):
                    raise SetupError(
                        "num_modes='auto' with a single non-AutoImpedanceSpec cannot determine "
                        "the number of modes. Provide a tuple of specs, "
                        "or use AutoImpedanceSpec for automatic conductor detection."
                    )
            return self

        # For explicit num_modes, check tuple length matches
        if isinstance(val, (tuple, list)):
            if len(val) != num_modes:
                raise SetupError(
                    f"Given {len(val)} impedance specifications in the 'MicrowaveModeSpec', "
                    f"but the number of modes requested is {num_modes}. Please ensure that the "
                    "number of impedance specifications is equal to the number of modes, or provide "
                    "a single specification to apply to all modes."
                )

        return self

    def _check_path_integrals_within_box(self, box: Box) -> None:
        """Raise SetupError if a ``CustomImpedanceSpec`` includes a path specification
        defined outside a candidate box.
        """
        for impedance_ind, impedance_spec in enumerate(self._impedance_specs_as_tuple):
            if impedance_spec is None:
                continue

            # Use the impedance spec's own validation method and add context if it fails
            try:
                impedance_spec._check_path_integrals_within_box(box)
            except SetupError as e:
                raise SetupError(
                    f"A 'MicrowaveModeSpec' must be setup with all path specifications defined within "
                    f"the bounds of the mode solving plane. The impedance specification at index "
                    f"'{impedance_ind}' failed validation: {e}"
                ) from e

    def _validate_auto_impedance_setup(
        self,
        center: Coordinate,
        size: Size,
        colocate: bool,
        volumetric_structures: list[Structure],
        grid: Grid,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        simulation_geometry: Box,
        label: str = "",
        interior_disjoint_geometries: bool = True,
    ) -> None:
        """Validate that auto impedance specification can be set up for the given mode plane.

        Parameters
        ----------
        center : Coordinate
            Center of the mode plane.
        size : Size
            Size of the mode plane.
        colocate : bool
            Whether field data is colocated.
        volumetric_structures : list[Structure]
            List of volumetric structures in the simulation.
        grid : Grid
            Simulation grid for snapping paths.
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry conditions for the simulation in (x, y, z) directions.
        simulation_geometry : Box
            Simulation domain box used for boundary conditions.
        label : str = ""
            Optional label for error messages.
        interior_disjoint_geometries : bool = True
            If ``True``, conductors on the plane will not be overridden by other materials,
            allowing a faster merging path that skips overlap removal.
        """
        if not self._using_auto_current_spec:
            return
        mode_plane_analyzer = ModePlaneAnalyzer(
            center=center,
            size=size,
            field_data_colocated=colocate,
        )
        try:
            mode_plane_analyzer.get_conductor_bounding_boxes(
                volumetric_structures,
                grid,
                symmetry,
                simulation_geometry,
                interior_disjoint_geometries=interior_disjoint_geometries,
            )
        except SetupError as e:
            raise SetupError(f"Failed to setup auto impedance specification{label}. {e!s}") from e


class MicrowaveTerminalModeSpec(MicrowaveModeSpec):
    """The :class:`.MicrowaveTerminalModeSpec` class specifies how quantities related to transmission line
    terminals are computed. All specs must be specified explicitly here: num_modes must be an integer; and impedance_specs
    must not contain AutoImpedanceSpec, and must be labeled by terminal names. Terminal mapping needs to be
    provided in the presence of differential pairs.

    Note
    -----
    This class is for internal usage only.

    Example
    -------
    >>> import tidy3d as td
    >>> from tidy3d.components.microwave.mode_spec import MicrowaveTerminalModeSpec
    >>> # Using custom impedance specification for multiple terminals
    >>> current_spec = td.AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(2, 1, 0), sign="+"
    ... )
    >>> custom_impedance = td.CustomImpedanceSpec(
    ...     current_spec=current_spec
    ... )
    >>> terminal_mode_spec_custom = MicrowaveTerminalModeSpec(
    ...     num_modes=1,
    ...     impedance_specs={"T0": custom_impedance},
    ... )
    """

    num_modes: PositiveInt = Field(
        ...,
        title="Number of modes",
        description="Number of modes (terminals) returned by mode solver.",
    )

    impedance_specs: dict[str, CustomImpedanceSpec] = Field(
        ...,
        title="Impedance Specification For Each Singled-ended Terminal",
        description="Field controls how the voltage, current, and impedance are calculated for each "
        "singled-ended terminal. "
        "The size of the dictionary should match the number of modes field. ",
    )

    terminals_mapping: Optional[dict[str, Union[str, tuple[str, str]]]] = Field(
        None,
        title="Terminals Mapping",
        description="Mapping from terminal (including differential pairs) labels to single-ended terminal labels.",
    )

    @model_validator(mode="after")
    def _validate_terminal_mode_spec(self) -> Self:
        """Validate impedance definitions consistency, num_modes match, and terminals mapping."""
        # Validate consistent impedance definitions
        val = self.impedance_specs
        if len(val) > 0:
            specs = list(val.values())
            first_definition = specs[0].impedance_definition
            for impedance_spec in specs[1:]:
                if impedance_spec.impedance_definition != first_definition:
                    raise SetupError("Inconsistent impedance definitions across terminals.")

        # Check impedance specs consistent with num_modes
        if len(val) != self.num_modes:
            raise SetupError(
                f"Given {len(val)} impedance specifications in the 'MicrowaveTerminalModeSpec', "
                f"but the number of modes requested is {self.num_modes}. Please ensure that the "
                "number of impedance specifications is equal to the number of modes."
            )

        # Check terminals mapping consistency with impedance specs
        terminals_mapping = self.terminals_mapping
        if terminals_mapping is not None:
            if len(terminals_mapping) != len(val):
                raise SetupError(
                    f"Given {len(terminals_mapping)} terminals mapping in the 'MicrowaveTerminalModeSpec', "
                    f"but the number of impedance specifications is {len(val)}. Please ensure that the "
                    "number of terminals mapping is equal to the number of impedance specifications."
                )

            for terminal_label in terminals_mapping.values():
                # Handle both single terminal (str) and differential pair (tuple[str, str])
                labels_to_check = (
                    (terminal_label,) if isinstance(terminal_label, str) else terminal_label
                )
                for label in labels_to_check:
                    if label not in val.keys():
                        raise SetupError(
                            f"Terminal label '{label}' is not present in the impedance specifications."
                        )

        return self

    @cached_property
    def impedance_definition(self) -> ImpedanceDef:
        """Impedance definition (consistent across all terminals)."""
        return next(iter(self.impedance_specs.values())).impedance_definition

    @cached_property
    def _terminal_indices(self) -> list[str]:
        """List of terminal indices."""
        if self.terminals_mapping is None:
            return list(self.impedance_specs.keys())
        return list(self.terminals_mapping.keys())

    @cached_property
    def _impedance_specs_as_tuple(self) -> tuple[Optional[ImpedanceSpecType], ...]:
        """Gets the impedance_specs field converted to a tuple."""
        return tuple(self.impedance_specs.values())


MicrowaveModeSpecType = Union[MicrowaveModeSpec, MicrowaveTerminalModeSpec]
