"""Specification for modes associated with transmission lines."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd
from shapely import Polygon

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.bound_ops import bounds_contains
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    ImpedanceSpecType,
)
from tidy3d.components.mode_spec import AbstractModeSpec
from tidy3d.components.types import Coordinate2D, annotate_type
from tidy3d.components.types.base import ArrayFloat2D
from tidy3d.constants import fp_eps
from tidy3d.exceptions import SetupError

# Threshold for determining whether a mode is QuasiTEM
DEFAULT_QUASI_TEM_THRESHOLD = 0.97

ConductorIdentifierType = Union[Coordinate2D, str, ArrayFloat2D]


class TerminalSpec(MicrowaveBaseModel):
    """Specifies the desired voltage pattern across conductors for a transmission line mode.

    Identifies which conductors should be at positive versus negative voltage for mode selection
    and ordering in coupled transmission line systems. Conductors can be identified using point
    coordinates, structure names, or geometric regions for maximum flexibility.

    Notes
    -----
    For most use cases, identifying conductors using a single (x, y) coordinate point or a
    structure name string is recommended. If desired, users may specify geometric regions using
    arrays of vertices, which are then converted to points, lines, or polygons to identify
    conductors via intersections.

    Example
    -------
    >>> # Recommended: Identify conductors using point coordinates
    >>> # Differential mode: conductor 1 positive, conductor 2 negative
    >>> diff_mode = TerminalSpec(
    ...     plus_terminals=((1.0, 0.5),),  # Point inside positive conductor
    ...     minus_terminals=((-1.0, 0.5),)  # Point inside negative conductor
    ... )
    >>>
    >>> # Common mode: both conductors positive relative to ground
    >>> common_mode = TerminalSpec(
    ...     plus_terminals=((1.0, 0.5), (-1.0, 0.5)),  # Both conductors positive
    ...     minus_terminals=()  # Ground plane is reference
    ... )
    >>>
    >>> # Alternative: Identify conductors by structure name
    >>> named_mode = TerminalSpec(
    ...     plus_terminals=("trace1",),
    ...     minus_terminals=("trace2",)
    ... )
    >>>
    >>> # Advanced: Identify conductor using polygon region
    >>> import numpy as np
    >>> polygon_mode = TerminalSpec(
    ...     plus_terminals=(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),),  # Square region
    ...     minus_terminals=()
    ... )
    """

    plus_terminals: tuple[ConductorIdentifierType, ...] = pd.Field(
        ...,
        title="Positive Terminals",
        description="Identifies conductors that should be at positive voltage for this mode. "
        "Each conductor can be specified as: (1) a (u, v) coordinate tuple locating a point inside "
        "the conductor (recommended), (2) a string matching a structure name, or (3) an Nx2 array "
        "of vertices defining a geometric region (point for N=1, line for N=2, polygon for N>2). "
        "The coordinate or region should lie within the desired conductor in the mode plane.",
    )
    minus_terminals: tuple[ConductorIdentifierType, ...] = pd.Field(
        ...,
        title="Negative Terminals",
        description="Identifies conductors that should be at negative voltage for this mode. "
        "Each conductor can be specified as: (1) a (u, v) coordinate tuple locating a point inside "
        "the conductor (recommended), (2) a string matching a structure name, or (3) an Nx2 array "
        "of vertices defining a geometric region (point for N=1, line for N=2, polygon for N>2). "
        "The coordinate or region should lie within the desired conductor in the mode plane.",
    )

    @pd.validator("plus_terminals", "minus_terminals", each_item=True)
    def _validate_conductor_identifiers(cls, val):
        """Validate conductor identification inputs."""

        # If it's a string or tuple, pass through
        if isinstance(val, (str, tuple)):
            return val

        # If it's a numpy array, validate shape and geometry
        if isinstance(val, np.ndarray):
            # Check that 2D arrays have exactly 2 columns (u, v coordinates)
            if val.shape[1] != 2:
                raise ValueError(
                    f"Arrays must have exactly 2 columns for (u, v) coordinates, got shape {val.shape}"
                )
            # For 2D arrays, check number of points and polygon validity
            num_points = val.shape[0]
            if num_points <= 2:
                return val
            elif num_points > 2:
                polygon = Polygon(val)
                if not polygon.is_valid:
                    raise ValueError(
                        f"A supplied set of vertices {val} did not result in a valid "
                        "polygon, make sure there are no self-intersections."
                    )
        return val


class MicrowaveModeSpec(AbstractModeSpec, MicrowaveBaseModel):
    """
    The :class:`.MicrowaveModeSpec` class specifies how quantities related to transmission line
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
    """

    impedance_specs: Union[
        annotate_type(ImpedanceSpecType),
        tuple[Optional[annotate_type(ImpedanceSpecType)], ...],
    ] = pd.Field(
        default_factory=AutoImpedanceSpec._default_without_license_warning,
        title="Impedance Specifications",
        description="Field controls how the impedance is calculated for each mode calculated by the mode solver. "
        "Can be a single impedance specification (which will be applied to all modes) or a tuple of specifications "
        "(one per mode). The number of impedance specifications should match the number of modes field. "
        "When an impedance specification of ``None`` is used, the impedance calculation will be "
        "ignored for the associated mode.",
    )

    terminal_specs: Optional[tuple[TerminalSpec, ...]] = pd.Field(
        None,
        title="Terminal Specifications",
        description="Optional tuple of terminal specifications for mode selection and ordering in "
        "transmission line systems. Each 'TerminalSpec' defines the desired voltage pattern (which conductors "
        "should be positive vs. negative) for a mode. When provided, the mode solver automatically reorders "
        "computed modes to match the terminal specification order and applies phase corrections to ensure "
        "correct voltage polarity.",
    )

    quasi_tem_threshold: float = pd.Field(
        DEFAULT_QUASI_TEM_THRESHOLD,
        ge=0.0,
        le=1.0,
        title="Quasi-TEM Mode Threshold",
        description="Threshold used to determine whether a mode is a Quasi-TEM mode. "
        "If both the TE and TM waveguide polarization fractions are less than this threshold, "
        "the mode is considered as a Quasi-TEM mode.",
    )

    @cached_property
    def _impedance_specs_as_tuple(self) -> tuple[Optional[ImpedanceSpecType]]:
        """Gets the impedance_specs field converted to a tuple."""
        if isinstance(self.impedance_specs, Union[tuple, list]):
            return tuple(self.impedance_specs)
        return (self.impedance_specs,)

    @cached_property
    def _using_auto_current_spec(self) -> bool:
        """Checks whether at least one of the modes will require an auto setup of the current path specification."""
        return any(
            isinstance(impedance_spec, AutoImpedanceSpec)
            for impedance_spec in self._impedance_specs_as_tuple
        )

    @pd.validator("impedance_specs", always=True)
    def check_impedance_specs_consistent_with_num_modes(cls, val, values):
        """Check that the number of impedance specifications is equal to the number of modes.
        A single impedance spec is also permitted."""
        num_modes = values.get("num_modes")
        if isinstance(val, Union[tuple, list]):
            num_impedance_specs = len(val)
        else:
            return val

        # Otherwise, check that the count matches
        if num_impedance_specs != num_modes:
            raise SetupError(
                f"Given {num_impedance_specs} impedance specifications in the 'MicrowaveModeSpec', "
                f"but the number of modes requested is {num_modes}. Please ensure that the "
                "number of impedance specifications is equal to the number of modes, or provide "
                "a single specification to apply to all modes."
            )

        return val

    def _check_path_integrals_within_box(self, box: Box):
        """Raise SetupError if a ``CustomImpedanceSpec`` includes a path specification
        defined outside a candidate box.
        """
        for impedance_ind, impedance_spec in enumerate(self._impedance_specs_as_tuple):
            if isinstance(impedance_spec, AutoImpedanceSpec) or impedance_spec is None:
                continue

            # Check both voltage and current specs using the same logic
            specs_to_check = [
                (impedance_spec.voltage_spec, "voltage"),
                (impedance_spec.current_spec, "current"),
            ]

            for spec, spec_type in specs_to_check:
                if spec is None:
                    continue

                box_bounds = box.bounds
                # If the box is a plane (one dimension is zero), we need to ignore
                # the bounds check along the normal axis
                if box.size.count(0.0) == 1:
                    normal_axis = box._normal_axis
                    # Convert tuple to list so we can modify it
                    box_bounds = [list(box_bounds[0]), list(box_bounds[1])]
                    # Set the bounds along normal axis to match the spec bounds
                    box_bounds[0][normal_axis] = spec.bounds[0][normal_axis]
                    box_bounds[1][normal_axis] = spec.bounds[1][normal_axis]
                    # Convert back to tuple for bounds_contains
                    box_bounds = (tuple(box_bounds[0]), tuple(box_bounds[1]))

                if not bounds_contains(
                    box_bounds, spec.bounds, fp_eps, np.finfo(np.float32).smallest_normal
                ):
                    raise SetupError(
                        "A 'MicrowaveModeSpec' must be setup with all path specifications defined within "
                        f"the bounds of the mode solving plane. The 'CustomImpedanceSpec' at index "
                        f"'{impedance_ind}' was provided with a {spec_type} path specification with bounds "
                        f"'{spec.bounds}', but the mode plane bounds are '{box.bounds}'."
                    )
