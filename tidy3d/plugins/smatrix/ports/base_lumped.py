"""Class and custom data array for representing a scattering matrix port based on lumped circuit elements."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.lumped_element import _IMPEDANCE_ATOL
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.types import Complex
from tidy3d.constants import HERTZ, OHM
from tidy3d.exceptions import ValidationError

from .base_terminal import AbstractTerminalPort

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.grid.grid import Grid, YeeGrid
    from tidy3d.components.lumped_element import LumpedElementType
    from tidy3d.components.microwave.path_integrals.integrals.voltage import (
        AxisAlignedVoltageIntegral,
    )
    from tidy3d.components.monitor import FieldMonitor
    from tidy3d.components.types import Coordinate, FreqArray

DEFAULT_PORT_NUM_CELLS = 3
DEFAULT_REFERENCE_IMPEDANCE = 50


class ImpedanceSpec(MicrowaveBaseModel):
    """Impedance specification for a lumped port.

    Combines a reference impedance with an optional measurement frequency used to infer
    reactive components in the FDTD load model. For a purely real impedance (e.g. 50 Ω),
    only :attr:`impedance` is needed. For a complex impedance ``Z = R + jX``, :attr:`frequency`
    must also be provided so the imaginary part can be mapped to a series inductor or capacitor.

    Example
    -------
    >>> spec_real = ImpedanceSpec(impedance=50)
    >>> spec_inductive = ImpedanceSpec(impedance=50+30j, frequency=1e9)
    >>> spec_capacitive = ImpedanceSpec(impedance=50-20j, frequency=2e9)
    """

    impedance: Complex = Field(
        DEFAULT_REFERENCE_IMPEDANCE,
        title="Impedance",
        description="Reference port impedance ``Z = R + jX`` in ohms. "
        "For a complex value with non-zero imaginary part, :attr:`frequency` must be provided.",
        json_schema_extra={"units": OHM},
    )

    frequency: PositiveFloat | None = Field(
        None,
        title="Measurement Frequency",
        description="Frequency (Hz) at which the complex :attr:`impedance` was measured. "
        "Required when the imaginary part of :attr:`impedance` is non-zero.",
        json_schema_extra={"units": HERTZ},
    )

    @model_validator(mode="after")
    def _validate_spec(self) -> Self:
        Z = complex(self.impedance)
        if abs(Z) < _IMPEDANCE_ATOL:
            self._raise_validation_error_at_loc(
                ValidationError(
                    "'impedance' must be non-zero (Z=0 is a short circuit with infinite admittance)."
                ),
                "impedance",
            )
        if Z.real < 0:
            self._raise_validation_error_at_loc(
                ValidationError(
                    f"'impedance' must have a non-negative real part (passive load). Got Re(Z) = {Z.real}."
                ),
                "impedance",
            )
        if abs(Z.imag) >= _IMPEDANCE_ATOL:
            if Z.real < _IMPEDANCE_ATOL:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "When 'impedance' has a non-zero imaginary part, Re(impedance) must be "
                        "strictly positive to ensure a stable (damped) RLC pole in the FDTD load. "
                        f"Got Re(Z) = {Z.real}."
                    ),
                    "impedance",
                )
            if self.frequency is None:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "'frequency' must be provided when 'impedance' has a non-zero imaginary "
                        "part, so that the reactive component value can be inferred."
                    ),
                    "frequency",
                )
        return self


class AbstractLumpedPort(AbstractTerminalPort):
    """Abstract base class for lumped ports.

    Lumped ports model a physical port as a combination of an excitation source and a
    load element (resistor or RLC network). The reference impedance is set via
    :attr:`impedance`, which accepts either a plain real number (e.g. ``50``) or a full
    :class:`ImpedanceSpec` for complex-valued impedances paired with a measurement frequency.
    """

    impedance: Complex | ImpedanceSpec = Field(
        DEFAULT_REFERENCE_IMPEDANCE,
        title="Impedance",
        description="Reference port impedance in ohms. Accepts a plain real number "
        "(e.g. ``50``) for a purely resistive load, or an :class:`ImpedanceSpec` for a "
        "complex-valued impedance (which also requires a measurement frequency). "
        "Passing a complex scalar with a non-zero imaginary part raises a validation error; "
        "use :class:`ImpedanceSpec` instead.",
    )

    @model_validator(mode="after")
    def _validate_impedance(self) -> Self:
        """Reject complex scalar impedance with non-zero imaginary part."""
        if isinstance(self.impedance, ImpedanceSpec):
            return self
        Z = self._impedance
        if abs(Z.imag) >= _IMPEDANCE_ATOL:
            self._raise_validation_error_at_loc(
                ValidationError(
                    "A scalar 'impedance' must be real-valued. "
                    "For a complex impedance with non-zero imaginary part, use "
                    "'impedance=ImpedanceSpec(impedance=..., frequency=...)' instead."
                ),
                "impedance",
            )
        if abs(Z) < _IMPEDANCE_ATOL:
            self._raise_validation_error_at_loc(
                ValidationError("'impedance' must be non-zero."),
                "impedance",
            )
        if Z.real < 0:
            self._raise_validation_error_at_loc(
                ValidationError(f"'impedance' must have a non-negative real part. Got {Z.real}."),
                "impedance",
            )
        return self

    @cached_property
    def _impedance(self) -> complex:
        """Resolved port impedance as a complex number."""
        if isinstance(self.impedance, ImpedanceSpec):
            return complex(self.impedance.impedance)
        return complex(self.impedance)

    num_grid_cells: PositiveInt | None = Field(
        DEFAULT_PORT_NUM_CELLS,
        title="Port grid cells",
        description="Number of mesh grid cells associated with the port along each direction, "
        "which are added through automatic mesh refinement. "
        "A value of ``None`` will turn off automatic mesh refinement.",
    )

    enable_snapping_points: bool = Field(
        True,
        title="Snap Grid To Lumped Port",
        description="When enabled, snapping points are automatically generated to snap grids to key "
        "geometric features of the lumped port for more accurate modelling.",
    )

    @cached_property
    def _voltage_monitor_name(self) -> str:
        return f"{self.name}_voltage"

    @cached_property
    def _current_monitor_name(self) -> str:
        return f"{self.name}_current"

    def snapped_center(self, grid: Grid) -> Coordinate:
        """Get the exact center of this port after snapping along the injection axis.
        Ports are snapped to the nearest Yee cell boundary to match the exact position
        of the load.
        """
        center = list(self.center)
        normal_axis = self.injection_axis
        normal_port_center = center[normal_axis]
        center[normal_axis] = snap_coordinate_to_grid(grid, normal_port_center, normal_axis)
        return tuple(center)

    @cached_property
    @abstractmethod
    def to_load(self, snap_center: float | None = None) -> LumpedElementType:
        """Create a load from the lumped port."""

    @abstractmethod
    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: float | None = None, grid: Grid | None = None
    ) -> FieldMonitor:
        """Field monitor to compute port voltage."""

    @abstractmethod
    def to_current_monitor(
        self, freqs: FreqArray, snap_center: float | None = None, grid: Grid | None = None
    ) -> FieldMonitor:
        """Field monitor to compute port current."""

    def to_monitors(
        self,
        freqs: FreqArray,
        snap_center: float | None = None,
        grid: Grid | None = None,
        **kwargs: Any,
    ) -> list[FieldMonitor]:
        """Field monitors to compute port voltage and current."""
        return [
            self.to_voltage_monitor(freqs, snap_center, grid),
            self.to_current_monitor(freqs, snap_center, grid),
        ]

    @abstractmethod
    def _make_plot_voltage_integral(self) -> AxisAlignedVoltageIntegral:
        """Create a voltage path integral for plotting (no grid needed)."""

    @abstractmethod
    def _check_grid_size(self, yee_grid: YeeGrid) -> None:
        """Raises :class:`SetupError` if the grid is too coarse at port locations."""

    @property
    def _is_using_mesh_refinement(self) -> bool:
        """Check if this lumped port is using any mesh refinement options.

        Returns ``True`` if snapping points are enabled or custom grid cell count is specified.
        """
        return self.enable_snapping_points or self.num_grid_cells is not None
