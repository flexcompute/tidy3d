"""Microwave sources for RF and transmission line simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tidy3d.components.microwave.mode_spec import MicrowaveTerminalModeSpec
from tidy3d.components.source.field import AbstractModeSource

if TYPE_CHECKING:
    from tidy3d.compat import Self


class MicrowaveTerminalSource(AbstractModeSource):
    """Injects current source to excite a specific terminal mode.

    Example
    -------
    >>> from tidy3d import GaussianPulse, AxisAlignedCurrentIntegralSpec, CustomImpedanceSpec
    >>> pulse = GaussianPulse(freq0=10e9, fwidth=1e9)
    >>>
    >>> # Define impedance spec for terminal
    >>> current_spec = AxisAlignedCurrentIntegralSpec(
    ...     center=(0, 0, 0), size=(2, 1, 0), sign="+"
    ... )
    >>> impedance_spec = CustomImpedanceSpec(current_spec=current_spec)
    >>>
    >>> # Create terminal mode spec
    >>> terminal_spec = MicrowaveTerminalModeSpec(
    ...     num_modes=1,
    ...     impedance_specs={"port1": impedance_spec}
    ... )
    >>>
    >>> # Create terminal source
    >>> terminal_source = MicrowaveTerminalSource(
    ...     size=(10, 10, 0),
    ...     source_time=pulse,
    ...     mode_spec=terminal_spec,
    ...     terminal_label="port1",
    ...     direction='+'
    ... )

    See Also
    --------
    :class:`.MicrowaveTerminalModeSpec`:
        Specification for terminal modes with explicit impedance definitions.
    :class:`.ModeSource`:
        Standard mode source for photonic waveguide modes.
    """

    mode_spec: MicrowaveTerminalModeSpec = Field(
        ...,
        title="Terminal Mode Specification",
        description="Specification for terminal modes with explicit impedance definitions. "
        "Defines how voltage, current, and impedance are computed for each terminal.",
    )

    terminal_label: str = Field(
        ...,
        title="Terminal Label",
        description="String identifier for the terminal to excite. Must match the label of "
        "a terminal (single-ended or differential pair).",
    )

    @model_validator(mode="after")
    def _validate_terminal_label(self) -> Self:
        """Validate that terminal_label exists in mode_spec terminal indices."""
        terminal_indices = self.mode_spec._terminal_indices
        if self.terminal_label not in terminal_indices:
            raise ValueError(
                f"terminal_label '{self.terminal_label}' not found in mode_spec. "
                f"Available terminals: {terminal_indices}"
            )

        return self
