"""Microwave sources for RF and transmission line simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.microwave.mode_spec import MicrowaveTerminalModeSpec
from tidy3d.components.source.base import Source
from tidy3d.components.source.field import AbstractModeSource
from tidy3d.components.types import Axis
from tidy3d.components.validators import assert_line_or_plane

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

    # This class has no construction scope, so it calls the source rule that replaced the
    # ``BroadbandSource`` validator directly until the rf unit migrates it.
    @model_validator(mode="after")
    def _validate_num_freqs_limit(self) -> Self:
        """Enforce method-specific bounds on num_freqs."""
        from flexcompute.core._migration.em.source.validation import validate_num_freqs_limit

        from tidy3d.em.translate.components import _present

        _present(self, validate_num_freqs_limit(self))
        return self


class LumpedVoltageSource(Source):
    """Ideal voltage source across a lumped gap.

    The source holds the potential at the end of the box with the larger coordinate along
    ``voltage_axis`` above the potential at the other end by the voltage of ``source_time``, in
    volts, along every path through the box parallel to ``voltage_axis``. The same convention
    defines the voltage of a :class:`.LumpedPort`. After its waveform ends, and throughout when its
    amplitude is zero, the source holds the voltage at zero, so it shorts the gap.

    The box is planar (one zero-size dimension) or a line (two zero-size dimensions), like a
    lumped element, and is placed on the grid like one: a sheet snapped to the grid, a line widened
    to a sheet one grid cell wide. Before upload, a source that cannot set its voltage where it is
    placed is rejected:

    - not completely inside the simulation domain, or driving no grid edge;
    - within half a grid cell along ``voltage_axis``, so both its ends snap to one grid plane;
    - its gap shorted by a conductor, or meeting another voltage source;
    - touching a lumped element or a 2D material other than at its two ends, or overlapping one;
    - on a symmetry plane that holds its voltage at zero;
    - next to a medium a lumped element cannot be placed in.

    The source is meant for a gap in a dielectric or vacuum: a conductive medium inside the gap
    lowers the voltage it sets, by up to a few percent for a metal.

    Frequency-domain monitor data is normalized by the spectrum of the source voltage.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=5e9, fwidth=2e9)
    >>> source = LumpedVoltageSource(
    ...     center=(0, 0, 500), size=(3000, 0, 1000), voltage_axis=2, source_time=pulse
    ... )
    """

    voltage_axis: Axis = Field(
        title="Voltage Axis",
        description="Axis along which the voltage is applied. Must be a nonzero-size dimension "
        "of the source.",
    )

    _plane_validator = assert_line_or_plane()

    @model_validator(mode="after")
    def _voltage_axis_has_extent(self) -> Self:
        """The voltage is applied across a nonzero extent."""
        if self.size[self.voltage_axis] == 0:
            self._raise_validation_error_at_loc(
                "'voltage_axis' must be a nonzero-size dimension of the source.", "voltage_axis"
            )
        return self

    @cached_property
    def _pol_vector(self) -> tuple[float, float, float]:
        """Unit vector along ``voltage_axis``, for plotting."""
        pol_vector = [0.0, 0.0, 0.0]
        pol_vector[self.voltage_axis] = 1.0
        return tuple(pol_vector)
