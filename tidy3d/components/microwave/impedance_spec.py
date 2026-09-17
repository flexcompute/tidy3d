"""Reference-impedance specification for a lumped port.

``ImpedanceSpec`` is published as ``tidy3d.ImpedanceSpec``. That exposure was a
mistake: the class is lumped-port machinery that belonged with the terminal
S-matrix plugin, and it was exported from the top level alongside the ports
that consume it. It shipped that way in 2.12, so the SemVer guarantee on the
top-level ``tidy3d`` namespace now pins it in place. It lives here rather than
under ``tidy3d/plugins/smatrix/`` because that plugin surface was removed at
2.13, and it must stay reachable as ``tidy3d.ImpedanceSpec`` until 3.0 removes
it along with the rest of the RF classes. See memo 0093.

Nothing in Tidy3D consumes it. The lumped ports that do are in Flexcompute RF
(``flexcompute.rf.tidy3d``), which imports this class rather than carrying a
copy, so the ``ImpedanceSpec`` type tag has one owner until 3.0 moves the
definition there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, PositiveFloat, model_validator

from tidy3d.components.lumped_element import _IMPEDANCE_ATOL
from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.types import Complex
from tidy3d.constants import HERTZ, OHM
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self

DEFAULT_REFERENCE_IMPEDANCE = 50


class ImpedanceSpec(MicrowaveBaseModel):
    """Impedance specification for a lumped port.

    Combines a reference impedance with an optional measurement frequency used to infer
    reactive components in the FDTD load model. For a purely real impedance (e.g. 50 Ω),
    only :attr:`impedance` is needed. For a complex impedance ``Z = R + jX``, :attr:`frequency`
    must also be provided so the imaginary part can be mapped to a series inductor or capacitor.

    Note
    ----
    The lumped ports that consume this specification live in Flexcompute RF
    (``flexcompute.rf.tidy3d``), which shares this class. The definition stays in
    Tidy3D until ``3.0``, when it moves to Flexcompute RF.

    Example
    -------
    >>> spec_real = ImpedanceSpec(impedance=50)
    >>> spec_inductive = ImpedanceSpec(impedance=50+30j, frequency=1e9)
    >>> spec_capacitive = ImpedanceSpec(impedance=50-20j, frequency=2e9)
    """

    impedance: Complex = Field(
        default=DEFAULT_REFERENCE_IMPEDANCE,
        title="Impedance",
        description="Reference port impedance ``Z = R + jX`` in ohms. "
        "For a complex value with non-zero imaginary part, :attr:`frequency` must be provided.",
        json_schema_extra={"units": OHM},
    )

    frequency: PositiveFloat | None = Field(
        default=None,
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
