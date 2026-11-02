"""Translators for source classes whose schema needs data only public Tidy3D can compute."""

from __future__ import annotations

from typing import Any

from tidy3d.exceptions import SetupError, format_chained_exception_message


def _remove_dc_component(derivative_order: int) -> bool:
    """``remove_dc_component`` of an extension Gaussian pulse: derivative order 1 removes it."""

    if derivative_order not in (0, 1):
        raise SetupError(
            f"The 'tidy3d-extras' BroadbandPulse holds a Gaussian pulse of derivative order "
            f"{derivative_order}; the schema BroadbandPulse represents orders 0 and 1 only."
        )
    return derivative_order == 1


def fill_broadband_pulse(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill the schema-only fields from the ``tidy3d-extras`` pulse, so the schema evaluates it
    without the extension.

    A missing extension stops the translation before any other check, then a ``freq_range``
    the source rule rejects stops it with the rule's error at the field.
    """

    from flexcompute.core._migration.em.source.validation import validate_broadband_freq_range

    from tidy3d.components.source.time import BroadbandPulse
    from tidy3d.em.translate.components import _present
    from tidy3d.packaging import check_tidy3d_extras_licensed_feature

    check_tidy3d_extras_licensed_feature("BroadbandPulse")
    # model_construct: validating would run this translation again on the construction path.
    pulse = BroadbandPulse.model_construct(**payload)
    # The extension cannot build a pulse from a range the source rule rejects.
    _present(pulse, validate_broadband_freq_range(pulse.freq_range))
    # The schema applies ``amplitude`` and ``phase`` itself, so it gets the unit pulse: the
    # extension scales every Gaussian pulse by ``amplitude`` and shifts every phase by ``phase``.
    unit_pulse = BroadbandPulse.model_construct(**{**payload, "amplitude": 1.0, "phase": 0.0})
    # The extension exposes its Gaussian pulses only through pickling, as
    # ``(reconstructor, (gaussian_pulses,))``; read them from a public accessor once it has one.
    # Any other layout or derivative order fails here with a named ``SetupError``.
    try:
        _, (gaussian_pulses,) = unit_pulse._source.__reduce__()
        gaussian_pulses = tuple(
            (freq0, fwidth, amplitude, phase, offset, _remove_dc_component(derivative_order))
            for freq0, fwidth, amplitude, phase, offset, derivative_order in gaussian_pulses
        )
    except SetupError:
        # ``SetupError`` is a ``ValueError``: an unknown derivative order keeps its own message.
        raise
    except (ValueError, TypeError) as exc:
        raise SetupError(
            format_chained_exception_message(
                "Internal error: unexpected 'tidy3d-extras' BroadbandPulse layout", exc
            )
        ) from exc
    return {
        **payload,
        "gaussian_pulses": gaussian_pulses,
        # From the pulse as given, so the schema's grid and run time match bit for bit.
        "pulse_freq0": pulse._freq0,
        "pulse_end_time": pulse.end_time(),
    }
