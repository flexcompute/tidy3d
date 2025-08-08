from __future__ import annotations

import importlib
import warnings


def _reload_smatrix_in_warns():
    import tidy3d.plugins.smatrix as smatrix  # ensure it is importable

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(smatrix)
    return [str(w.message) for w in caught]


def test_smatrix_rf_shim_deprecation_warning_on_import():
    messages = _reload_smatrix_in_warns()
    assert any("RF APIs moved" in m for m in messages), (
        "Expected RF deprecation warning not emitted"
    )


def test_smatrix_rf_shim_symbol_equivalence():
    # Import canonical RF symbols first
    # Reload smatrix to ensure re-exports are present then import shimmed names
    import tidy3d.plugins.smatrix as smatrix
    from tidy3d.plugins.rf import (
        CoaxialLumpedPort as RFCoaxialLumpedPort,
    )
    from tidy3d.plugins.rf import (
        LumpedPort as RFLumpedPort,
    )
    from tidy3d.plugins.rf import (
        TerminalComponentModeler as RFTerminalComponentModeler,
    )
    from tidy3d.plugins.rf import (
        WavePort as RFWavePort,
    )
    from tidy3d.plugins.rf import (
        ab_to_s as rf_ab_to_s,
    )
    from tidy3d.plugins.rf import (
        s_to_z as rf_s_to_z,
    )

    importlib.reload(smatrix)

    ShimCoaxialLumpedPort = smatrix.CoaxialLumpedPort
    ShimLumpedPort = smatrix.LumpedPort
    ShimTerminalComponentModeler = smatrix.TerminalComponentModeler
    ShimWavePort = smatrix.WavePort
    shim_ab_to_s = smatrix.ab_to_s
    shim_s_to_z = smatrix.s_to_z

    # Classes and functions should be exactly the same objects
    assert ShimTerminalComponentModeler is RFTerminalComponentModeler
    assert ShimLumpedPort is RFLumpedPort
    assert ShimCoaxialLumpedPort is RFCoaxialLumpedPort
    assert ShimWavePort is RFWavePort
    assert shim_ab_to_s is rf_ab_to_s
    assert shim_s_to_z is rf_s_to_z


def test_smatrix_rf_shim_utils_equivalence():
    # Canonical RF funcs
    # Reload smatrix then get shims
    import tidy3d.plugins.smatrix as smatrix
    from tidy3d.plugins.rf import (
        check_port_impedance_sign as rf_check_port_impedance_sign,
    )
    from tidy3d.plugins.rf import (
        compute_F as rf_compute_F,
    )
    from tidy3d.plugins.rf import (
        compute_port_VI as rf_compute_port_VI,
    )
    from tidy3d.plugins.rf import (
        compute_power_delivered_by_port as rf_compute_power_delivered_by_port,
    )
    from tidy3d.plugins.rf import (
        compute_power_wave_amplitudes as rf_compute_power_wave_amplitudes,
    )

    importlib.reload(smatrix)

    shim_check_port_impedance_sign = smatrix.check_port_impedance_sign
    shim_compute_F = smatrix.compute_F
    shim_compute_port_VI = smatrix.compute_port_VI
    shim_compute_power_delivered_by_port = smatrix.compute_power_delivered_by_port
    shim_compute_power_wave_amplitudes = smatrix.compute_power_wave_amplitudes

    assert shim_compute_F is rf_compute_F
    assert shim_compute_power_wave_amplitudes is rf_compute_power_wave_amplitudes
    assert shim_compute_power_delivered_by_port is rf_compute_power_delivered_by_port
    assert shim_compute_port_VI is rf_compute_port_VI
    assert shim_check_port_impedance_sign is rf_check_port_impedance_sign
