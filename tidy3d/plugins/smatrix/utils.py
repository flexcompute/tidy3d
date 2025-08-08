"""Deprecation shims for RF utils; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.utils import (
    ab_to_s as _ab_to_s,
)
from tidy3d.plugins.rf.utils import (
    check_port_impedance_sign as _check_port_impedance_sign,
)
from tidy3d.plugins.rf.utils import (
    compute_F as _compute_F,
)
from tidy3d.plugins.rf.utils import (
    compute_port_VI as _compute_port_VI,
)
from tidy3d.plugins.rf.utils import (
    compute_power_delivered_by_port as _compute_power_delivered_by_port,
)
from tidy3d.plugins.rf.utils import (
    compute_power_wave_amplitudes as _compute_power_wave_amplitudes,
)
from tidy3d.plugins.rf.utils import (
    s_to_z as _s_to_z,
)

__all__ = [
    "ab_to_s",
    "check_port_impedance_sign",
    "compute_F",
    "compute_port_VI",
    "compute_power_delivered_by_port",
    "compute_power_wave_amplitudes",
    "s_to_z",
]

warnings.warn(
    "tidy3d.plugins.smatrix.utils RF utilities are deprecated; use tidy3d.plugins.rf.utils",
    DeprecationWarning,
    stacklevel=2,
)

ab_to_s = _ab_to_s
s_to_z = _s_to_z
compute_F = _compute_F
compute_port_VI = _compute_port_VI
compute_power_wave_amplitudes = _compute_power_wave_amplitudes
compute_power_delivered_by_port = _compute_power_delivered_by_port
check_port_impedance_sign = _check_port_impedance_sign
