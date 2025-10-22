"""Type definitions for mode specifications."""

from __future__ import annotations

from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.mode_spec import ModeSpec

# Type aliases
ModeSpecType = ModeSpec | MicrowaveModeSpec
