"""Type definitions for mode specifications."""

from __future__ import annotations

from typing import Union

from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.mode_spec import ModeSpec

# Type aliases
ModeSpecType = Union[ModeSpec, MicrowaveModeSpec]
