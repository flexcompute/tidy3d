"""preserve from tidy3d.web.environment import Env backward compatibility"""

from __future__ import annotations

# TODO(FXC-3827): Remove this re-export once `tidy3d.web.environment` shim is retired in 2.12.
from .core.environment import Env

__all__ = ["Env"]
