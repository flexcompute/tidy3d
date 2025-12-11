"""Runtime environment detection for tidy3d.

This module must have ZERO dependencies on other tidy3d modules to avoid
circular imports. It is imported very early in the initialization chain.
"""

from __future__ import annotations

import sys

# Detect WASM/Pyodide environment where web and filesystem features are unavailable
WASM_BUILD = "pyodide" in sys.modules or sys.platform == "emscripten"
