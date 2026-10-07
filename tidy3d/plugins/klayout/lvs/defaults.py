"""Default values for KLayout LVS."""

from __future__ import annotations

from pathlib import Path

DEFAULT_RESULTSFILE = Path("lvs_results.lvsdb")
DEFAULT_VERBOSE = True
# The runset variables that receive the layout, schematic and results paths, by default name.
FILE_VARIABLES = ("gdsfile", "schematicfile", "resultsfile")
