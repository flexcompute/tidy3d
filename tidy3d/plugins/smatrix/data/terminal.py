"""Migration shim for terminal component-modeler data now owned by Flexcompute RF."""

from __future__ import annotations

from tidy3d._rf_migration import migrated_rf_module

__getattr__ = migrated_rf_module(__name__)
