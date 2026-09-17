"""Migration shim for the microwave plugins now owned by Flexcompute RF.

Every module under this package was removed. This shim is the only one left and
it stands for all of them: Python imports a parent before it resolves a child,
so an old path from anywhere in the namespace reports the move here first.
"""

from __future__ import annotations

from tidy3d._rf_migration import migrated_rf_module

__getattr__ = migrated_rf_module(__name__)
