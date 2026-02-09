from __future__ import annotations

from typing import Any

from tidy3d.packaging import check_import

# TODO Complicated as trimesh should be a core package unless decoupled implementation types in functional location.
#  We need to restructure.
if check_import("trimesh"):
    try:
        import trimesh

        TrimeshType = trimesh.Trimesh
    except ImportError:
        # Package is installed but broken (e.g., missing dependencies)
        TrimeshType = Any
else:
    TrimeshType = Any
