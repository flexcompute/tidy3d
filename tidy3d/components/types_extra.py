from typing import Any

from ..packaging import check_import

# TODO Complicated as trimesh should be a core package unless decoupled implementation types in functional location.
#  We need to restructure.
if check_import("trimesh"):
    import trimesh  # Won't add much overhead if already imported

    TrimeshType = trimesh.Trimesh
else:
    TrimeshType = Any
