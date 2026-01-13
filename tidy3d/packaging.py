"""Compatibility shim for :mod:`tidy3d._common.packaging`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as partially migrated to _common
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from tidy3d._common.config import config
from tidy3d._common.exceptions import Tidy3dImportError
from tidy3d._common.packaging import (
    _check_tidy3d_extras_available,
    check_import,
    check_tidy3d_extras_licensed_feature,
    get_numpy_major_version,
    requires_vtk,
    tidy3d_extras,
    verify_packages_import,
    vtk,
)

if TYPE_CHECKING:
    from tidy3d._common.packaging import F


def supports_local_subpixel(fn: F) -> F:
    """When decorating a method, checks that 'tidy3d-extras' is available,
    conditioned on 'config.simulation.use_local_subpixel'."""

    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any) -> Any:
        preference = config.simulation.use_local_subpixel

        if preference is False:
            tidy3d_extras["use_local_subpixel"] = False
            return fn(*args, **kwargs)

        try:
            check_tidy3d_extras_licensed_feature("local_subpixel", quiet=(preference is None))
        except Tidy3dImportError as exc:
            tidy3d_extras["use_local_subpixel"] = False
            if preference is True:
                raise Tidy3dImportError(
                    "To suppress this error, you can set "
                    "'config.simulation.use_local_subpixel=False'."
                ) from exc
            # preference is None, so we can just return
            return fn(*args, **kwargs)

        # local_subpixel is available
        tidy3d_extras["use_local_subpixel"] = True
        return fn(*args, **kwargs)

    return _fn


def disable_local_subpixel(fn: F) -> F:
    """When decorating a method, temporarily disables local subpixel."""

    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any) -> Any:
        simulation = config.simulation
        previous = simulation.use_local_subpixel

        simulation.use_local_subpixel = False
        try:
            return fn(*args, **kwargs)
        finally:
            simulation.use_local_subpixel = previous

    return _fn
