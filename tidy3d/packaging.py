"""
This file contains a set of functions relating to packaging tidy3d for distribution. Sections of the codebase should depend on this file, but this file should not depend on any other part of the codebase.

This section should only depend on the standard core installation in the pyproject.toml, and should not depend on any other part of the codebase optional imports.

Re-exports the optional third-party guards from :mod:`tidy3d._optional_imports`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, TypeVar

from tidy3d._optional_imports import (  # noqa: F401
    check_import,
    get_numpy_major_version,
    pyvista,
    requires_pyvista,
    requires_vtk,
    verify_packages_import,
    vtk,
)
from tidy3d.config import config

from .exceptions import Tidy3dImportError
from .version import __version__

F = TypeVar("F", bound=Callable[..., Any])

tidy3d_extras = {"mod": None, "use_local_subpixel": None}


def _check_tidy3d_extras_available(quiet: bool = False) -> None:
    """Helper function to check if 'tidy3d-extras' is available and version matched.

    Parameters
    ----------
    quiet : bool
        If True, suppress error logging when raising exceptions.

    Raises
    ------
    Tidy3dImportError
        If tidy3d-extras is not available or not properly initialized.
    """
    if tidy3d_extras["mod"] is not None:
        return

    module_exists = find_spec("tidy3d_extras") is not None
    if not module_exists:
        raise Tidy3dImportError(
            "The package 'tidy3d-extras' is absent. "
            "Please install the 'tidy3d-extras' package using, for "
            r"example, 'pip install tidy3d\[extras]'.",
            log_error=not quiet,
        )

    try:
        import tidy3d_extras as tidy3d_extras_mod

    except ImportError as exc:
        raise Tidy3dImportError(
            "The package 'tidy3d-extras' did not initialize correctly.",
            log_error=not quiet,
        ) from exc

    if not hasattr(tidy3d_extras_mod, "__version__"):
        raise Tidy3dImportError(
            "The package 'tidy3d-extras' did not initialize correctly. "
            "Please install the 'tidy3d-extras' package using, for "
            r"example, 'pip install tidy3d\[extras]'.",
            log_error=not quiet,
        )

    version = tidy3d_extras_mod.__version__

    if version is None:
        raise Tidy3dImportError(
            "The package 'tidy3d-extras' did not initialize correctly, "
            "likely due to an invalid API key.",
            log_error=not quiet,
        )

    if version != __version__:
        raise Tidy3dImportError(
            f"The version of 'tidy3d-extras' is {version}, but the version of 'tidy3d' is {__version__}. "
            "They must match. You can install the correct "
            r"version using 'pip install tidy3d\[extras]'.",
            log_error=not quiet,
        )

    tidy3d_extras["mod"] = tidy3d_extras_mod


def check_tidy3d_extras_licensed_feature(feature_name: str, quiet: bool = False) -> None:
    """Helper function to check if a specific feature is licensed in 'tidy3d-extras'.

    Parameters
    ----------
    feature_name : str
        The name of the feature to check for.
    quiet : bool
        If True, suppress error logging when raising exceptions.

    Raises
    ------
    Tidy3dImportError
        If the feature is not available with your license.
    """

    try:
        _check_tidy3d_extras_available(quiet=quiet)
    except Tidy3dImportError as exc:
        raise Tidy3dImportError(
            f"The package 'tidy3d-extras' is required for this feature '{feature_name}'.",
            log_error=not quiet,
        ) from exc

    features = tidy3d_extras["mod"].extension._features()
    if feature_name not in features:
        raise Tidy3dImportError(
            f"The feature '{feature_name}' is not available with your license. "
            "Please contact Tidy3D support, or upgrade your license.",
            log_error=not quiet,
        )


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
