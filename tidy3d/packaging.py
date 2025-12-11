"""
This file contains a set of functions relating to packaging tidy3d for distribution. Sections of the codebase should depend on this file, but this file should not depend on any other part of the codebase.

This section should only depend on the standard core installation in the pyproject.toml, and should not depend on any other part of the codebase optional imports.
"""

from __future__ import annotations

import functools
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np

from tidy3d.config import config

from .exceptions import Tidy3dImportError
from .version import __version__

vtk = {
    "mod": None,
    "id_type": np.int64,
    "vtk_to_numpy": None,
    "numpy_to_vtkIdTypeArray": None,
    "numpy_to_vtk": None,
}

tidy3d_extras = {"mod": None, "use_local_subpixel": None}


def check_import(module_name: str) -> bool:
    """
    Check if a module or submodule section has been imported. This is a functional way of loading packages that will still load the corresponding module into the total space.

    Parameters
    ----------
    module_name

    Returns
    -------
    bool
        True if the module has been imported, False otherwise.

    """
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False


def verify_packages_import(modules: list, required: Literal["any", "all"] = "all"):
    def decorator(func):
        """
        When decorating a method, requires that the specified modules are available. It will raise an error if the
        module is not available depending on the value of the 'required' parameter which represents the type of
        import required.

        There are a few options to choose for the 'required' parameter:
        - 'all': All the modules must be available for the operation to continue without raising an error
        - 'any': At least one of the modules must be available for the operation to continue without raising an error

        Parameters
        ----------
        func
            The function to decorate.

        Returns
        -------
        checks_modules_import
            The decorated function.

        """

        @functools.wraps(func)
        def checks_modules_import(*args: Any, **kwargs: Any):
            """
            Checks if the modules are available. If they are not available, it will raise an error depending on the value.
            """
            available_modules_status = []
            maximum_amount_modules = len(modules)

            module_id_i = 0
            for module in modules:
                # Starts counting from one so that it can be compared to len(modules)
                module_id_i += 1
                import_available = check_import(module)
                available_modules_status.append(
                    import_available
                )  # Stores the status of the module import

                if not import_available:
                    if required == "all":
                        raise Tidy3dImportError(
                            f"The package '{module}' is required for this operation, but it was not found. "
                            f"Please install the '{module}' dependencies using, for example, "
                            f"'pip install tidy3d[<see_options_in_pyproject.toml>]"
                        )
                    if required == "any":
                        # Means we need to verify that at least one of the modules is available
                        if (
                            not any(available_modules_status)
                        ) and module_id_i == maximum_amount_modules:
                            # Means that we have reached the last module and none of them were available
                            raise Tidy3dImportError(
                                f"The package '{module}' is required for this operation, but it was not found. "
                                f"Please install the '{module}' dependencies using, for example, "
                                f"'pip install tidy3d[<see_options_in_pyproject.toml>]"
                            )
                    else:
                        raise ValueError(
                            f"The value '{required}' is not a valid value for the 'required' parameter. "
                            f"Please use any 'all' or 'any'."
                        )
                else:
                    # Means that the module is available, so we can just continue with the operation
                    pass
            return func(*args, **kwargs)

        return checks_modules_import

    return decorator


def requires_vtk(fn):
    """When decorating a method, requires that vtk is available."""

    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any):
        if vtk["mod"] is None:
            try:
                import vtk as vtk_mod
                from vtk.util.numpy_support import (
                    numpy_to_vtk,
                    numpy_to_vtkIdTypeArray,
                    vtk_to_numpy,
                )
                from vtkmodules.vtkCommonCore import vtkLogger

                vtk["mod"] = vtk_mod
                vtk["vtk_to_numpy"] = vtk_to_numpy
                vtk["numpy_to_vtkIdTypeArray"] = numpy_to_vtkIdTypeArray
                vtk["numpy_to_vtk"] = numpy_to_vtk

                vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_WARNING)

                if vtk["mod"].vtkIdTypeArray().GetDataTypeSize() == 4:
                    vtk["id_type"] = np.int32

            except ImportError as exc:
                raise Tidy3dImportError(
                    "The package 'vtk' is required for this operation, but it was not found. "
                    "Please install the 'vtk' dependencies using, for example, "
                    "'pip install .[vtk]'."
                ) from exc

        return fn(*args, **kwargs)

    return _fn


def get_numpy_major_version(module=np):
    """
    Extracts the major version of the installed numpy accordingly.

    Parameters
    ----------
    module : module
        The module to extract the version from. Default is numpy.

    Returns
    -------
    int
        The major version of the module.
    """
    # Get the version of the module
    module_version = module.__version__

    # Extract the major version number
    major_version = int(module_version.split(".")[0])

    return major_version


def _check_tidy3d_extras_available(quiet: bool = False):
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


def check_tidy3d_extras_licensed_feature(feature_name: str, quiet: bool = False):
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


def supports_local_subpixel(fn):
    """When decorating a method, checks that 'tidy3d-extras' is available,
    conditioned on 'config.simulation.use_local_subpixel'."""

    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any):
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


def disable_local_subpixel(fn):
    """When decorating a method, temporarily disables local subpixel."""

    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any):
        simulation = config.simulation
        previous = simulation.use_local_subpixel

        simulation.use_local_subpixel = False
        try:
            return fn(*args, **kwargs)
        finally:
            simulation.use_local_subpixel = previous

    return _fn
