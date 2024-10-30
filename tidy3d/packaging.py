"""
This file contains a set of functions relating to packaging tidy3d for distribution. Sections of the codebase should depend on this file, but this file should not depend on any other part of the codebase.

This section should only depend on the standard core installation in the pyproject.toml, and should not depend on any other part of the codebase optional imports.
"""

import functools
from importlib import import_module
from typing import Literal

import numpy as np

from .exceptions import Tidy3dImportError

vtk = {
    "mod": None,
    "id_type": np.int64,
    "vtk_to_numpy": None,
    "numpy_to_vtkIdTypeArray": None,
    "numpy_to_vtk": None,
}


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
        def checks_modules_import(*args, **kwargs):
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
                    elif required == "any":
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
    def _fn(*args, **kwargs):
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

            except ImportError:
                raise Tidy3dImportError(
                    "The package 'vtk' is required for this operation, but it was not found. "
                    "Please install the 'vtk' dependencies using, for example, "
                    "'pip install .[vtk]'."
                )

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
