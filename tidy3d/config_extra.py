"""This file contains a set of configuration variables that are not required for the core functionality of tidy3d,
but are required by external integrations or extra functionality like the HEAT or CHARGE solvers, etc."""

import numpy as np
from .packaging import verify_packages_import

__all__ = ["get_vtk_id_type", "verify_and_configure_vtk"]


@verify_packages_import(["vtk"])
def verify_and_configure_vtk(fn):
    """
    This function just does a custom configuration of the vtk installation. It just sets a configuration for the vtk
    functional usage.

    """
    from vtkmodules.vtkCommonCore import vtkLogger

    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_WARNING)

    return fn


@verify_and_configure_vtk
def get_vtk_id_type() -> np.dtype:
    """Returns the numpy dtype corresponding to vtkIdType."""
    import vtk

    if vtk.vtkIdTypeArray().GetDataTypeSize() == 4:
        return np.int32
    else:
        return np.int64
