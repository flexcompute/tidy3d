"""Defines custom error messages."""

from ...log import Tidy3dError


class AdjointError(Tidy3dError):
    """An error in setting up the adjoint solver."""
