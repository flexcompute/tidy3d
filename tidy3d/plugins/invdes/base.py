# base class for all of the invdes fields
from __future__ import annotations

import abc
from types import FunctionType
import inspect

import tidy3d as td


class InvdesBaseModel(td.components.base.Tidy3dBaseModel, abc.ABC):
    """Base class for ``invdes`` components, with special function serialization and file IO."""

    @staticmethod
    def _get_fn_source(function: FunctionType) -> str:
        """Get a function source as a string, return ``None`` if not available."""
        try:
            return inspect.getsource(function)
        except (TypeError, OSError):
            return None


# set the json encoders for function types
invdes_encoders = InvdesBaseModel.__config__.json_encoders
invdes_encoders[FunctionType] = InvdesBaseModel._get_fn_source
