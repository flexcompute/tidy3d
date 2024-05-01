# base class for all of the invdes fields
from __future__ import annotations

import abc
from types import FunctionType
import inspect
import typing
import pathlib

import jax
import numpy as np
import dill
import jax.numpy as jnp
import jaxlib
import tidy3d as td

"""
Notes on the code block below:

'ArrayImpl' type needed to tell pydantic how to serialize jax arrays. See bottom of file for detail.
We try to import this type, but these imports can sometimes change with jax versions.
So if that fails, we can warn the user to file an issue, and use a standin instead of erroring.
The code will still work, but users wont be able to write jax-containing objects to file until it's
properly fixed.
"""
try:
    from jaxlib.xla_extension import ArrayImpl
except ImportError:
    td.log.warning(
        "Could not import 'ArrayImpl' in this version of 'jax'. "
        "If you encountered this error, please file an 'Issue' on the 'tidy3d' front end github "
        "repository at 'https://github.com/flexcompute/tidy3d/issues'. To help us fix the issue, "
        "please mention the version of 'tidy3d', 'jax', and 'jaxlib', printed below: \n"
        f"\ttidy3d=={td.__version__}\n"
        f"\tjax=={jax.__version__}\n"
        f"\tjaxlib=={jaxlib.__version__}\n"
    )
    ArrayImpl = jnp.ndarray


class InvdesBaseModel(td.components.base.Tidy3dBaseModel, abc.ABC):
    """Base class for ``invdes`` components, with special function serialization and file IO."""

    @staticmethod
    def _get_fn_source(function: FunctionType) -> str:
        """Get a function source as a string, return ``None`` if not available."""
        try:
            return inspect.getsource(function)
        except (TypeError, OSError):
            return None

    @staticmethod
    def _make_np_array(arr: typing.Any) -> np.ndarray:
        """Turn any array into a ``numpy`` array."""
        jax_arr = jnp.array(arr)
        untraced_arr = jax.lax.stop_gradient(jax_arr)
        return np.array(untraced_arr)

    @staticmethod
    def _check_fname(fname: str) -> None:
        """Warn if fname isn't pickle."""
        suffix = pathlib.Path(fname).suffix
        if not any(suffix.lower == ext for ext in (".pkl", ".pickle", ".dill")):
            td.log.warning(
                "'invdes' components must be saved using 'dill'. "
                f"Found a filename of '{fname}'. Consider using a 'pickle' extension, such as "
                "'.pkl' if you aren't already."
            )

    def to_file(self, fname: str) -> None:
        """Saving ``invdes`` components to file. Only pickle files are supported."""
        self._check_fname(fname)
        with open(fname, "wb") as f_handle:
            dill.dump(self.dict(), f_handle)

    @classmethod
    def from_file(cls, fname: str) -> InvdesBaseModel:
        """Loading ``invdes`` components from file. Only pickle files are supported."""
        cls._check_fname(fname)
        with open(fname, "rb") as f_handle:
            self_dict = dill.load(f_handle)
        return cls(**self_dict)


# set the json encoders for function types and for jax arrays
invdes_encoders = InvdesBaseModel.__config__.json_encoders
invdes_encoders[FunctionType] = InvdesBaseModel._get_fn_source
invdes_encoders[ArrayImpl] = InvdesBaseModel._make_np_array
