# base class for all of the invdes fields
from __future__ import annotations

import abc
import dill
from types import FunctionType
import jax.numpy as jnp
import inspect
import numpy as np
import typing
import jax
import tidy3d as td

# TODO: below is a little sketchy
try:
    from jaxlib.xla_extension import ArrayImpl
except ImportError:
    td.log.warning(
        "Could not import 'ArrayImpl' in this version of 'jax'. "
        "If you encountered this error, please file an 'Issue' on the 'tidy3d' front end github "
        "repository at 'https://github.com/flexcompute/tidy3d/issues'."
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
        jax_arr = jnp.array(arr)
        untraced_arr = jax.lax.stop_gradient(jax_arr)
        return np.array(untraced_arr)

    @staticmethod
    def _check_fname(fname: str) -> None:
        """Warn if fname isn't pickle."""
        if all(ext not in fname for ext in ("pkl", "pickle", "dill")):
            td.log.warning(
                "'invdes' components must be saved using 'dill'. "
                f"Found a filename of '{fname}'. Consider using a 'pickle' extension, such as "
                "'.pkl'."
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


# set the json encoder for function types to None
invdes_encoders = InvdesBaseModel.__config__.json_encoders
invdes_encoders[FunctionType] = InvdesBaseModel._get_fn_source
invdes_encoders[ArrayImpl] = InvdesBaseModel._make_np_array
