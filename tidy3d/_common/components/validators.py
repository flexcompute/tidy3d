"""Defines various validation functions that get used to ensure inputs are legit"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from tidy3d._common.components.autograd.utils import get_static, hasbox
from tidy3d._common.components.data.data_array import DATA_ARRAY_MAP
from tidy3d._common.exceptions import ValidationError
from tidy3d._common.log import log

if TYPE_CHECKING:
    from typing import Callable, Optional

    from pydantic import FieldValidationInfo

T = TypeVar("T")

""" Explanation of pydantic validators:

    Validators are class methods that are added to the models to validate their fields (kwargs).
    The functions on this page return validators based on config arguments
    and are generally in multiple components of tidy3d.
    The inner functions (validators) are decorated with @pydantic.validator, which is configured.
    First argument is the string of the field being validated in the model.
    ``allow_reuse`` lets us use the validator in more than one model.
    ``always`` makes sure if the model is changed, the validator gets called again.

    The function being decorated by @pydantic.validator generally takes
    ``cls`` the class that the validator is added to.
    ``val`` the value of the field being validated.
    ``values`` a dictionary containing all of the other fields of the model.
    It is important to note that the validator only has access to fields that are defined
    before the field being validated.
    Fields defined under the validated field will not be in ``values``.

    All validators generally should throw an exception if the validation fails
    and return val if it passes.
    Sometimes, we can use validators to change ``val`` or ``values``,
    but this should be done with caution as it can be hard to reason about.

    To add a validator from this file to the pydantic model,
    put it in the model's main body and assign it to a variable (class method).
    For example ``_plane_validator = assert_plane()``.
    Note, if the assigned name ``_plane_validator`` is used later on for another validator, say,
    the original validator will be overwritten so be aware of this.

    For more details: `Pydantic Validators <https://pydantic-docs.helpmanual.io/usage/validators/>`_
"""

# Lowest frequency supported (Hz)
MIN_FREQUENCY = 1e5

FloatArray = Union[Sequence[float], NDArray]


def _assert_min_freq(freqs: FloatArray, msg_start: str) -> None:
    """Check if all ``freqs`` are above the minimum frequency."""
    if np.min(freqs) < MIN_FREQUENCY:
        raise ValidationError(
            f"{msg_start} must be no lower than {MIN_FREQUENCY:.0e} Hz. "
            "Note that the unit of frequency is 'Hz'."
        )


def _warn_unsupported_traced_argument(
    *names: str,
) -> Callable[[type, Any, FieldValidationInfo], Any]:
    @field_validator(*names)
    @classmethod
    def _warn_traced_arg(cls: type, val: Any, info: FieldValidationInfo) -> Any:
        if hasbox(val):
            log.warning(
                f"Field '{info.field_name}' of '{cls.__name__}' received an autograd tracer "
                f"(i.e., a value being tracked for automatic differentiation). "
                f"Automatic differentiation through this field is unsupported, "
                f"so the tracer has been converted to its static value. "
                f"If you want to avoid this warning, you manually unbox the value "
                f"using the 'autograd.tracer.getval' function before passing it to Tidy3D."
            )
            return get_static(val)
        return val

    return _warn_traced_arg


def warn_if_dataset_none(
    field_name: str,
) -> Callable[[type, Optional[dict[str, Any]]], Optional[dict[str, Any]]]:
    """Warn if a Dataset field has None in its dictionary."""

    @field_validator(field_name, mode="before")
    @classmethod
    def _warn_if_none(cls: type, val: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Warn if the DataArrays fail to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning(f"Loading {field_name} without data.", custom_loc=[field_name])
                return None
        return val

    return _warn_if_none


# FIXME: this validator doesn't do anything
def validate_name_str() -> Callable[[type, Optional[str]], Optional[str]]:
    """make sure the name does not include [, ] (used for default names)"""

    @field_validator("name")
    @classmethod
    def field_has_unique_names(cls: type, val: Optional[str]) -> Optional[str]:
        """raise exception if '[' or ']' in name"""
        # if val and ('[' in val or ']' in val):
        #     raise SetupError(f"'[' or ']' not allowed in name: {val} (used for defaults)")
        return val

    return field_has_unique_names
