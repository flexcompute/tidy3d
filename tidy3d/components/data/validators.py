# special validators for Datasets
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import field_validator

from tidy3d.exceptions import ValidationError

from .data_array import DataArray
from .dataset import AbstractFieldDataset, ScalarFieldDataArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_core.core_schema import ValidationInfo


# this can't go in validators.py because that file imports dataset.py
def validate_no_nans(*field_names: str) -> Callable[[Any, ValidationInfo], Any]:
    """Raise validation error if nans found in Dataset, or other data-containing item."""

    @field_validator(*field_names)
    def no_nans(val: Any, info: ValidationInfo) -> Any:
        """Raise validation error if nans found in Dataset, or other data-containing item."""

        if val is None:
            return val

        def error_if_has_nans(value: Any, identifier: str | None = None) -> None:
            """Recursively check if value (or iterable) has nans and error if so."""

            def has_nans(values: Any) -> bool:
                """Base case: do these values contain NaN?"""
                try:
                    return np.any(np.isnan(values))
                # if this fails for some reason (fails in adjoint, for example), don't check it.
                except Exception:
                    return False

            if isinstance(value, (tuple, list)):
                for i, _value in enumerate(value):
                    error_if_has_nans(_value, identifier=f"[{i}]")

            elif isinstance(value, AbstractFieldDataset):
                for key, val in value.field_components.items():
                    error_if_has_nans(val, identifier=f".{key}")

            elif isinstance(value, DataArray):
                error_if_has_nans(value.values)

            else:
                if has_nans(value):
                    # the identifier is used to make the message more clear by appending some more info
                    field_name_display = info.field_name
                    if identifier:
                        field_name_display += identifier

                    raise ValidationError(
                        f"Found 'NaN' values in '{field_name_display}'. "
                        "If they were not intended, please double check your construction. "
                        "If intended, to replace these data points with a value 'x', "
                        "call 'values = np.nan_to_num(values, nan=x)'."
                    )

        error_if_has_nans(val)
        return val

    return no_nans


def validate_can_interpolate(
    *field_names: str,
) -> Callable[[AbstractFieldDataset], AbstractFieldDataset]:
    """Make sure the data in ``field_name`` can be interpolated."""

    @field_validator(*field_names)
    def check_fields_interpolate(val: AbstractFieldDataset) -> AbstractFieldDataset:
        if isinstance(val, AbstractFieldDataset):
            for name, data in val.field_components.items():
                if isinstance(data, ScalarFieldDataArray):
                    data._interp_validator(name)
        return val

    return check_fields_interpolate
