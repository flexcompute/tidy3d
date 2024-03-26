# validator utilities for invdes plugin

import typing
import pydantic.v1 as pd
import tidy3d as td


def ignore_inherited_field(field_name: str) -> typing.Callable:
    """Create validator that ignores a field inherited but not set by user."""

    @pd.validator(field_name, always=True)
    def _ignore_field(cls, val):
        """Ignore supplied field value and warn."""
        if val is not None:
            td.log.warning(
                f"Field '{field_name}' was supplied but the 'invdes' plugin will automatically "
                "set this field internally using the design region specifications. "
                "The supplied value will be ignored. "
            )
        return None

    return _ignore_field
