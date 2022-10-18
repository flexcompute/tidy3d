# pylint:disable=unused-argument
""" Defines various validation functions that get used to ensure inputs are legit """
from typing import Any

import pydantic

from .geometry import Box
from ..log import ValidationError, SetupError

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


def get_value(key: str, values: dict) -> Any:
    """Grab value from values dictionary. If not present, raise an error before continuing."""
    val = values.get(key)
    if val is None:
        raise ValidationError(f"value {key} not defined, must be present to validate.")
    return val


def assert_plane():
    """makes sure a field's `size` attribute has exactly 1 zero"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.count(0.0) != 1:
            raise ValidationError(f"'{cls.__name__}' object must be planar, given size={val}")
        return val

    return is_plane


def validate_name_str():
    """make sure the name doesnt include [, ] (used for default names)"""

    @pydantic.validator("name", allow_reuse=True, always=True, pre=True)
    def field_has_unique_names(cls, val):
        """raise exception if '[' or ']' in name"""
        # if val and ('[' in val or ']' in val):
        #     raise SetupError(f"'[' or ']' not allowed in name: {val} (used for defaults)")
        return val

    return field_has_unique_names


def validate_unique(field_name: str):
    """Make sure the given field has unique entries."""

    @pydantic.validator(field_name, always=True, allow_reuse=True)
    def field_has_unique_entries(cls, val):
        """Check if the field has unique entries."""
        if len(set(val)) != len(val):
            raise SetupError(f"Entries of '{field_name}' must be unique.")
        return val

    return field_has_unique_entries


def validate_mode_objects_symmetry(field_name: str):
    """If a Mode object, this checks that the object is fully in the main quadrant in the presence
    of symmetry along a given axis, or else centered on the symmetry center."""

    obj_type = "ModeSource" if field_name == "sources" else "ModeMonitor"

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def check_symmetry(cls, val, values):
        """check for intersection of each structure with simulation bounds."""
        sim_center = values.get("center")
        for position_index, geometric_object in enumerate(val):
            if geometric_object.type == obj_type:
                bounds_min, _ = geometric_object.bounds
                for dim, sym in enumerate(values.get("symmetry")):
                    if (
                        sym != 0
                        and bounds_min[dim] < sim_center[dim]
                        and geometric_object.center[dim] != sim_center[dim]
                    ):
                        raise SetupError(
                            f"Mode object '{geometric_object}' "
                            f"(at `simulation.{field_name}[{position_index}]`) "
                            "in presence of symmetries must be in the main quadrant, "
                            "or centered on the symmetry axis."
                        )

        return val

    return check_symmetry


def assert_unique_names(field_name: str):
    """makes sure all elements of a field have unique .name values"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def field_has_unique_names(cls, val, values):
        """make sure each element of val has a unique name (if specified)."""
        field_names = [field.name for field in val if field.name]
        unique_names = set(field_names)
        if len(unique_names) != len(field_names):
            raise SetupError(f"'{field_name}' names are not unique, given {field_names}.")
        return val

    return field_has_unique_names


def assert_objects_in_sim_bounds(field_name: str):
    """Makes sure all objects in field are at least partially inside of simulation bounds."""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def objects_in_sim_bounds(cls, val, values):
        """check for intersection of each structure with simulation bounds."""
        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        for position_index, geometric_object in enumerate(val):
            if not sim_box.intersects(geometric_object.geometry):
                raise SetupError(
                    f"'{geometric_object}' "
                    f"(at `simulation.{field_name}[{position_index}]`) "
                    "is completely outside of simulation domain."
                )

        return val

    return objects_in_sim_bounds


def enforce_monitor_fields_present():
    """Make sure all of the fields in the monitor are present in the correponding data."""

    @pydantic.root_validator(skip_on_failure=True, allow_reuse=True)
    def _contains_fields(cls, values):
        """Make sure the initially specified fields are here."""
        for field_name in values.get("monitor").fields:
            if values.get(field_name) is None:
                raise SetupError(f"missing field {field_name}")
        return values

    return _contains_fields


def required_if_symmetry_present(field_name: str):
    """Make a field required (not None) if any non-zero symmetry eigenvalue is present."""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def _make_required(cls, val, values):
        """Ensure val is not None if the symmetry is non-zero along any dimension."""
        symmetry = values.get("symmetry")
        if any(sym_val != 0 for sym_val in symmetry) and val is None:
            raise SetupError(f"'{field_name}' must be provided if symmetry present.")
        return val

    return _make_required
