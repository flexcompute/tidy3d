"""Defines various validation functions that get used to ensure inputs are legit"""

import numpy as np
import pydantic.v1 as pydantic
from autograd.tracer import isbox

from ..exceptions import SetupError, ValidationError
from ..log import log
from .autograd.utils import get_static
from .base import DATA_ARRAY_MAP, skip_if_fields_missing
from .data.dataset import Dataset, FieldDataset
from .geometry.base import Box
from .mode_spec import ModeSpec
from .types import Tuple

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


def assert_line():
    """makes sure a field's ``size`` attribute has exactly 2 zeros"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_line(cls, val):
        """Raise validation error if not 1 dimensional."""
        if val.count(0.0) != 2:
            raise ValidationError(f"'{cls.__name__}' object must be a line, given size={val}")
        return val

    return is_line


def assert_plane():
    """makes sure a field's ``size`` attribute has exactly 1 zero"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.count(0.0) != 1:
            raise ValidationError(f"'{cls.__name__}' object must be planar, given size={val}")
        return val

    return is_plane


def assert_line_or_plane():
    """makes sure a field's ``size`` attribute has either 1 or 2 zeros"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_line_or_plane(cls, val):
        """Raise validation error if not a line or plane."""
        if val.count(0.0) == 0 or val.count(0.0) == 3:
            raise ValidationError(
                f"'{cls.__name__}' object must be a line or a plane, given size={val}. "
            )
        return val

    return is_line_or_plane


def assert_volumetric():
    """makes sure a field's ``size`` attribute has no zero entry"""

    @pydantic.validator("size", allow_reuse=True, always=True)
    def is_volumetric(cls, val):
        """Raise validation error if volume is 0."""
        if val.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' object must be volumetric, given size={val}. "
                "If intending to make a 2D simulation, please set the size of "
                f"'{cls.__name__}' along the zero dimension to a dummy non-zero value."
            )
        return val

    return is_volumetric


def validate_name_str():
    """make sure the name does not include [, ] (used for default names)"""

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
    @skip_if_fields_missing(["center", "symmetry"])
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
                            f"{obj_type} at 'simulation.{field_name}[{position_index}]' "
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


def assert_objects_in_sim_bounds(
    field_name: str, error: bool = True, strict_inequality: bool = False
):
    """Makes sure all objects in field are at least partially inside of simulation bounds."""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    @skip_if_fields_missing(["center", "size"])
    def objects_in_sim_bounds(cls, val, values):
        """check for intersection of each structure with simulation bounds."""
        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        # Do a strict check, unless simulation is 0D along a dimension
        strict_ineq = [size != 0 and strict_inequality for size in sim_size]

        with log as consolidated_logger:
            for position_index, geometric_object in enumerate(val):
                if not sim_box.intersects(geometric_object.geometry, strict_inequality=strict_ineq):
                    message = (
                        f"'simulation.{field_name}[{position_index}]' "
                        "is outside of the simulation domain."
                    )
                    custom_loc = [field_name, position_index]

                    if error:
                        raise SetupError(message)
                    consolidated_logger.warning(message, custom_loc=custom_loc)

        return val

    return objects_in_sim_bounds


def assert_objects_contained_in_sim_bounds(
    field_name: str, error: bool = True, strict_inequality: bool = False
):
    """Makes sure all objects in field are completely inside the simulation bounds."""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    @skip_if_fields_missing(["center", "size"])
    def objects_contained_in_sim_bounds(cls, val, values):
        """check for containment of each structure with simulation bounds."""
        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        # Do a strict check, unless simulation is 0D along a dimension
        strict_ineq = [size != 0 and strict_inequality for size in sim_size]

        with log as consolidated_logger:
            for position_index, geometric_object in enumerate(val):
                if not sim_box.contains(geometric_object.geometry, strict_inequality=strict_ineq):
                    message = (
                        f"'simulation.{field_name}[{position_index}]' "
                        "is not completely inside the simulation domain."
                    )
                    custom_loc = [field_name, position_index]

                    if error:
                        raise SetupError(message)
                    consolidated_logger.warning(message, custom_loc=custom_loc)

        return val

    return objects_contained_in_sim_bounds


def enforce_monitor_fields_present():
    """Make sure all of the fields in the monitor are present in the corresponding data."""

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
    @skip_if_fields_missing(["symmetry"])
    def _make_required(cls, val, values):
        """Ensure val is not None if the symmetry is non-zero along any dimension."""
        symmetry = values.get("symmetry")
        if any(sym_val != 0 for sym_val in symmetry) and val is None:
            raise SetupError(f"'{field_name}' must be provided if symmetry present.")
        return val

    return _make_required


def warn_if_dataset_none(field_name: str):
    """Warn if a Dataset field has None in its dictionary."""

    @pydantic.validator(field_name, pre=True, always=True, allow_reuse=True)
    def _warn_if_none(cls, val: Dataset) -> Dataset:
        """Warn if the DataArrays fail to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning(f"Loading {field_name} without data.", custom_loc=[field_name])
                return None
        return val

    return _warn_if_none


def assert_single_freq_in_range(field_name: str):
    """Assert only one frequency supplied in source and it's in source time range."""

    @pydantic.validator(field_name, always=True, allow_reuse=True)
    @skip_if_fields_missing(["source_time"])
    def _single_frequency_in_range(cls, val: FieldDataset, values: dict) -> FieldDataset:
        """Assert only one frequency supplied and it's in source time range."""
        if val is None:
            return val
        source_time = values.get("source_time")
        fmin, fmax = source_time.frequency_range()
        for name, scalar_field in val.field_components.items():
            freqs = scalar_field.f
            if len(freqs) != 1:
                raise SetupError(
                    f"'{field_name}.{name}' must have a single frequency, "
                    f"contains {len(freqs)} frequencies."
                )
            freq = float(freqs[0])
            if (freq < fmin) or (freq > fmax):
                raise SetupError(
                    f"'{field_name}.{name}' contains frequency: {freq:.2e} Hz, which is outside "
                    f"of the 'source_time' frequency range [{fmin:.2e}-{fmax:.2e}] Hz."
                )
        return val

    return _single_frequency_in_range


def _warn_potential_error(
    field_name: str,
    base_value: float,
    val_change_range: Tuple[float, float],
    allowed_real_range: Tuple[float, float],
    allowed_imag_range: Tuple[float, float],
):
    """Basic validation that perturbations do not drive a parameter out of physical bounds."""

    min_val = val_change_range[0] + base_value
    max_val = val_change_range[1] + base_value

    for part_range, part_func, part_name in zip(
        [allowed_real_range, allowed_imag_range], [np.real, np.imag], ["Re", "Im"]
    ):
        if part_range is not None:
            min_allowed, max_allowed = part_range

            if min_allowed is not None and part_func(min_val) < min_allowed:
                log.warning(
                    f"'{part_name}({field_name})' could "
                    f"become less than '{min_allowed}' for a perturbation "
                    "medium.",
                    custom_loc=[field_name],
                )

            if max_allowed is not None and part_func(max_val) > max_allowed:
                log.warning(
                    f"'{part_name}({field_name})' could "
                    f"become greater than '{max_allowed}' for a perturbation "
                    "medium.",
                    custom_loc=[field_name],
                )


def validate_parameter_perturbation(
    field_name: str,
    base_field_name: str,
    allowed_real_range: Tuple[Tuple[float, float], ...],
    allowed_imag_range: Tuple[Tuple[float, float], ...] = None,
    allowed_complex: bool = True,
):
    """Assert perturbations do not drive a parameter out of physical bounds."""

    @pydantic.validator(field_name, always=True, allow_reuse=True)
    def _warn_perturbed_val_range(cls, val, values):
        """Assert perturbations do not drive a parameter out of physical bounds."""

        if val is not None:
            # get base values
            base_values = values[base_field_name]

            # check that shapes of base parameter and perturbations coincide
            if np.shape(base_values) != np.shape(val):
                raise SetupError(
                    f"Shape of perturbations '{field_name}' ({np.shape(val)}) does not coincide"
                    f" with shape of base parameter '{base_field_name}' ({np.shape(base_values)})."
                )

            for tuple_ind, (base_tuple, perturb_tuple) in enumerate(
                zip(np.atleast_1d(base_values), np.atleast_1d(val))
            ):
                tuple_ind_str = "" if np.shape(base_values) == () else f"[{tuple_ind}]"
                for paramer_ind, (base_value, perturb, real_range, imag_range) in enumerate(
                    zip(
                        np.atleast_1d(base_tuple),
                        np.atleast_1d(perturb_tuple),
                        allowed_real_range,
                        allowed_imag_range,
                    )
                ):
                    if perturb is not None:
                        # check real/complex type
                        if perturb.is_complex and not allowed_complex:
                            raise SetupError(
                                f"Perturbation of '{base_field_name}' cannot be complex."
                            )

                        ind_pointer = tuple_ind_str + (
                            "" if np.shape(base_tuple) == () else f"[{paramer_ind}]"
                        )

                        sub_field_name = base_field_name + ind_pointer

                        _warn_potential_error(
                            field_name=sub_field_name,
                            base_value=base_value,
                            val_change_range=perturb.perturbation_range,
                            allowed_real_range=real_range,
                            allowed_imag_range=imag_range,
                        )
        return val

    return _warn_perturbed_val_range


def _assert_min_freq(freqs, msg_start: str):
    """Check if all ``freqs`` are above the minimum frequency."""
    if np.min(freqs) < MIN_FREQUENCY:
        raise ValidationError(
            f"{msg_start} must be no lower than {MIN_FREQUENCY:.0e} Hz. "
            "Note that the unit of frequency is 'Hz'."
        )


def validate_freqs_min():
    """Validate lower bound for monitor, and mode solver frequencies."""

    @pydantic.validator("freqs", always=True, allow_reuse=True)
    def freqs_lower_bound(cls, val):
        """Raise validation error if any of ``freqs`` is lower than ``MIN_FREQUENCY``."""
        _assert_min_freq(val, msg_start=f"All of '{cls.__name__}.freqs'")
        return val

    return freqs_lower_bound


def validate_freqs_not_empty():
    """Validate that the array of frequencies is not empty."""

    @pydantic.validator("freqs", always=True, allow_reuse=True)
    def freqs_not_empty(cls, val):
        """Raise validation error if ``freqs`` is an empty Tuple."""
        if len(val) == 0:
            raise ValidationError(f"'{cls.__name__}.freqs' cannot be empty (size 0).")
        return val

    return freqs_not_empty


def validate_mode_plane_radius(mode_spec: ModeSpec, plane: Box, msg_prefix: str = ""):
    """Validate that the radius of a mode spec with a bend is not smaller than half the size of
    the plane along the radial direction."""

    if not mode_spec.bend_radius:
        return

    # radial axis is the plane axis that is not the bend axis
    _, plane_axs = plane.pop_axis([0, 1, 2], plane.size.index(0.0))
    radial_ax = plane_axs[(mode_spec.bend_axis + 1) % 2]

    if np.abs(mode_spec.bend_radius) < plane.size[radial_ax] / 2:
        raise ValueError(
            f"{msg_prefix} bend radius is smaller than half the mode plane size "
            "along the radial axis, which can produce wrong results."
        )


def _warn_unsupported_traced_argument(name: str):
    @pydantic.validator(name, always=True, allow_reuse=True)
    def _warn_traced_arg(cls, val, values):
        if isbox(val):
            log.warning(
                f"Field '{name}' of '{cls.__name__}' received an autograd tracer "
                f"(i.e., a value being tracked for automatic differentiation). "
                f"Automatic differentiation through this field is unsupported, "
                f"so the tracer has been converted to its static value. "
                f"If you want to avoid this warning, you manually unbox the value "
                f"using the 'autograd.tracer.getval' function before passing it to Tidy3D."
            )
            return get_static(val)
        return val

    return _warn_traced_arg
