"""Compatibility shim for :mod:`tidy3d._common.components.validators`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from tidy3d._common.components.validators import (
    MIN_FREQUENCY,
    FloatArray,
    _assert_min_freq,
    _warn_unsupported_traced_argument,
    validate_name_str,
    warn_if_dataset_none,
)
from tidy3d.components.data.data_array import DATA_ARRAY_MAP
from tidy3d.components.geometry.base import Box
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, Optional

    from pydantic import FieldValidationInfo

    from tidy3d import Simulation
    from tidy3d._common.components.validators import T
    from tidy3d.components.base_sim.simulation import AbstractSimulation
    from tidy3d.components.data.monitor_data import AbstractFieldData
    from tidy3d.components.types import FreqArray
    from tidy3d.plugins.smatrix import AbstractComponentModeler


def named_obj_descr(obj: Any, field_name: str, position_index: int) -> str:
    """Generate a string describing a named object which can be used in error messages."""
    descr = f"simulation.{field_name}[{position_index}] (no `name` was specified)"
    if hasattr(obj, "name") and obj.name:
        descr = f"'{obj.name}' (simulation.{field_name}[{position_index}])"
    return descr


def assert_line() -> Callable[[type, tuple[float, ...]], tuple[float, ...]]:
    """makes sure a field's ``size`` attribute has exactly 2 zeros"""

    @field_validator("size")
    @classmethod
    def is_line(cls: type, val: tuple[float, ...]) -> tuple[float, ...]:
        """Raise validation error if not 1 dimensional."""
        if val.count(0.0) != 2:
            raise ValidationError(f"'{cls.__name__}' object must be a line, given size={val}")
        return val

    return is_line


def assert_plane() -> Callable[[type, tuple[float, ...]], tuple[float, ...]]:
    """makes sure a field's ``size`` attribute has exactly 1 zero"""

    @field_validator("size")
    @classmethod
    def is_plane(cls: type, val: tuple[float, ...]) -> tuple[float, ...]:
        """Raise validation error if not planar."""
        if val.count(0.0) != 1:
            raise ValidationError(f"'{cls.__name__}' object must be planar, given size={val}")
        return val

    return is_plane


def assert_line_or_plane() -> Callable[[type, tuple[float, ...]], tuple[float, ...]]:
    """makes sure a field's ``size`` attribute has either 1 or 2 zeros"""

    @field_validator("size")
    @classmethod
    def is_line_or_plane(cls: type, val: tuple[float, ...]) -> tuple[float, ...]:
        """Raise validation error if not a line or plane."""
        if val.count(0.0) == 0 or val.count(0.0) == 3:
            raise ValidationError(
                f"'{cls.__name__}' object must be a line or a plane, given size={val}. "
            )
        return val

    return is_line_or_plane


def assert_volumetric() -> Callable[[type, tuple[float, ...]], tuple[float, ...]]:
    """makes sure a field's ``size`` attribute has no zero entry"""

    @field_validator("size")
    @classmethod
    def is_volumetric(cls: type, val: tuple[float, ...]) -> tuple[float, ...]:
        """Raise validation error if volume is 0."""
        if val.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' object must be volumetric, given size={val}. "
                "If intending to make a 2D simulation, please set the size of "
                f"'{cls.__name__}' along the zero dimension to a dummy non-zero value."
            )
        return val

    return is_volumetric


def validate_unique(
    *field_names: str,
) -> Callable[[type, Sequence[Any], FieldValidationInfo], Sequence[Any]]:
    """Make sure the given field has unique entries."""

    @field_validator(*field_names)
    @classmethod
    def field_has_unique_entries(
        cls: type, val: Sequence[Any], info: FieldValidationInfo
    ) -> Sequence[Any]:
        """Check if the field has unique entries."""
        if len(set(val)) != len(val):
            raise SetupError(f"Entries of '{info.field_name}' must be unique.")
        return val

    return field_has_unique_entries


def validate_mode_objects_symmetry(field_name: str) -> Callable[[T], T]:
    """If a Mode object, this checks that the object is fully in the main quadrant in the presence
    of symmetry along a given axis, or else centered on the symmetry center."""

    obj_type = "ModeSource" if field_name == "sources" else "ModeMonitor"

    @model_validator(mode="after")
    def check_symmetry(self: T) -> T:
        """check for intersection of each structure with simulation bounds."""
        val: Sequence[Any] = getattr(self, field_name)
        sim_center = self.center
        for position_index, geometric_object in enumerate(val):
            if geometric_object.type == obj_type:
                bounds_min, _ = geometric_object.bounds
                for dim, sym in enumerate(self.symmetry):
                    if (
                        sym != 0
                        and bounds_min[dim] < sim_center[dim]
                        and geometric_object.center[dim] != sim_center[dim]
                    ):
                        obj_descr = named_obj_descr(geometric_object, field_name, position_index)
                        raise SetupError(
                            f"{obj_type}: {obj_descr} in presence of symmetries must be in the main "
                            "quadrant, or centered on the symmetry axis."
                        )

        return self

    return check_symmetry


def assert_unique_names(
    *field_names: str,
) -> Callable[[type, Sequence[Any], FieldValidationInfo], Sequence[Any]]:
    """makes sure all elements of a field have unique .name values"""

    @field_validator(*field_names)
    @classmethod
    def field_has_unique_names(
        cls: type, val: Sequence[Any], info: FieldValidationInfo
    ) -> Sequence[Any]:
        """make sure each element of val has a unique name (if specified)."""
        field_names = [field.name for field in val if field.name]
        unique_names = set(field_names)
        if len(unique_names) != len(field_names):
            raise SetupError(f"'{info.field_name}' names are not unique, given {field_names}.")
        return val

    return field_has_unique_names


def assert_objects_in_sim_bounds(
    field_name: str, error: bool = True, strict_inequality: bool = False
) -> Callable[[AbstractSimulation], AbstractSimulation]:
    """Makes sure all objects in field are at least partially inside of simulation bounds."""

    @model_validator(mode="after")
    def objects_in_sim_bounds(self: AbstractSimulation) -> AbstractSimulation:
        """check for intersection of each structure with simulation bounds."""
        val: Sequence[Any] = getattr(self, field_name)
        sim_center = self.center
        sim_size = self.size
        sim_box = Box(size=sim_size, center=sim_center)

        # Do a strict check, unless simulation is 0D along a dimension
        strict_ineq: list[bool] = [size != 0 and strict_inequality for size in sim_size]

        with log as consolidated_logger:
            for position_index, geometric_object in enumerate(val):
                if not sim_box.intersects(geometric_object.geometry, strict_inequality=strict_ineq):
                    obj_descr = named_obj_descr(geometric_object, field_name, position_index)
                    message = f"{obj_descr} is outside of the simulation domain."
                    custom_loc = [field_name, position_index]
                    if error:
                        raise SetupError(message)
                    consolidated_logger.warning(message, custom_loc=custom_loc)

        return self

    return objects_in_sim_bounds


def assert_objects_contained_in_sim_bounds(
    field_name: str,
    error: bool = True,
    strict_inequality: bool = False,
    strict_for_zero_size_dim: bool = False,
) -> Callable[[Simulation], Simulation]:
    """Makes sure all objects in field are completely inside the simulation bounds."""

    @model_validator(mode="after")
    def objects_contained_in_sim_bounds(self: Simulation) -> Simulation:
        """check for containment of each structure with simulation bounds."""
        val: Sequence[Any] = getattr(self, field_name)
        sim_center = self.center
        sim_size = self.size
        sim_box = Box(size=sim_size, center=sim_center)

        # Do a strict check, unless simulation is 0D along a dimension
        strict_ineq: list[bool] = [size != 0 and strict_inequality for size in sim_size]
        with log as consolidated_logger:
            for position_index, geometric_object in enumerate(val):
                geo_strict_ineq = list(strict_ineq)
                # Optionally ensure that zero size dimensions are strictly contained
                if strict_for_zero_size_dim:
                    zero_dims = geometric_object.geometry.zero_dims
                    for zero_dim in zero_dims:
                        geo_strict_ineq[zero_dim] = True
                if not sim_box.contains(
                    geometric_object.geometry, strict_inequality=geo_strict_ineq
                ):
                    obj_descr = named_obj_descr(geometric_object, field_name, position_index)
                    message = f"{obj_descr} is not completely inside the simulation domain."
                    custom_loc = [field_name, position_index]
                    if error:
                        raise SetupError(message)
                    consolidated_logger.warning(message, custom_loc=custom_loc)

        return self

    return objects_contained_in_sim_bounds


def enforce_monitor_fields_present() -> Callable[[AbstractFieldData], AbstractFieldData]:
    """Make sure all of the fields in the monitor are present in the corresponding data."""

    @model_validator(mode="after")
    def _contains_fields(self: AbstractFieldData) -> AbstractFieldData:
        """Make sure the initially specified fields are here."""
        for field_name in self.monitor.fields:
            if getattr(self, field_name) is None:
                raise SetupError(f"missing field {field_name}")
        return self

    return _contains_fields


def required_if_symmetry_present(field_name: str) -> Callable[[T], T]:
    """Make a field required (not None) if any non-zero symmetry eigenvalue is present."""

    @model_validator(mode="after")
    def _make_required(self: T) -> T:
        """Ensure val is not None if the symmetry is non-zero along any dimension."""
        val = getattr(self, field_name)
        symmetry = self.symmetry
        if any(sym_val != 0 for sym_val in symmetry) and val is None:
            raise SetupError(f"'{field_name}' must be provided if symmetry present.")
        return self

    return _make_required


def warn_backward_waist_distance(field_name: str) -> Callable[[T], T]:
    """Warn if a backward-propagating beam uses a non-zero waist distance."""

    @model_validator(mode="after")
    def _warn_backward_nonzero(self: T) -> T:
        """Emit deprecation warning for backward propagation with non-zero waist."""
        direction = self.direction
        if direction != "-":
            return self
        waist_value = getattr(self, field_name)
        waist_array = np.atleast_1d(waist_value)
        if not np.all(np.isclose(waist_array, 0.0)):
            log.warning(
                f"Behavior of {self.__class__.__name__} with direction '-' and non-zero '{field_name}' will "
                "change in version 2.11 to be consistent with upcoming beam overlap monitors and "
                "ports. Currently, the waist distance is interpreted w.r.t. the directed "
                "propagation axis, so switching 'direction' also switches the position of the "
                "waist in the global reference frame. In the future, the waist position will be "
                "defined such that it is the same for backward- and forward-propagating beams.",
            )
        return self

    return _warn_backward_nonzero


def assert_single_freq_in_range(field_name: str) -> Callable[[T], T]:
    """Assert only one frequency supplied in source and it's in source time range."""

    @model_validator(mode="after")
    def _single_frequency_in_range(self: T) -> T:
        """Assert only one frequency supplied and it's in source time range."""
        val = getattr(self, field_name, None)
        if val is None:
            return self
        source_time = self.source_time
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
        return self

    return _single_frequency_in_range


def validate_parameter_perturbation(
    field_name: str,
    base_field_name: str,
    allowed_complex: bool = True,
) -> Callable[[type, Any, FieldValidationInfo], Any]:
    """Assert perturbations have a valid shape and data type."""

    @field_validator(field_name)
    @classmethod
    def _check_perturbed_val(cls: type, val: Any, info: FieldValidationInfo) -> Any:
        """Assert perturbations have a valid shape and data type."""

        if val is not None:
            if base_field_name not in info.data:
                return val

            # get base values
            base_values = info.data[base_field_name]

            # check that shapes of base parameter and perturbations coincide
            if np.shape(base_values) != np.shape(val):
                raise SetupError(
                    f"Shape of perturbations '{field_name}' ({np.shape(val)}) does not coincide"
                    f" with shape of base parameter '{base_field_name}' ({np.shape(base_values)})."
                )

            for perturb_tuple in np.atleast_1d(val):
                for perturb in np.atleast_1d(perturb_tuple):
                    if perturb is not None:
                        # check real/complex type
                        if perturb.is_complex and not allowed_complex:
                            raise SetupError(
                                f"Perturbation of '{base_field_name}' cannot be complex."
                            )

        return val

    return _check_perturbed_val


def validate_freqs_min() -> Callable[[type, FreqArray], FreqArray]:
    """Validate lower bound for monitor, and mode solver frequencies."""

    @field_validator("freqs")
    @classmethod
    def freqs_lower_bound(cls: type, val: FreqArray) -> FreqArray:
        """Raise validation error if any of ``freqs`` is lower than ``MIN_FREQUENCY``."""
        _assert_min_freq(val, msg_start=f"All of '{cls.__name__}.freqs'")
        return val

    return freqs_lower_bound


def validate_freqs_not_empty() -> Callable[[type, FreqArray], FreqArray]:
    """Validate that the array of frequencies is not empty."""

    @field_validator("freqs")
    @classmethod
    def freqs_not_empty(cls: type, val: FreqArray) -> FreqArray:
        """Raise validation error if ``freqs`` is an empty Tuple."""
        if len(val) == 0:
            raise ValidationError(f"'{cls.__name__}.freqs' cannot be empty (size 0).")
        return val

    return freqs_not_empty


def validate_freqs_unique() -> Callable[[AbstractComponentModeler, FreqArray], FreqArray]:
    """Validate that the array of frequencies does not have duplicate entries."""

    @field_validator("freqs")
    @classmethod
    def freqs_unique(cls: AbstractComponentModeler, val: FreqArray) -> FreqArray:
        """Raise validation error if ``freqs`` has duplicate entries."""
        if len(set(val)) != len(val):
            raise ValidationError(f"'{cls.__name__}.freqs' must not contain duplicate entries.")
        return val

    return freqs_unique
