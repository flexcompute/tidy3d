"""Defines various validation functions that get used to ensure inputs are legit"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator, model_validator

from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .autograd.utils import get_static, hasbox
from .base import DATA_ARRAY_MAP
from .geometry.base import Box

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import FieldValidationInfo

    from tidy3d import Simulation
    from tidy3d.components.base_sim.simulation import AbstractSimulation
    from tidy3d.components.data.monitor_data import AbstractFieldData
    from tidy3d.components.types import FreqArray
    from tidy3d.plugins.smatrix import AbstractComponentModeler


T = TypeVar("T")

"""Explanation of Pydantic validators (v2).

    Validators are class methods that validate and coerce model inputs. This module defines
    reusable validator factories that are shared across tidy3d components.

    In Pydantic v2 we use:
    - ``@field_validator("field_name")`` for field-local checks/coercions. It can access
      already-validated fields via ``info.data``, but ``info.data`` only contains fields
      validated earlier, so avoid order-dependent cross-field logic.
    - ``@model_validator(mode="after")`` for cross-field constraints that need the full model.

    To attach a validator from this file to a Pydantic model, assign the factory result in the
    class body, e.g. ``_plane_validator = assert_plane()``. Avoid reusing the same attribute
    name for multiple validators, or earlier validators may be overwritten.

    For the ``Medium`` and ``Simulation`` class families, prefer explicit orchestration of
    cross-field checks in ``_run_after_validators()`` (with a short docstring) and call validator
    factories via ``call_wrapped_validator(...)`` to keep ordering explicit.

    For more details: `Pydantic validators <https://docs.pydantic.dev/latest/concepts/validators/>`_
"""

# Lowest frequency supported (Hz)
MIN_FREQUENCY = 1e5

FloatArray = Sequence[float] | NDArray


def wrapped_angle_distance(angle: float, target: float, period: float) -> float:
    """Return the shortest distance between two angles on a periodic domain."""
    return abs((angle - target + period / 2) % period - period / 2)


def is_close_to_glancing_angle(angle_theta: float, cutoff: float) -> bool:
    """Check whether ``angle_theta`` is close to any odd multiple of ``π/2``."""
    return wrapped_angle_distance(angle_theta, np.pi / 2, np.pi) < cutoff


def named_obj_descr(obj: Any, field_name: str, position_index: int) -> str:
    """Generate a string describing a named object which can be used in error messages."""
    descr = f"simulation.{field_name}[{position_index}] (no `name` was specified)"
    if hasattr(obj, "name") and obj.name:
        descr = f"'{obj.name}' (simulation.{field_name}[{position_index}])"
    return descr


def points_outside_bounds(
    points: NDArray, bounds: NDArray, strict_inequality: Sequence[bool]
) -> NDArray:
    """Return a mask for points outside axis-aligned bounds."""
    points = np.asarray(points, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    outside_lower = np.zeros(points.shape[0], dtype=bool)
    outside_upper = np.zeros(points.shape[0], dtype=bool)
    for axis, strict in enumerate(strict_inequality):
        if strict:
            outside_lower |= points[:, axis] <= bounds[0, axis]
            outside_upper |= points[:, axis] >= bounds[1, axis]
        else:
            outside_lower |= points[:, axis] < bounds[0, axis]
            outside_upper |= points[:, axis] > bounds[1, axis]

    return outside_lower | outside_upper


def call_wrapped_validator(
    factory: Callable[..., Any], instance: Any, *args: Any, **kwargs: Any
) -> Any:
    """Call the wrapped pydantic validator produced by a factory."""
    return factory(*args, **kwargs).wrapped(instance)


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


# FIXME: this validator doesn't do anything
def validate_name_str() -> Callable[[type, str | None], str | None]:
    """make sure the name does not include [, ] (used for default names)"""

    @field_validator("name")
    @classmethod
    def field_has_unique_names(cls: type, val: str | None) -> str | None:
        """raise exception if '[' or ']' in name"""
        # if val and ('[' in val or ']' in val):
        #     raise SetupError(f"'[' or ']' not allowed in name: {val} (used for defaults)")
        return val

    return field_has_unique_names


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
    """If a Mode-like object (ModeSource / ModeMonitor / ModeTimeMonitor),
    check that the object is fully in the main quadrant in the presence
    of symmetry along a given axis, or else centered on the symmetry
    center. ModeTimeMonitor shares the ModeMonitor mode-solver pipeline
    and inherits the same restriction."""

    if field_name == "sources":
        obj_types: tuple[str, ...] = ("ModeSource",)
    else:
        obj_types = ("ModeMonitor", "ModeTimeMonitor")

    @model_validator(mode="after")
    def check_symmetry(self: T) -> T:
        """check for intersection of each structure with simulation bounds."""
        val: Sequence[Any] = getattr(self, field_name)
        sim_center = self.center
        for position_index, geometric_object in enumerate(val):
            if geometric_object.type in obj_types:
                bounds_min, _ = geometric_object.bounds
                for dim, sym in enumerate(self.symmetry):
                    if (
                        sym != 0
                        and bounds_min[dim] < sim_center[dim]
                        and geometric_object.center[dim] != sim_center[dim]
                    ):
                        obj_descr = named_obj_descr(geometric_object, field_name, position_index)
                        self._raise_validation_error_at_loc(
                            f"{geometric_object.type}: {obj_descr} in presence of symmetries must "
                            "be in the main quadrant, or centered on the symmetry axis.",
                            field_name,
                            position_index,
                        )

        return self

    return check_symmetry


def validate_field_projection_monitors_2d(
    monitors: Sequence[Any] | None,
    sim_size: tuple[float, float, float],
    raise_error: Callable[[str, int], None] | None = None,
) -> None:
    """Validate lower-dimensional field projection monitor settings."""

    if not monitors:
        return

    non_zero_dims = sum(1 for size in sim_size if size != 0)
    if non_zero_dims == 3:
        return

    from .monitor import (
        AbstractFieldProjectionMonitor,
        FieldProjectionAngleMonitor,
        FieldProjectionCartesianMonitor,
        FieldProjectionKSpaceMonitor,
    )

    if sim_size[0] == 0:
        plane = "y-z"
    elif sim_size[1] == 0:
        plane = "x-z"
    else:
        plane = "x-y"

    for monitor_ind, monitor in enumerate(monitors):
        if not isinstance(monitor, AbstractFieldProjectionMonitor):
            continue

        if non_zero_dims == 1:
            message = f"Monitor '{monitor.name}' is not supported in 1D simulations."
            if raise_error is not None:
                raise_error(message, monitor_ind)
            raise SetupError(message)

        if not monitor.far_field_approx:
            message = (
                f"Exact far-field projection for 2D simulations is not yet available for Monitor '{monitor.name}'. "
                "Currently, only 'far_field_approx = True' is supported."
            )
            if raise_error is not None:
                raise_error(message, monitor_ind)
            raise SetupError(message)

        if isinstance(monitor, FieldProjectionAngleMonitor):
            config = {
                "y-z": {"valid_value": [np.pi / 2, 3 * np.pi / 2], "coord": "phi"},
                "x-z": {"valid_value": [0, np.pi], "coord": "phi"},
                "x-y": {"valid_value": [np.pi / 2], "coord": "theta"},
            }[plane]

            coord = getattr(monitor, config["coord"])
            if not all(value in config["valid_value"] for value in coord):
                replacements = {
                    np.pi: "np.pi",
                    np.pi / 2: "np.pi/2",
                    3 * np.pi / 2: "3*np.pi/2",
                    0: "0",
                }
                valid_values_str = ", ".join(replacements.get(val) for val in config["valid_value"])
                message = (
                    f"For a 2D simulation in the {plane} plane, the observation "
                    f"angle '{config['coord']}' of monitor "
                    f"'{monitor.name}' should be set to "
                    f"'{valid_values_str}'"
                )
                if raise_error is not None:
                    raise_error(message, monitor_ind)
                raise SetupError(message)

            continue

        if isinstance(monitor, FieldProjectionCartesianMonitor):
            config = {
                "y-z": {"valid_proj_axes": [1, 2], "coord": ["x", "x"]},
                "x-z": {"valid_proj_axes": [0, 2], "coord": ["x", "y"]},
                "x-y": {"valid_proj_axes": [0, 1], "coord": ["y", "y"]},
            }[plane]
        elif isinstance(monitor, FieldProjectionKSpaceMonitor):
            config = {
                "y-z": {"valid_proj_axes": [1, 2], "coord": ["ux", "ux"]},
                "x-z": {"valid_proj_axes": [0, 2], "coord": ["ux", "uy"]},
                "x-y": {"valid_proj_axes": [0, 1], "coord": ["uy", "uy"]},
            }[plane]
        else:
            continue

        valid_proj_axes = config["valid_proj_axes"]
        invalid_proj_axis = [i for i in range(3) if i not in valid_proj_axes]

        if monitor.proj_axis in invalid_proj_axis:
            message = (
                f"For a 2D simulation in the {plane} plane, the 'proj_axis' of "
                f"monitor '{monitor.name}' should be set to one of {valid_proj_axes}."
            )
            if raise_error is not None:
                raise_error(message, monitor_ind)
            raise SetupError(message)

        for idx, axis in enumerate(valid_proj_axes):
            coord = getattr(monitor, config["coord"][idx])
            if monitor.proj_axis == axis and not all(value in [0] for value in coord):
                message = (
                    f"For a 2D simulation in the {plane} plane with "
                    f"'proj_axis = {monitor.proj_axis}', '{config['coord'][idx]}' of monitor "
                    f"'{monitor.name}' should be set to '[0]'."
                )
                if raise_error is not None:
                    raise_error(message, monitor_ind)
                raise SetupError(message)


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
                if getattr(geometric_object, "_skip_sim_bounds_intersection_validation", False):
                    continue
                if not sim_box.intersects(geometric_object.geometry, strict_inequality=strict_ineq):
                    obj_descr = named_obj_descr(geometric_object, field_name, position_index)
                    message = f"{obj_descr} is outside of the simulation domain."
                    custom_loc = [field_name, position_index]
                    if error:
                        self._raise_validation_error_at_loc(message, *custom_loc)
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
                        self._raise_validation_error_at_loc(message, *custom_loc)
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
                self._raise_validation_error_at_loc(f"missing field {field_name}", field_name)
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
            self._raise_validation_error_at_loc(
                f"'{field_name}' must be provided if symmetry present.", field_name
            )
        return self

    return _make_required


def warn_if_dataset_none(
    field_name: str,
) -> Callable[[type, dict[str, Any] | None], dict[str, Any] | None]:
    """Warn if a Dataset field has None in its dictionary."""

    @field_validator(field_name, mode="before")
    @classmethod
    def _warn_if_none(cls: type, val: dict[str, Any] | None) -> dict[str, Any] | None:
        """Warn if the DataArrays fail to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning(f"Loading {field_name} without data.", custom_loc=[field_name])
                return None
        return val

    return _warn_if_none


def warn_backward_waist_distance(field_name: str) -> Callable[[T], T]:
    """Warn about changed waist distance behavior for backward-propagating beams."""

    @model_validator(mode="after")
    def _warn_backward_nonzero(self: T) -> T:
        """Emit warning about changed waist distance interpretation."""
        direction = self.direction
        if direction != "-":
            return self
        waist_value = getattr(self, field_name)
        waist_array = np.atleast_1d(waist_value)
        if not np.all(np.isclose(waist_array, 0.0)):
            log.warning(
                f"Starting in version 2.11, the behavior of {self.__class__.__name__} with direction '-' "
                f"and non-zero '{field_name}' has changed. The waist position is now defined "
                "consistently for both forward- and backward-propagating beams: a positive "
                f"'{field_name}' always places the beam waist behind the source/monitor plane "
                "(toward the negative normal axis). This ensures reciprocity between Gaussian "
                "sources and overlap monitors used for port-based S-matrix calculations. "
                "If your simulation relied on the previous behavior (where the waist position "
                "flipped with direction), you may need to adjust your waist distance values.",
                log_once=True,
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
                self._raise_validation_error_at_loc(
                    f"'{field_name}.{name}' must have a single frequency, "
                    f"contains {len(freqs)} frequencies.",
                    field_name,
                    name,
                )
            freq = float(freqs[0])
            if (freq < fmin) or (freq > fmax):
                self._raise_validation_error_at_loc(
                    f"'{field_name}.{name}' contains frequency: {freq:.2e} Hz, which is outside "
                    f"of the 'source_time' frequency range [{fmin:.2e}-{fmax:.2e}] Hz.",
                    field_name,
                    name,
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


def _assert_min_freq(freqs: FloatArray, msg_start: str) -> None:
    """Check if all ``freqs`` are above the minimum frequency."""
    if np.min(freqs) < MIN_FREQUENCY:
        raise ValidationError(
            f"{msg_start} must be no lower than {MIN_FREQUENCY:.0e} Hz. "
            "Note that the unit of frequency is 'Hz'."
        )


def validate_colocated_integration() -> Callable[[type, bool, FieldValidationInfo], bool]:
    """Ensure use_colocated_integration=False is only used with colocate=False."""

    @field_validator("use_colocated_integration")
    @classmethod
    def _check_colocated_integration(cls: type, val: bool, info: FieldValidationInfo) -> bool:
        colocate = info.data.get("colocate", True)
        if colocate and not val:
            raise ValidationError(
                "'use_colocated_integration' can only be set to 'False' when 'colocate' is 'False'."
            )
        return val

    return _check_colocated_integration


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


def validate_freqs_num_not_too_many(
    warn_num_freqs: int,
) -> Callable[[type, FreqArray, FieldValidationInfo], FreqArray]:
    """Warn if the number of ``freqs`` exceeds ``warn_num_freqs``.

    When the instance has a non-empty ``name`` available on ``info.data``, it
    is used in the warning message; otherwise the class name is used.
    """

    @field_validator("freqs")
    @classmethod
    def _warn_num_freqs(cls: type, val: FreqArray, info: FieldValidationInfo) -> FreqArray:
        """Warn if number of frequencies is too large."""
        if len(val) > warn_num_freqs:
            # Prefer the instance's ``name`` so the warning identifies the specific
            # offending instance; fall back to the class name when ``name`` is
            # absent or empty.
            name = info.data.get("name")
            identifier = f"'{name}'" if name else f"'{cls.__name__}'"
            log.warning(
                f"A large number ({len(val)}) of frequencies detected in {identifier}. "
                "This can lead to solver slow-down and increased cost. "
                "Consider decreasing the number of frequencies. This may become a "
                "hard limit in future Tidy3D versions.",
                custom_loc=["freqs"],
            )
        return val

    return _warn_num_freqs


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
