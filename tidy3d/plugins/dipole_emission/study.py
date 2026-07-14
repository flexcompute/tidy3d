"""Study objects for dipole emission analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, PositiveFloat, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import PointDataArray, SphericalAngleDataArray
from tidy3d.components.data.point_cloud import canonicalize_point_cloud_points
from tidy3d.components.geometry.base import Box
from tidy3d.components.monitor import DIPOLE_EMISSION_DIPOLE_AXES, DipoleEmissionMonitor
from tidy3d.components.simulation import (
    MAX_SIMULATION_DATA_SIZE_GB,
    WARN_MONITOR_DATA_SIZE_GB,
    Simulation,
)
from tidy3d.components.source.field import TFSF, FixedAngleSpec
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, Axis
from tidy3d.components.types.time import SourceTimeType
from tidy3d.components.validators import points_outside_bounds
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import check_tidy3d_extras_licensed_feature
from tidy3d.plugins.dipole_emission.constants import (
    EMISSION_MONITOR_NAME,
    MESH_OVERRIDE_NAME,
    RESERVED_NAME_PREFIX,
    SOURCE_NAME_PREFIX,
)
from tidy3d.plugins.dipole_emission.data_array import (
    DipoleEmissionStudyDataArray,
    DipoleEmissionStudyPositionDataArray,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Direction = Literal["+", "-"]
Polarization = Literal["p", "s"]
FEATURE_NAME = "dipole_emission"


class EmissionAnalysisRegion(Box):
    """Region over which dipole emission is evaluated.

    The region should contain the sampled emitters and the part of the optical
    structure whose scattering response contributes to the collected radiation.
    The ``normal_axis`` and ``direction`` fields select the collection half-space
    used to interpret far-field angles. For example, ``normal_axis=2`` and
    ``direction="+"`` describes radiation collected toward ``+z``. Generated
    simulations require that collection side to be represented by a single
    background medium with real, nondispersive refractive index over the
    requested frequencies. The ``dl`` field controls the spatial resolution in
    the plane transverse to the collection direction.
    """

    normal_axis: Axis = Field(
        title="Normal Axis",
        description="Coordinate axis normal to the selected collection side.",
    )

    direction: Direction = Field(
        title="Collection Direction",
        description="Side of the structure, along ``normal_axis``, where emission is collected.",
    )

    dl: PositiveFloat | tuple[PositiveFloat, PositiveFloat] = Field(
        title="Transverse Grid Size",
        description=(
            "Spatial resolution along axes transverse to ``normal_axis``. A scalar applies "
            "to both transverse axes; a pair sets the two transverse resolutions independently."
        ),
        json_schema_extra={"units": MICROMETER},
    )

    @model_validator(mode="after")
    def _validate_box_size(self) -> EmissionAnalysisRegion:
        """Require a finite, positive 3D analysis box."""
        size = np.asarray(self.size, dtype=float)
        if size.shape != (3,) or not np.all(np.isfinite(size)):
            raise ValidationError("'analysis_region.size' must contain three finite values.")
        if np.any(size <= 0):
            raise ValidationError("'analysis_region.size' must be positive along all axes.")
        return self

    @property
    def tangential_axes(self) -> tuple[int, int]:
        """Axes transverse to ``normal_axis``."""
        return tuple(axis for axis in range(3) if axis != self.normal_axis)

    @cached_property
    def geometry(self) -> Box:
        """:class:`~tidy3d.Box` representation of the analysis region."""
        return Box(center=self.center, size=self.size)

    def mesh_override(self, name: str) -> MeshOverrideStructure:
        """Create the transverse-resolution override for this region."""
        dl_values = [None, None, None]
        transverse_dl = self.dl if isinstance(self.dl, tuple) else (self.dl, self.dl)
        for axis, dl in zip(self.tangential_axes, transverse_dl):
            dl_values[axis] = dl
        return MeshOverrideStructure(
            geometry=self.geometry,
            dl=tuple(dl_values),
            enforce=True,
            name=name,
        )


class DipoleEmissionStudy(Tidy3dBaseModel):
    """Dipole-emission sweep definition.

    A study evaluates radiative emission from classical electric dipoles in a
    passive optical environment. It reports angular radiation intensity per
    squared dipole moment for x-, y-, and z-oriented dipoles after summing over
    the sampled dipole positions using ``position_weights``. Optional
    position-resolved samples may be retained for selected position indexes. The
    result describes radiation into the selected collection half-space;
    non-radiative loss, trapped guided modes, and total Purcell-factor
    accounting are not included.
    """

    base_sim: Simulation = Field(
        title="Base Simulation",
        description=(
            "Source-free simulation describing the passive optical structure. Monitors in "
            "this simulation are available only through optional diagnostic batch data."
        ),
    )

    analysis_region: EmissionAnalysisRegion = Field(
        title="Analysis Region",
        description=(
            "Region containing the sampled dipoles and defining the far-field collection side."
        ),
    )

    positions: PointDataArray = Field(
        title="Dipole Positions",
        description="Sampled dipole locations as a ``PointDataArray`` with dims ``('index', 'axis')``.",
    )

    position_weights: ArrayFloat1D | ArrayFloat2D | None = Field(
        None,
        title="Position Weights",
        description=(
            "Weights used when summing radiation intensity over ``positions``. Provide one "
            "weight per position, or one weight per position and Cartesian dipole axis. If "
            "omitted, all sampled positions have unit weight."
        ),
    )

    store_position_indexes: tuple[int, ...] = Field(
        (),
        title="Stored Position Indexes",
        description=(
            "Zero-based indexes into ``positions`` for individual position-resolved radiation "
            "intensity samples to retain. The default stores only the summed result."
        ),
    )

    source_time: SourceTimeType = Field(
        title="Source Time",
        description=(
            "Spectral envelope for the emission calculation. The envelope is included in "
            "``radiation_intensity`` and divided out by the "
            "``radiation_intensity_transfer(...)`` method."
        ),
    )

    freqs: ArrayFloat1D = Field(
        title="Frequencies",
        description="Frequencies at which the emission response is evaluated.",
    )

    angles: SphericalAngleDataArray = Field(
        title="Emission Angles",
        description=(
            "Far-field observation directions as ``(theta, phi)`` pairs in radians, with "
            "``theta`` measured away from the selected collection normal and restricted to "
            "``abs(theta) < pi/2``."
        ),
    )

    polarizations: tuple[Polarization, ...] = Field(
        default=("p", "s"),
        title="Emission Polarizations",
        description=(
            "Far-field polarization components to evaluate at each observation angle. The "
            "``p`` component lies in the plane formed by the observation direction and "
            "collection normal; ``s`` is orthogonal to that plane."
        ),
    )

    @field_validator("positions")
    @classmethod
    def _validate_positions(cls, val: PointDataArray) -> PointDataArray:
        """Validate sampled dipole positions."""
        try:
            return canonicalize_point_cloud_points(
                val,
                empty_error="'positions' must contain at least one point.",
                require_real=True,
                require_finite=True,
                cast_to_float=True,
                coordinate_name="'positions'",
                preserve_index=True,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from None

    @field_validator("store_position_indexes")
    @classmethod
    def _validate_store_position_indexes(cls, val: Sequence[int]) -> tuple[int, ...]:
        """Validate optional stored position indexes."""
        indexes = tuple(index for index in val)
        if len(set(indexes)) != len(indexes):
            raise ValidationError("'store_position_indexes' must not contain duplicates.")
        if any(index < 0 for index in indexes):
            raise ValidationError("'store_position_indexes' entries must be nonnegative.")
        return indexes

    @field_validator("freqs")
    @classmethod
    def _validate_freqs(cls, val: ArrayFloat1D) -> ArrayFloat1D:
        """Validate positive output frequencies."""
        freqs = np.asarray(val, dtype=float)
        if freqs.ndim != 1 or len(freqs) == 0:
            raise ValidationError("'freqs' must be a non-empty 1D array.")
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
            raise ValidationError("'freqs' must contain only finite positive values.")
        if len(np.unique(freqs)) != len(freqs):
            raise ValidationError("'freqs' must be unique.")
        return freqs

    @field_validator("polarizations")
    @classmethod
    def _validate_polarizations(cls, val: Sequence[Polarization]) -> tuple[Polarization, ...]:
        """Validate requested emission-polarization labels."""
        if len(val) == 0:
            raise ValidationError("'polarizations' must contain at least one entry.")
        if len(set(val)) != len(val):
            raise ValidationError("'polarizations' must not contain duplicates.")
        return tuple(val)

    @field_validator("angles")
    @classmethod
    def _validate_angles(cls, val: SphericalAngleDataArray) -> SphericalAngleDataArray:
        """Validate far-field observation angles."""
        values = np.asarray(val.values)
        if np.iscomplexobj(values):
            raise ValidationError("'angles' must contain real values.")
        try:
            data = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValidationError("'angles' must contain real values.") from None
        if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
            raise ValidationError("'angles' must have shape '(N, 2)' with columns '(theta, phi)'.")
        if not np.all(np.isfinite(data)):
            raise ValidationError("'angles' must contain only finite values.")
        if np.any(np.abs(data[:, 0]) >= np.pi / 2):
            raise ValidationError("'angles' theta values must satisfy abs(theta) < pi/2.")
        return val

    @model_validator(mode="after")
    def _validate_study(self) -> DipoleEmissionStudy:
        """Validate study-level invariants that span fields."""
        if len(self.base_sim.sources) != 0:
            self._raise_validation_error_at_loc(
                "'base_sim' must not contain sources.", "base_sim", "sources"
            )
        if self.base_sim.symmetry != (0, 0, 0):
            self._raise_validation_error_at_loc(
                "'base_sim.symmetry' must be '(0, 0, 0)' for dipole emission studies.",
                "base_sim",
                "symmetry",
            )
        for monitor_index, monitor in enumerate(self.base_sim.monitors):
            if monitor.name.startswith(RESERVED_NAME_PREFIX):
                self._raise_validation_error_at_loc(
                    f"Monitor name '{monitor.name}' conflicts with the reserved "
                    f"dipole-emission prefix '{RESERVED_NAME_PREFIX}'.",
                    "base_sim",
                    "monitors",
                    monitor_index,
                    "name",
                )

        points = np.asarray(self.positions.values, dtype=float)
        sim_strict_axes = np.asarray([size != 0 for size in self.base_sim.size], dtype=bool)
        if np.any(points_outside_bounds(points, self.base_sim.bounds, sim_strict_axes)):
            self._raise_validation_error_at_loc(
                "'positions' must lie inside the simulation domain.", "positions"
            )
        region_strict_axes = np.asarray(
            [size != 0 for size in self.analysis_region.size], dtype=bool
        )
        if np.any(points_outside_bounds(points, self.analysis_region.bounds, region_strict_axes)):
            self._raise_validation_error_at_loc(
                "'positions' must lie inside 'analysis_region'.", "positions"
            )
        self._call_with_validation_loc(("source_time",), self._validate_source_spectrum)
        self._call_with_validation_loc(("position_weights",), self._validate_position_weights)
        self._call_with_validation_loc(
            ("store_position_indexes",), self._validate_stored_position_index_range
        )
        self._validate_study_batch_data_size()
        return self

    def _validate_position_weights(self) -> None:
        """Validate position weights against the sampled positions."""
        if self.position_weights is None:
            return
        weights = np.asarray(self.position_weights, dtype=float)
        point_shape = np.asarray(self.positions.values).shape
        if weights.shape not in ((self.positions.sizes["index"],), point_shape):
            raise ValidationError(
                "'position_weights' must have one value per position or match the shape of "
                "'positions'."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValidationError("'position_weights' must contain finite nonnegative values.")
        if not np.any(weights > 0):
            raise ValidationError("'position_weights' must not be all zero (no emitters).")

    def _validate_source_spectrum(self) -> None:
        """Validate source spectrum required by transfer normalization."""
        _ = self.pulse_spectrum_abs2

    def _validate_stored_position_index_range(self) -> None:
        """Validate stored position indexes against the sampled positions."""
        if not self.store_position_indexes:
            return
        num_positions = self.positions.sizes["index"]
        if max(self.store_position_indexes) >= num_positions:
            raise ValidationError(
                "'store_position_indexes' entries must be valid position indexes."
            )

    def _validate_study_batch_data_size(self) -> None:
        """Warn or error when estimated study batch data is too large to download."""
        num_tasks = len(self._expected_task_names())
        precision_factor = 2 if self.base_sim.precision == "double" else 1
        intrinsic_size_gb = (
            self._emission_monitor().storage_size(num_cells=0, tmesh=[])
            * precision_factor
            * num_tasks
            / 1e9
        )
        diagnostic_size_gb = 0.0
        with log as consolidated_logger:
            if intrinsic_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                consolidated_logger.warning(
                    f"Dipole-emission results are estimated to download "
                    f"{intrinsic_size_gb:1.2f}GB across the study batch. Consider reducing "
                    "'store_position_indexes', 'angles', 'polarizations', or 'freqs'."
                )
            for monitor_index, (monitor_name, monitor_size) in enumerate(
                self.base_sim.monitors_data_size.items()
            ):
                repeated_monitor_size_gb = monitor_size * num_tasks / 1e9
                if repeated_monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Diagnostic monitor '{monitor_name}' is estimated to download "
                        f"{repeated_monitor_size_gb:1.2f}GB across the dipole-emission batch. "
                        "Consider removing it from 'base_sim' or reducing its stored data.",
                        custom_loc=["base_sim", "monitors", monitor_index],
                    )
                diagnostic_size_gb += repeated_monitor_size_gb

        total_size_gb = intrinsic_size_gb + diagnostic_size_gb
        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            error_loc = (
                ("base_sim", "monitors")
                if diagnostic_size_gb > intrinsic_size_gb
                else ("store_position_indexes",)
            )
            self._raise_validation_error_at_loc(
                f"Dipole-emission study data is estimated to download {total_size_gb:.2f}GB "
                f"across the study batch, including {intrinsic_size_gb:.2f}GB from "
                f"dipole-emission results and {diagnostic_size_gb:.2f}GB from diagnostic "
                f"monitors; a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB is allowed. "
                "Consider reducing 'store_position_indexes', 'angles', 'polarizations', "
                "or 'freqs', or removing diagnostic monitors from 'base_sim'.",
                *error_loc,
            )

    def _check_license(self) -> None:
        """Check that the local dipole-emission feature is licensed."""
        check_tidy3d_extras_licensed_feature(FEATURE_NAME)

    @property
    def pulse_spectrum_abs2(self) -> ArrayFloat1D:
        """Squared source spectrum magnitude sampled at ``freqs``."""
        amps = np.asarray([self.source_time.amp_freq(freq) for freq in self.freqs], dtype=complex)
        values = np.abs(amps) ** 2
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise SetupError(
                "Dipole emission requires a nonzero finite source spectrum at every output frequency."
            )
        return values

    @property
    def position_weights_array(self) -> ArrayFloat1D | ArrayFloat2D:
        """Weights used to sum radiation intensity over sampled positions and axes."""
        if self.position_weights is None:
            return np.ones(self.positions.sizes["index"], dtype=float)
        return np.asarray(self.position_weights, dtype=float)

    @staticmethod
    def _opposite_direction(direction: Direction) -> Direction:
        """Reciprocal-probe direction into the analysis region."""
        return "-" if direction == "+" else "+"

    @staticmethod
    def _task_name(angle_index: int, polarization: Polarization) -> str:
        """Generated batch task name for one reciprocity probe."""
        return f"{RESERVED_NAME_PREFIX}angle{angle_index:03d}_{polarization}"

    @staticmethod
    def _source_name(angle_index: int, polarization: Polarization) -> str:
        """Study source name for one angle and polarization."""
        return f"{SOURCE_NAME_PREFIX}_angle{angle_index:03d}_{polarization}"

    def _angle_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored ``(theta, phi)`` arrays in study order."""
        return (
            np.asarray(self.angles.sel(spherical_coordinate="theta").values, dtype=float),
            np.asarray(self.angles.sel(spherical_coordinate="phi").values, dtype=float),
        )

    def _expected_task_names(self) -> tuple[tuple[str, int, Polarization], ...]:
        """Expected task names in reduction stacking order."""
        return tuple(
            (self._task_name(angle_index, polarization), angle_index, polarization)
            for angle_index in range(self.angles.sizes["index"])
            for polarization in self.polarizations
        )

    def _source(
        self, angle_index: int, theta: float, phi: float, polarization: Polarization
    ) -> TFSF:
        """Build the reciprocity probe source for one angle and polarization."""
        return TFSF(
            center=self.analysis_region.center,
            size=self.analysis_region.size,
            source_time=self.source_time,
            direction=self._opposite_direction(self.analysis_region.direction),
            injection_axis=self.analysis_region.normal_axis,
            angle_theta=theta,
            angle_phi=phi,
            pol_angle=0.0 if polarization == "p" else np.pi / 2,
            angular_spec=FixedAngleSpec(),
            name=self._source_name(angle_index, polarization),
        )

    def _emission_monitor(self) -> DipoleEmissionMonitor:
        """Build the monitor used to store reduced emission data."""
        return DipoleEmissionMonitor(
            points=self.positions,
            freqs=self.freqs,
            position_weights=self.position_weights_array,
            store_position_indexes=self.store_position_indexes,
            name=EMISSION_MONITOR_NAME,
        )

    def _result_coords(self) -> dict[str, Any]:
        """Coordinates for compact study-level emission arrays.

        The ``angle`` dimension carries only integer indices; the corresponding
        ``(theta, phi)`` directions live solely in ``self.angles`` to avoid
        duplicating angle metadata that the HDF5 round trip would not preserve.
        """
        return {
            "dipole_axis": list(DIPOLE_EMISSION_DIPOLE_AXES),
            "polarization": list(self.polarizations),
            "angle": np.arange(self.angles.sizes["index"]),
            "f": self.freqs,
        }

    def _position_result_coords(self) -> dict[str, Any]:
        """Coordinates for selected position-resolved study-level emission arrays."""
        coords = self._result_coords()
        position_index_values = np.asarray(self.positions.coords["index"].values)
        coords["index"] = position_index_values[list(self.store_position_indexes)]
        return coords

    def _monitor_data_from_batch_data(self, batch_data: Any) -> dict[tuple[int, Polarization], Any]:
        """Extract reduced dipole-emission monitor data from user-managed ``BatchData``."""
        monitor_data_by_task: dict[tuple[int, Polarization], Any] = {}
        for task_name, angle_index, polarization in self._expected_task_names():
            try:
                sim_data = batch_data[task_name]
            except KeyError as exc:
                raise SetupError(
                    f"Dipole emission batch data is missing task '{task_name}' from "
                    "to_simulations(...). "
                    f"Original error: {exc}"
                ) from exc
            except (TypeError, AttributeError) as exc:
                raise SetupError(
                    "DipoleEmissionStudy.compose(...) expects batch data that supports "
                    f"task-name lookup. Original error: {exc}"
                ) from exc

            self._validate_task_simulation_source(
                sim_data=sim_data,
                task_name=task_name,
                angle_index=angle_index,
                polarization=polarization,
            )

            try:
                monitor_data = sim_data[EMISSION_MONITOR_NAME]
            except KeyError as exc:
                raise SetupError(
                    f"Dipole emission task '{task_name}' is missing monitor "
                    f"'{EMISSION_MONITOR_NAME}'. Original error: {exc}"
                ) from exc
            except (TypeError, AttributeError) as exc:
                raise SetupError(
                    f"Dipole emission task '{task_name}' does not contain ordinary "
                    f"SimulationData monitor lookup. Original error: {exc}"
                ) from exc

            monitor_data_by_task[(angle_index, polarization)] = monitor_data

        return monitor_data_by_task

    def _validate_task_simulation_source(
        self,
        sim_data: Any,
        task_name: str,
        angle_index: int,
        polarization: Polarization,
    ) -> None:
        """Validate generated source provenance when batch data exposes a simulation."""
        simulation = getattr(sim_data, "simulation", None)
        if simulation is None:
            return

        sources = tuple(getattr(simulation, "sources", ()))
        theta_values, phi_values = self._angle_values()
        expected_source = self._source(
            angle_index=angle_index,
            theta=theta_values[angle_index],
            phi=phi_values[angle_index],
            polarization=polarization,
        )

        if len(sources) != 1 or not isinstance(sources[0], TFSF):
            raise SetupError(
                f"Dipole emission task '{task_name}' must contain the generated TFSF source."
            )

        source = sources[0]
        numeric_checks = (
            ("center", source.center, expected_source.center),
            ("size", source.size, expected_source.size),
            ("angle_theta", source.angle_theta, expected_source.angle_theta),
            ("angle_phi", source.angle_phi, expected_source.angle_phi),
            ("pol_angle", source.pol_angle, expected_source.pol_angle),
        )
        for field_name, actual, expected in numeric_checks:
            if not np.allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=0.0):
                raise SetupError(
                    f"Dipole emission task '{task_name}' has source '{field_name}' that "
                    "does not match this study."
                )

        scalar_checks = (
            ("name", source.name, expected_source.name),
            ("direction", source.direction, expected_source.direction),
            ("injection_axis", source.injection_axis, expected_source.injection_axis),
            ("angular_spec", type(source.angular_spec), type(expected_source.angular_spec)),
            ("source_time", source.source_time, expected_source.source_time),
        )
        for field_name, actual, expected in scalar_checks:
            if actual != expected:
                raise SetupError(
                    f"Dipole emission task '{task_name}' has source '{field_name}' that "
                    "does not match this study."
                )

    def _validate_monitor_array(
        self,
        values: np.ndarray,
        task_label: str,
        array_name: str,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Validate reduced monitor arrays before stacking them into study data."""
        if values.shape != expected_shape:
            raise SetupError(
                f"Dipole emission array '{array_name}' in task '{task_label}' has shape "
                f"{values.shape}; expected {expected_shape}."
            )

    def _validate_monitor_data(
        self,
        monitor_data: Any,
        task_label: str,
    ) -> None:
        """Validate a reduced monitor data object before composing study data."""
        from tidy3d.components.data.monitor_data import DipoleEmissionData

        if not isinstance(monitor_data, DipoleEmissionData):
            raise SetupError(
                f"Dipole emission task '{task_label}' must contain DipoleEmissionData; "
                f"found {type(monitor_data).__name__}."
            )

        expected_monitor = self._emission_monitor()
        monitor = monitor_data.monitor

        if monitor.name != expected_monitor.name:
            raise SetupError(
                f"Dipole emission monitor data in task '{task_label}' references monitor "
                f"'{monitor.name}', expected '{expected_monitor.name}'."
            )
        if tuple(monitor.store_position_indexes) != tuple(expected_monitor.store_position_indexes):
            raise SetupError(
                f"Dipole emission monitor data in task '{task_label}' has stored position indexes "
                "that do not match this study."
            )

        numeric_checks = (
            ("freqs", monitor.freqs, expected_monitor.freqs),
            ("points", monitor.points.values, expected_monitor.points.values),
            ("position_weights", monitor.position_weights, expected_monitor.position_weights),
        )
        for field_name, actual, expected in numeric_checks:
            actual_array = np.asarray(actual, dtype=float)
            expected_array = np.asarray(expected, dtype=float)
            if actual_array.shape != expected_array.shape or not np.allclose(
                actual_array, expected_array, rtol=1e-12, atol=0.0
            ):
                raise SetupError(
                    f"Dipole emission monitor data in task '{task_label}' has '{field_name}' "
                    "that does not match this study."
                )

    def _compose_from_monitor_data(
        self, monitor_data_by_task: dict[tuple[int, Polarization], Any]
    ) -> Any:
        """Build ``DipoleEmissionStudyData`` from reduced per-task monitor data."""
        from tidy3d.plugins.dipole_emission.data import DipoleEmissionStudyData

        num_angles = self.angles.sizes["index"]
        num_pols = len(self.polarizations)
        num_freqs = len(self.freqs)
        num_position_samples = len(self.store_position_indexes)
        pol_index_by_label = {
            polarization: index for index, polarization in enumerate(self.polarizations)
        }

        monitor_records = []
        result_dtypes = []
        expected_monitor_shape = (len(DIPOLE_EMISSION_DIPOLE_AXES), num_freqs)
        expected_position_shape = (
            num_position_samples,
            len(DIPOLE_EMISSION_DIPOLE_AXES),
            num_freqs,
        )
        for task_name, angle_index, polarization in self._expected_task_names():
            monitor_data = monitor_data_by_task[(angle_index, polarization)]
            task_label = f"{task_name}/{EMISSION_MONITOR_NAME}"
            self._validate_monitor_data(monitor_data, task_label)

            intensity_values = np.asarray(monitor_data.radiation_intensity.values)
            self._validate_monitor_array(
                intensity_values,
                task_label,
                "radiation_intensity",
                expected_monitor_shape,
            )
            result_dtypes.append(intensity_values.dtype)

            intensity_position_values = None
            if num_position_samples:
                if monitor_data.radiation_intensity_at_positions is None:
                    raise SetupError(
                        f"Dipole emission task '{task_name}' is missing position-resolved "
                        "radiation intensity requested by 'store_position_indexes'."
                    )
                intensity_position_values = np.asarray(
                    monitor_data.radiation_intensity_at_positions.values,
                )
                self._validate_monitor_array(
                    intensity_position_values,
                    task_label,
                    "radiation_intensity_at_positions",
                    expected_position_shape,
                )
                result_dtypes.append(intensity_position_values.dtype)

            monitor_records.append(
                (angle_index, polarization, intensity_values, intensity_position_values)
            )

        result_dtype = np.result_type(*result_dtypes)
        if not np.issubdtype(result_dtype, np.floating):
            result_dtype = float

        radiation_intensity_values = np.empty(
            (len(DIPOLE_EMISSION_DIPOLE_AXES), num_pols, num_angles, num_freqs),
            dtype=result_dtype,
        )
        position_values = None
        if num_position_samples:
            position_values = np.empty(
                (
                    num_position_samples,
                    len(DIPOLE_EMISSION_DIPOLE_AXES),
                    num_pols,
                    num_angles,
                    num_freqs,
                ),
                dtype=result_dtype,
            )

        for (
            angle_index,
            polarization,
            intensity_values,
            intensity_position_values,
        ) in monitor_records:
            pol_index = pol_index_by_label[polarization]
            radiation_intensity_values[:, pol_index, angle_index, :] = intensity_values

            if num_position_samples:
                position_values[:, :, pol_index, angle_index, :] = intensity_position_values

        radiation_intensity = DipoleEmissionStudyDataArray(
            radiation_intensity_values,
            dims=DipoleEmissionStudyDataArray._dims,
            coords=self._result_coords(),
        )
        radiation_intensity_at_positions = None
        if num_position_samples:
            radiation_intensity_at_positions = DipoleEmissionStudyPositionDataArray(
                position_values,
                dims=DipoleEmissionStudyPositionDataArray._dims,
                coords=self._position_result_coords(),
            )
        return DipoleEmissionStudyData(
            radiation_intensity=radiation_intensity,
            radiation_intensity_at_positions=radiation_intensity_at_positions,
            study=self,
        )

    def to_simulations(self) -> dict[str, Simulation]:
        """Build the simulations for this emission study without submitting them.

        Use this method when you want to inspect the study simulations or submit
        the batch yourself. The returned simulations preserve any
        diagnostic monitors from ``base_sim``; those diagnostic results are not
        part of ``DipoleEmissionStudyData`` and remain available only from the raw
        batch data.

        Returns
        -------
        dict[str, Simulation]
            One ordinary Tidy3D simulation per stored observation angle and far-field
            polarization. The simulations are not submitted by this method.
        """
        theta_values, phi_values = self._angle_values()
        grid_spec = self.base_sim.grid_spec.updated_copy(
            override_structures=(
                *self.base_sim.grid_spec.override_structures,
                self.analysis_region.mesh_override(MESH_OVERRIDE_NAME),
            )
        )

        simulations = {}
        for angle_index, (theta, phi) in enumerate(zip(theta_values, phi_values)):
            for polarization in self.polarizations:
                source = self._source(angle_index, theta, phi, polarization)
                emission_monitor = self._emission_monitor()
                simulations[self._task_name(angle_index, polarization)] = (
                    self.base_sim.updated_copy(
                        sources=(
                            *self.base_sim.sources,
                            source,
                        ),
                        monitors=(*self.base_sim.monitors, emission_monitor),
                        grid_spec=grid_spec,
                        normalize_index=None,
                    )
                )
        return simulations

    def compose(self, batch_data: Any) -> Any:
        """Reduce user-managed batch data into ``DipoleEmissionStudyData``.

        This method is intended for users who submit simulations from
        ``to_simulations(...)`` manually and want to produce the same reduced
        emission result returned by ``run(...)``.

        Parameters
        ----------
        batch_data : td.web.BatchData
            Batch results from simulations produced by ``to_simulations(...)``.
            Data from any user monitors copied from ``base_sim`` remains in this
            raw batch object and is not embedded in the returned
            ``DipoleEmissionStudyData``.

        Returns
        -------
        DipoleEmissionStudyData
            Serializable reduced emission data containing stored ``radiation_intensity`` with
            dimensions ``("dipole_axis", "polarization", "angle", "f")``.
            ``radiation_intensity_transfer(...)`` normalizes ``radiation_intensity`` by
            the bulk emitted power at a provided reference refractive index.
            Position-resolved arrays are included only when
            ``store_position_indexes`` is nonempty.
        """
        self._check_license()
        return self._compose_from_monitor_data(self._monitor_data_from_batch_data(batch_data))

    def _run_batch(
        self,
        folder_name: str,
        batch_kwargs: dict[str, Any],
    ) -> Any:
        """Submit the study simulations."""
        from tidy3d.web import Batch

        batch_kwargs = dict(batch_kwargs)
        batch_init_kwargs = {}
        for key in (
            "verbose",
            "solver_version",
            "callback_url",
            "num_workers",
            "reduce_simulation",
            "pay_type",
            "lazy",
        ):
            if key in batch_kwargs:
                batch_init_kwargs[key] = batch_kwargs.pop(key)

        batch = Batch(
            simulations=self.to_simulations(),
            folder_name=folder_name,
            **batch_init_kwargs,
        )
        return batch.run(**batch_kwargs)

    def run(
        self,
        folder_name: str = "dipole_emission",
        return_batch_data: bool = False,
        **batch_kwargs: Any,
    ) -> Any:
        """Run this emission study and return angular radiation intensity.

        The study returns radiation per squared electric dipole moment, with
        dipole moment expressed in C*um, summed over the sampled dipole
        positions using ``position_weights``. The compact result is indexed by
        Cartesian dipole orientation, far-field polarization, observation
        angle, and frequency. To recover an absolute angular power density for
        a physical dipole moment ``d``, multiply ``radiation_intensity`` by
        ``|d|**2``.

        Parameters
        ----------
        folder_name : str = "dipole_emission"
            Cloud folder name used for the study batch.
        return_batch_data : bool = False
            If ``False``, return only the reduced ``DipoleEmissionStudyData``. If
            ``True``, return ``(dipole_data, batch_data)`` so copied diagnostic
            monitors and raw simulation data remain available to the caller.
        **batch_kwargs
            Additional keyword arguments forwarded to ``td.web.Batch(...)`` for
            batch construction options such as ``verbose`` and to
            ``td.web.Batch.run(...)`` for run options such as ``path_dir``.

        Returns
        -------
        DipoleEmissionStudyData or tuple[DipoleEmissionStudyData, td.web.BatchData]
            Reduced emission data, optionally paired with the raw batch data.
        """
        self._check_license()
        batch_data = self._run_batch(
            folder_name=folder_name,
            batch_kwargs=batch_kwargs,
        )
        emission_data = self.compose(batch_data)
        if return_batch_data:
            return emission_data, batch_data
        return emission_data
