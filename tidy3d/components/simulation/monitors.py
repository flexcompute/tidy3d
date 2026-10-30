"""Monitor storage estimates and geometry/frequency validation for Yee simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from flexcompute.core._migration.em.material.medium.comparison import optical_medium_key

from tidy3d.components.base import cached_property
from tidy3d.components.data.point_cloud import (
    POINT_CLOUD_PERMITTIVITY_COMPONENTS,
    point_cloud_nearest_sampled_cells_upper_bound,
    point_cloud_num_sampled_grid_fields,
    point_cloud_sampled_cells_upper_bound,
)
from tidy3d.components.diffraction import (
    diffraction_monitor_storage_size,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.monitor import (
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    FieldStructureTimeMonitor,
    FieldTimeMonitor,
    FreqMonitor,
    MediumMonitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
    SurfaceIntegrationMonitor,
    TimeMonitor,
)
from tidy3d.components.scene import Scene
from tidy3d.components.structure import _require_public_optical_medium
from tidy3d.config import config
from tidy3d.exceptions import SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Coords1D
    from tidy3d.components.medium import AbstractMedium, MediumType3D
    from tidy3d.components.monitor import AbstractGaussianOverlapMonitor, FieldMonitor, Monitor
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Coordinate
    from tidy3d.components.types.monitor import MonitorType

from . import constants


def _monitor_num_cells(self: Any, monitor: Monitor) -> int:
    """Total number of cells included in monitor based on simulation grid."""

    if isinstance(monitor, PointCloudFieldMonitor):
        return point_cloud_sampled_cells_upper_bound(
            num_cells=self.grid.num_cells,
            symmetry=self.symmetry,
            num_points=monitor.num_points,
            num_fields=point_cloud_num_sampled_grid_fields(monitor.fields),
        )
    if isinstance(monitor, PointCloudPermittivityMonitor):
        return point_cloud_nearest_sampled_cells_upper_bound(
            num_cells=self.grid.num_cells,
            symmetry=self.symmetry,
            num_points=monitor.num_points,
            num_components=len(POINT_CLOUD_PERMITTIVITY_COMPONENTS),
        )

    def num_cells_in_monitor(monitor: Monitor) -> int:
        """Get the number of measurement cells in a monitor given the simulation grid and
        downsampling."""
        if not self.intersects(monitor):
            # Monitor is outside of simulation domain; can happen e.g. for integration surfaces
            return 0
        num_cells = self.discretize_monitor(monitor).num_cells
        # take monitor downsampling into account
        num_cells = monitor.downsampled_num_cells(num_cells)
        return np.prod(np.array(num_cells, dtype=np.int64))

    if isinstance(monitor, SurfaceIntegrationMonitor):
        return sum(num_cells_in_monitor(mnt) for mnt in monitor.integration_surfaces)
    return num_cells_in_monitor(monitor)


def _validate_record_in_pml_unsupported(self: Any) -> None:
    """Error on ``record_in_pml=False`` in simulation types whose monitor data does not come
    from the FDTD solver, for which the truncation is not implemented."""
    for index, monitor in enumerate(self.monitors):
        if not getattr(monitor, "record_in_pml", True):
            self._raise_validation_error_at_loc(
                f"'record_in_pml=False' is not supported in '{type(self).__name__}' "
                f"(monitor '{monitor.name}').",
                "monitors",
                index,
                "record_in_pml",
            )

    return self


@classmethod
def _projection_monitor_mediums_in_bounds(
    cls: Any,
    center: Coordinate,
    size: Coordinate,
    monitor: FieldMonitor
    | SurfaceIntegrationMonitor
    | DiffractionMonitor
    | AbstractGaussianOverlapMonitor,
    structures: list[Structure],
) -> set[MediumType3D]:
    """Get media intersecting the in-domain portion of a projection surface or monitor."""

    sim_box = Box(center=center, size=size).to_static()
    monitor = monitor.to_static()
    structures = [structure.to_static() for structure in structures]
    mediums = set()
    has_nonzero_measure_region = False
    has_zero_measure_clip = False
    surfaces = [monitor]
    if isinstance(monitor, SurfaceIntegrationMonitor):
        surfaces = monitor.integration_surfaces

    for surface in surfaces:
        intersection_bounds = Box.bounds_intersection(surface.bounds, sim_box.bounds)
        if not all(bmin <= bmax for bmin, bmax in zip(*intersection_bounds)):
            continue

        clipped_surface = Box.from_bounds(*intersection_bounds).to_static()
        num_zero_dims = clipped_surface.size.count(0.0)
        if num_zero_dims == 1:
            has_nonzero_measure_region = True
            mediums.update(
                cls._projection_monitor_media_on_plane(
                    test_object=clipped_surface,
                    plane=clipped_surface,
                    structures=structures,
                )
            )
        elif num_zero_dims == 2 and sim_box.size.count(0.0) == 1:
            has_nonzero_measure_region = True
            mediums.update(
                cls._projection_monitor_media_on_plane(
                    test_object=clipped_surface,
                    plane=sim_box,
                    structures=structures,
                )
            )
        else:
            has_zero_measure_clip = True

    if has_zero_measure_clip and not has_nonzero_measure_region:
        raise SetupError(
            f"All in-domain clipped portions of '{monitor.name}' ({monitor.type}) collapse "
            "to zero-measure sets after clipping to the simulation bounds. "
            "Projection surfaces must have a nonzero in-domain integration region "
            "(area in 3D or line length in 2D)."
        )

    return mediums


@classmethod
def _projection_monitor_media_on_plane(
    cls: Any,
    test_object: Box,
    plane: Box,
    structures: list[Structure],
) -> set[MediumType3D]:
    """Get media intersecting a planar or line-like test object within a given plane."""

    test_shapes = plane.intersections_with(test_object)
    medium_shapes = Scene._filter_structures_plane_medium(structures, plane)
    mediums = set()

    for test_shape in test_shapes:
        if test_shape.is_empty:
            continue

        for medium, medium_shape in medium_shapes:
            overlap = test_shape & medium_shape
            if overlap.area > 0 or overlap.length > 0:
                mediums.add(medium)

    return mediums


def _validate_monitor_size(self: Any) -> None:
    """Ensures the monitors aren't storing too much data before simulation is uploaded."""

    if config.simulation.skip_size_checks:
        return

    total_size_gb = 0
    with log as consolidated_logger:
        datas = self.monitors_data_size
        for monitor_ind, (monitor_name, monitor_size) in enumerate(datas.items()):
            monitor_size_gb = monitor_size / 1e9
            if monitor_size_gb > constants.WARN_MONITOR_DATA_SIZE_GB:
                consolidated_logger.warning(
                    f"Estimated storage of {self._monitor_validation_label(monitor_name)} "
                    f"is {monitor_size_gb:1.2f}GB. "
                    "Consider making it smaller, using fewer frequencies, or spatial or "
                    "temporal downsampling using 'interval_space' and 'interval', respectively.",
                    custom_loc=[
                        "monitors",
                        self._monitor_validation_index(
                            monitor_name=monitor_name, fallback_index=monitor_ind
                        ),
                    ],
                )

            total_size_gb += monitor_size_gb

    if total_size_gb > constants.MAX_SIMULATION_DATA_SIZE_GB:
        raise SetupError(
            f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
            f"a maximum of {constants.MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed."
        )

    # Some monitors store much less data than what is needed internally. Make sure that the
    # internal storage also does not exceed the limit.
    for monitor_ind, monitor in enumerate(self.monitors):
        num_cells = self._monitor_num_cells(monitor)
        # intermediate storage needed, in GB
        solver_data = monitor._storage_size_solver(num_cells=num_cells, tmesh=self.tmesh) / 1e9
        if (
            isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor))
            and self.precision == "double"
        ):
            solver_data *= 2
        if solver_data > constants.MAX_MONITOR_INTERNAL_DATA_SIZE_GB:
            self._raise_validation_error_at_loc(
                f"Estimated internal storage of {self._monitor_validation_label(monitor)} is "
                f"{solver_data:1.2f}GB, which is larger than the maximum allowed "
                f"{constants.MAX_MONITOR_INTERNAL_DATA_SIZE_GB:.2f}GB. Consider making it smaller, "
                "using fewer frequencies, or spatial or temporal downsampling using "
                "'interval_space' and 'interval', respectively.",
                "monitors",
                self._monitor_validation_index(
                    monitor_name=monitor.name, fallback_index=monitor_ind
                ),
            )


def _validate_time_monitors_num_steps(self: Any) -> None:
    """Raise an error if non-0D time monitors have too many time steps."""
    if config.simulation.skip_size_checks:
        return

    for monitor in self.monitors:
        if (
            not isinstance(
                monitor, FieldTimeMonitor | FieldStructureTimeMonitor | AuxFieldTimeMonitor
            )
            or len(monitor.zero_dims) == 3
        ):
            continue
        num_time_steps = monitor.num_steps(self.tmesh)
        if num_time_steps > constants.MAX_TIME_MONITOR_STEPS:
            raise SetupError(
                f"Time monitor '{monitor.name}' records at {num_time_steps} time steps, which "
                f"is larger than the maximum allowed value of {constants.MAX_TIME_MONITOR_STEPS} when "
                "the monitor is not zero-dimensional. Change the geometry to a point monitor, "
                "or use 'start', 'stop', and 'interval' to reduce the number of time steps "
                "at which the monitor stores data."
            )


def _validate_freq_monitors_freq_range(self: Any) -> None:
    """Rise the error if any DFT monitors have frequencies outside of the simulation frequency range."""
    source_ranges = [source.source_time._frequency_range_sigma_cached for source in self.sources]
    if not source_ranges:
        return

    freq_min = (
        min((freq_range[0] for freq_range in source_ranges), default=0.0)
        * constants.MIN_MONITOR_FREQUENCY_RANGE_PARAMETER
    )
    freq_max = (
        max((freq_range[1] for freq_range in source_ranges), default=0.0)
        * constants.MAX_MONITOR_FREQUENCY_RANGE_PARAMETER
    )
    sci_fmin, sci_fmax = self._scientific_notation(freq_min, freq_max)

    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, FreqMonitor) or isinstance(
            monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
        ):
            continue

        freqs = np.array(monitor.freqs)
        if freqs.min() < freq_min or freqs.max() > freq_max:
            self._raise_validation_error_at_loc(
                f"Frequency {self._monitor_validation_label(monitor)} contains frequencies "
                f"outside of the simulation frequency range ({sci_fmin}, {sci_fmax})"
                "(Hz) as defined by the sources.",
                "monitors",
                self._monitor_validation_index(
                    monitor_name=monitor.name, fallback_index=monitor_ind
                ),
                "freqs",
            )


def _monitors_data_size(self: Any, tmesh: Coords1D) -> dict[str, float]:
    """Map monitor names to estimated storage sizes for a resolved time mesh."""
    data_size = {}
    for monitor in self.monitors:
        if isinstance(monitor, DiffractionMonitor):
            medium = self.monitor_medium(monitor)
            storage_size = float(diffraction_monitor_storage_size(self, monitor, medium))
        else:
            num_cells = self._monitor_num_cells(monitor)
            storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=tmesh))
        if isinstance(monitor, DipoleEmissionMonitor) and self.precision == "double":
            storage_size *= 2
        elif (
            isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor))
            and not isinstance(monitor, DipoleEmissionMonitor)
            and self.precision == "double"
        ):
            points_size = np.asarray(monitor.points.values).nbytes
            storage_size = points_size + 2 * (storage_size - points_size)
        data_size[monitor.name] = storage_size
    return data_size


@cached_property
def monitors_data_size(self: Any) -> dict[str, float]:
    """Dictionary mapping monitor names to their estimated storage size in bytes."""
    return self._monitors_data_size(self.tmesh)


def _validate_datasets_not_none(self: Any) -> None:
    """Ensures that all custom datasets are defined."""
    if any(dataset is None for dataset in self.custom_datasets):
        raise SetupError(
            "Data for a custom data component is missing. This can happen for example if the "
            "Simulation has been loaded from json. To save and load simulations with custom "
            "data, use hdf5 format instead."
        )


def _warn_time_monitors_outside_run_time(self: Any) -> None:
    """Warn if time monitors start after the simulation run_time."""
    with log as consolidated_logger:
        for monitor in self.monitors:
            if isinstance(monitor, TimeMonitor) and monitor.start > self._run_time:
                consolidated_logger.warning(
                    f"Monitor {monitor.name} has a start time {monitor.start:1.2e}s exceeding"
                    f"the simulation run time {self._run_time:1.2e}s. No data will be recorded."
                )


def monitor_medium(self: Any, monitor: MonitorType) -> AbstractMedium:
    """Return the medium in which the given monitor resides.

    Parameters
    -------
    monitor : :class:`.Monitor`
        Monitor whose associated medium is to be returned.

    Returns
    -------
    :class:`.AbstractMedium`
        Medium associated with the given :class:`.Monitor`.
    """
    medium_set = {
        _require_public_optical_medium(medium)
        for medium in Scene.intersecting_media(monitor, self.structures)
    }
    if len({optical_medium_key(medium) for medium in medium_set}) > 1:
        raise SetupError(  # post-init-tidy3d-error: ignore
            f"Monitor '{monitor.name}' intersects more than one medium."
        )
    medium = medium_set.pop() if medium_set else self.medium
    return _require_public_optical_medium(medium)
