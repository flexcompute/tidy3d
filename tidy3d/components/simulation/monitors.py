"""Monitor storage estimates and geometry/frequency validation for Yee simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BlochBoundary, Periodic
from tidy3d.components.data.point_cloud import (
    POINT_CLOUD_PERMITTIVITY_COMPONENTS,
    point_cloud_nearest_sampled_cells_upper_bound,
    point_cloud_num_sampled_grid_fields,
    point_cloud_sampled_cells_upper_bound,
)
from tidy3d.components.diffraction import (
    diffraction_monitor_storage_size,
    diffraction_order_grid_size,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.medium import (
    AbstractCustomMedium,
    AnisotropicMediumFromMedium2D,
    Medium2D,
)
from tidy3d.components.monitor import (
    AbstractFieldProjectionMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DipoleEmissionMonitor,
    DirectivityMonitor,
    FieldStructureMonitor,
    FieldTimeMonitor,
    FreqMonitor,
    MediumMonitor,
    ModeTimeMonitor,
    PermittivityMonitor,
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
    SurfaceIntegrationMonitor,
    TimeMonitor,
)
from tidy3d.components.scene import Scene
from tidy3d.components.structure import Structure
from tidy3d.components.types.monitor import SurfaceMonitorType
from tidy3d.components.validators import (
    named_obj_descr,
    points_outside_bounds,
    validate_field_projection_monitors_2d,
)
from tidy3d.config import config
from tidy3d.exceptions import SetupError, Tidy3dError
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.grid.grid import Coords1D
    from tidy3d.components.medium import AbstractMedium, MediumType3D
    from tidy3d.components.monitor import AbstractGaussianOverlapMonitor, FieldMonitor, Monitor
    from tidy3d.components.types import Bound, Coordinate
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


def _validate_mode_time_monitor_freq_range(self: Any) -> Self:
    """Validate the solve frequency for ``ModeTimeMonitor``\\ s with ``freq_spec=None``.

    A ``ModeTimeMonitor`` with ``freq_spec=None`` derives its single solve frequency from
    the sources (the central frequency of the first source). Error if there are no sources
    to derive it from; warn if the sources do not share a common central frequency.
    """
    needs_freq = [
        idx
        for idx, monitor in enumerate(self.monitors)
        if isinstance(monitor, ModeTimeMonitor) and monitor.freq_spec is None
    ]
    if not needs_freq:
        return self

    idx = needs_freq[0]
    self._check_source_freq_available(
        no_source_error=(
            f"'ModeTimeMonitor' '{self.monitors[idx].name}' has 'freq_spec=None', which "
            "requires the simulation to contain at least one source to derive the "
            "mode-sampling frequency from. Set 'freq_spec' explicitly for a "
            "source-free simulation."
        ),
        no_source_loc=("monitors", idx),
        multi_freq_warning=(
            "At least one 'ModeTimeMonitor' does not specify 'freq_spec', the frequency at "
            "which its mode profiles are solved. The central frequency of the first source "
            "will be used."
        ),
    )
    return self


def _warn_monitor_mediums_frequency_range(self: Any) -> Self:
    """Warn user if any DFT monitors have frequencies outside of medium frequency range."""
    val = self.monitors

    if val is None:
        return self

    structures = self.structures or []
    medium_bg = self.medium
    mediums = [medium_bg] + [structure.medium for structure in structures]

    with log as consolidated_logger:
        for monitor_index, monitor in enumerate(val):
            if not isinstance(monitor, FreqMonitor):
                continue

            freqs = np.array(monitor.freqs)
            fmin_mon = freqs.min()
            fmax_mon = freqs.max()
            for medium_index, medium in enumerate(mediums):
                # skip mediums that have no freq range (all freqs valid)
                if medium.frequency_range is None:
                    continue

                # make sure medium frequency range includes all monitor frequencies
                fmin_med, fmax_med = medium.frequency_range
                sci_fmin_med, sci_fmax_med = self._scientific_notation(fmin_med, fmax_med)

                if fmin_mon < fmin_med or fmax_mon > fmax_med:
                    if medium_index == 0:
                        medium_str = "The simulation background medium"
                        custom_loc = ["medium", "frequency_range"]
                    else:
                        medium_descr = named_obj_descr(medium, "mediums", medium_index)
                        medium_str = f"The medium associated with {medium_descr}"
                        custom_loc = [
                            "structures",
                            str(medium_index - 1),
                            "medium",
                            "frequency_range",
                        ]

                    monitor_descr = named_obj_descr(monitor, "monitors", monitor_index)
                    consolidated_logger.warning(
                        f"{medium_str} has a frequency range: ({sci_fmin_med}, {sci_fmax_med}) "
                        "(Hz) that does not fully cover the frequencies contained "
                        f"in {monitor_descr}."
                        "This can cause inaccuracies in the recorded results.",
                        custom_loc=custom_loc,
                    )
    return self


def _warn_monitor_simulation_frequency_range(self: Any) -> Self:
    """Warn if any DFT monitors have frequencies outside of the simulation frequency range."""
    val = self.monitors

    if val is None:
        return self

    source_ranges = [source.source_time._frequency_range_sigma_cached for source in self.sources]
    if not source_ranges:
        # Commented out to eliminate this message from Mode real time log in GUI
        # TODO: Bring it back when it doesn't interfere with mode solver
        # log.info("No sources in simulation.")
        return self

    freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
    freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)
    sci_fmin, sci_fmax = self._scientific_notation(freq_min, freq_max)

    with log as consolidated_logger:
        for monitor_index, monitor in enumerate(val):
            if not isinstance(monitor, FreqMonitor) or isinstance(
                monitor, PermittivityMonitor | MediumMonitor | PointCloudPermittivityMonitor
            ):
                continue

            freqs = np.array(monitor.freqs)
            if freqs.min() < freq_min or freqs.max() > freq_max:
                consolidated_logger.warning(
                    f"'monitors[{monitor_index}]' contains frequencies "
                    f"outside of the simulation frequency range ({sci_fmin}, {sci_fmax})"
                    "(Hz) as defined by the sources.",
                    custom_loc=["monitors", monitor_index, "freqs"],
                )
    return self


def _validate_point_cloud_monitor_points_in_bounds(self: Any) -> Self:
    """Error if any point-cloud monitor point lies outside the simulation domain."""

    if not self.monitors:
        return self

    bounds = np.asarray(self.bounds, dtype=float)
    strict_inequality = np.asarray([size != 0 for size in self.size], dtype=bool)
    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, (PointCloudFieldMonitor, PointCloudPermittivityMonitor)):
            continue

        points = np.asarray(monitor.points.values, dtype=float)
        outside = points_outside_bounds(points, bounds, strict_inequality)
        if np.any(outside):
            first_row = int(np.nonzero(outside)[0][0])
            first_index = np.asarray(monitor.points.coords["index"].values)[first_row]
            first_index = first_index.item() if hasattr(first_index, "item") else first_index
            num_outside = int(np.count_nonzero(outside))
            self._raise_validation_error_at_loc(
                f"Point-cloud monitor '{monitor.name}' has {num_outside} point(s) outside "
                "the simulation domain. The first outside point has index "
                f"{first_index} and coordinates {points[first_row].tolist()}.",
                "monitors",
                monitor_ind,
                "points",
            )

    return self


def _validate_field_structure_monitor_overlaps(self: Any) -> Self:
    """Reject a ``FieldStructureMonitor`` overlapping a structure with no public owner.

    The monitor records per-Yee structure-ownership indices into ``simulation.structures``
    (RFC EMSOLVER-0010). Derived structures carry no such index — 2D materials (converted to
    volumetric analogues), lumped elements, and the automatic PEC frames around mode sources
    and internal absorbers — so the monitor is not allowed to intersect them.

    The check runs against both the user-specified geometries and the finalized
    (grid-snapped) geometries the solver sees: conversion snaps a 2D sheet to the nearest
    grid boundary — up to half a cell along its normal — so the user geometry alone would
    miss a monitor that overlaps only the snapped sheet.
    """
    field_structure_monitors = [
        (ind, monitor)
        for ind, monitor in enumerate(self.monitors)
        if isinstance(monitor, FieldStructureMonitor)
    ]
    if not field_structure_monitors:
        return self

    # Unwrap the optical medium so MultiPhysicsMedium(optical=Medium2D(...)) is also caught;
    # its derived volumetric structure likewise has no public ownership index.
    derived = [
        ("a 2D-material structure", structure.geometry)
        for structure in self.structures
        if isinstance(structure._optical_medium, Medium2D | AnisotropicMediumFromMedium2D)
    ]
    derived += [("a lumped element", element.geometry) for element in self.lumped_elements]
    derived += [
        ("an automatic PEC frame around a mode source or internal absorber", frame.geometry)
        for frame in self._modal_plane_frames
    ]
    if not derived:
        return self

    if self._contains_converted_volumetric_structures:
        # The derived structures are exactly the finalized entries with no counterpart in
        # ``static_structures`` — the same identity rule the solver export uses to assign
        # "no public owner" ownership (``public_str_index = -1``).
        try:
            kept = {id(structure) for structure in self.static_structures}
            kept |= {id(frame) for frame in self._modal_plane_frames}
            derived += [
                (
                    "the grid-snapped volumetric equivalent of a 2D material or lumped element",
                    structure.geometry,
                )
                for structure in self._finalized_volumetric_structures
                if id(structure) not in kept
            ]
        except Tidy3dError:
            # Conversion can fail on degenerate grids (e.g. a sheet snapped to the domain
            # edge); that failure is diagnosed with a clearer error at export, so fall back
            # to the user-specified geometries only.
            pass

    for monitor_ind, monitor in field_structure_monitors:
        region = monitor.geometry
        for description, geometry in derived:
            if region.intersects(geometry):
                self._raise_validation_error_at_loc(
                    f"'FieldStructureMonitor' '{monitor.name}' overlaps {description}. This "
                    "monitor records per-Yee structure-ownership indices into "
                    "'simulation.structures', which is undefined for structures without a "
                    "public owner (2D materials, lumped elements, and automatic PEC frames). "
                    "Move or resize the monitor so it does not intersect these structures.",
                    "monitors",
                    monitor_ind,
                )
    return self


def _diffraction_monitor_boundaries(self: Any) -> Self:
    """If any :class:`.DiffractionMonitor` exists, ensure boundary conditions in the
    transverse directions are periodic or Bloch."""
    monitors = self.monitors
    boundary_spec = self.boundary_spec
    for monitor_index, monitor in enumerate(monitors):
        if isinstance(monitor, DiffractionMonitor):
            _, (n_x, n_y) = monitor.pop_axis(["x", "y", "z"], axis=monitor.normal_axis)
            boundaries = [
                boundary_spec[n_x].plus,
                boundary_spec[n_x].minus,
                boundary_spec[n_y].plus,
                boundary_spec[n_y].minus,
            ]
            # make sure the transverse boundaries are either periodic or Bloch
            for boundary in boundaries:
                if not isinstance(boundary, Periodic | BlochBoundary):
                    self._raise_validation_error_at_loc(
                        f"The 'DiffractionMonitor' {monitor.name} requires periodic "
                        f"or Bloch boundaries along dimensions {n_x} and {n_y}.",
                        "monitors",
                        monitor_index,
                    )
    return self


def _projection_monitors_homogeneous(self: Any) -> Self:
    """Error if any field projection monitor is not in a homogeneous region."""
    val = self.monitors

    if val is None:
        return self

    # list of structures including background as a Box()
    structure_bg = Structure(
        geometry=Box(
            size=self.size,
            center=self.center,
        ),
        medium=self.medium,
    )

    structures = self.structures or []
    total_structures = [structure_bg, *list(structures)]

    with log as consolidated_logger:
        for monitor_ind, monitor in enumerate(val):
            if isinstance(monitor, AbstractFieldProjectionMonitor | DiffractionMonitor):
                mediums = self._call_with_validation_loc(
                    ["monitors", monitor_ind],
                    self._projection_monitor_mediums_in_bounds,
                    center=self.center,
                    size=self.size,
                    monitor=monitor,
                    structures=total_structures,
                )
                if len(mediums) < 1:
                    continue
                # make sure there is no more than one medium in the returned list
                if len(mediums) > 1:
                    self._raise_validation_error_at_loc(
                        f"{len(mediums)} different mediums detected on plane "
                        f"intersecting a {monitor.type}. Plane must be homogeneous.",
                        "monitors",
                        monitor_ind,
                    )
                # 1 medium, check if the medium is spatially uniform
                if not list(mediums)[0].is_spatially_uniform:
                    consolidated_logger.warning(
                        f"Nonuniform custom medium detected on plane intersecting a {monitor.type}. "
                        "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                        custom_loc=["monitors", monitor_ind],
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


def _proj_distance_for_approx(self: Any) -> Self:
    """Warn if projection distance for projection monitors is not large compared to monitor or,
    simulation size, yet far_field_approx is True."""
    val = self.monitors

    if val is None:
        return self

    sim_size = self.size

    with log as consolidated_logger:
        for monitor_ind, monitor in enumerate(val):
            if not isinstance(monitor, AbstractFieldProjectionMonitor):
                continue

            name = monitor.name
            max_size = min(np.max(monitor.size), np.max(sim_size))

            if monitor.far_field_approx and np.abs(monitor.proj_distance) < 10 * max_size:
                consolidated_logger.warning(
                    f"Monitor {name} projects to a distance comparable to the size of the "
                    "monitor; we recommend setting ``far_field_approx=False`` to disable "
                    "far-field approximations for this monitor, because the approximations "
                    "are valid only when the observation points are very far compared to the "
                    "size of the monitor that records near fields.",
                    custom_loc=["monitors", monitor_ind],
                )
    return self


def _integration_surfaces_in_bounds(self: Any) -> Self:
    """Error if all of the integration surfaces are outside of the simulation domain."""
    val = self.monitors

    if val is None:
        return self

    sim_center = self.center
    sim_size = self.size
    sim_box = Box(size=sim_size, center=sim_center)

    for monitor_ind, mnt in enumerate(val):
        if not isinstance(mnt, SurfaceIntegrationMonitor):
            continue
        if not any(sim_box.intersects(surf) for surf in mnt.integration_surfaces):
            self._raise_validation_error_at_loc(
                f"All integration surfaces of monitor '{mnt.name}' are outside of the "
                "simulation bounds.",
                "monitors",
                monitor_ind,
            )

    return self


def _projection_monitors_distance(self: Any) -> Self:
    """Warn if the projection distance is large for exact projections."""
    val = self.monitors

    if val is None:
        return self

    sim_size = self.size

    with log as consolidated_logger:
        for idx, monitor in enumerate(val):
            if isinstance(monitor, AbstractFieldProjectionMonitor):
                if (
                    np.abs(monitor.proj_distance) > 1.0e4 * np.max(sim_size)
                    and not monitor.far_field_approx
                ):
                    monitor = monitor.copy(update={"far_field_approx": True})
                    val = list(val)
                    val[idx] = monitor
                    val = tuple(val)
                    consolidated_logger.warning(
                        "A very large projection distance was set for the field projection "
                        f"monitor '{monitor.name}'. Using exact field projections may result "
                        "in precision loss for large distances; automatically enabling "
                        "far-field approximations ('far_field_approx = True') for better "
                        "precision. To insist on exact projections, consider using client-side "
                        "projections via the 'FieldProjector' class, where higher precision is "
                        "available.",
                        custom_loc=["monitors", idx, "proj_distance"],
                    )
    return self


def _projection_monitors_boundaries(self: Any) -> Self:
    """Error if 3D field projection monitors are used with periodic or Bloch boundaries."""
    monitors = self.monitors

    if not monitors or self.size.count(0.0) != 0 or not any(self._periodic):
        return self

    for monitor_ind, monitor in enumerate(monitors):
        if isinstance(monitor, AbstractFieldProjectionMonitor):
            self._raise_validation_error_at_loc(
                f"Monitor '{monitor.name}' of type '{monitor.type}' cannot be used with "
                "periodic/Bloch boundaries in 3D simulations. This projection would "
                "require a periodic Green's function. Please use 'DiffractionMonitor' for "
                "transmission/reflection analysis with periodic/Bloch boundaries.",
                "monitors",
                monitor_ind,
            )

    return self


def _projection_mnts_2d(self: Any) -> Self:
    """
    Validate if the field projection monitor is set up for a 2D simulation and
    ensure the observation parameters are configured correctly.

    - For a 2D simulation in the x-y plane, ``theta`` should be set to ``pi/2``.
    - For a 2D simulation in the y-z plane, ``phi`` should be set to ``pi/2`` or ``3*pi/2``.
    - For a 2D simulation in the x-z plane, ``phi`` should be set to ``0`` or ``pi``.

    Note: Exact far field projection is not available yet. Currently, only
    ``far_field_approx = True`` is supported.
    """
    validate_field_projection_monitors_2d(
        self.monitors,
        self.size,
        raise_error=lambda message, monitor_ind: self._raise_validation_error_at_loc(
            message, "monitors", monitor_ind
        ),
    )
    return self


def _diffraction_and_directivity_monitor_medium(self: Any) -> Self:
    """If any :class:`.DiffractionMonitor` or  :class:`.DirectivityMonitor` exists, ensure it does not lie in a lossy medium."""
    monitors = self.monitors
    structures = self.structures
    medium = self.medium
    for monitor_ind, monitor in enumerate(monitors):
        if isinstance(monitor, DiffractionMonitor | DirectivityMonitor):
            medium_set = Scene.intersecting_media(monitor, structures)
            medium = medium_set.pop() if medium_set else medium
            freqs = np.array(monitor.freqs)
            if isinstance(medium, AbstractCustomMedium) and len(freqs) > 1:
                freqs = 0.5 * (np.min(freqs) + np.max(freqs))
            _, index_k = medium.nk_model(frequency=freqs)
            if not np.all(index_k == 0):
                self._raise_validation_error_at_loc(
                    f"'{monitor.type}' must not lie in a lossy medium.",
                    "monitors",
                    monitor_ind,
                )
    return self


def _diffraction_monitor_order_grid_size(self: Any) -> Self:
    """Error if a diffraction monitor would generate an excessively large order grid."""

    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, DiffractionMonitor):
            continue

        medium = self.monitor_medium(monitor)
        total_orders = diffraction_order_grid_size(self, monitor, medium)
        if total_orders > constants.MAX_DIFFRACTION_ORDER_GRID_SIZE:
            self._raise_validation_error_at_loc(
                f"The 'DiffractionMonitor' {monitor.name} would generate "
                f"{total_orders} diffraction order combinations, which exceeds "
                f"the supported limit of {constants.MAX_DIFFRACTION_ORDER_GRID_SIZE}. "
                "Verify that units are set correctly (by default, lengths are specified "
                "in microns and frequencies in Hz). Reduce the monitor frequencies, "
                "the refractive index on the monitor plane, or the simulation size "
                "along the transverse directions.",
                "monitors",
                monitor_ind,
            )

    return self


@classmethod
def _get_surface_monitor_bounds(
    cls: Any,
    center: Coordinate,
    size: Coordinate,
    monitor: SurfaceMonitorType,
    medium: MediumType3D,
    structures: list[Structure],
) -> list[Bound]:
    """Intersect a surface monitor with the bounding box of each PEC structure."""

    sim_box = Box(center=center, size=size)
    mnt_bounds = Box.bounds_intersection(monitor.bounds, sim_box.bounds)

    if medium.is_pec_like:
        return [mnt_bounds]

    bounds = []
    for structure in structures:
        if structure.medium.is_pec_like:
            intersection_bounds = Box.bounds_intersection(mnt_bounds, structure.geometry.bounds)
            if all(bmin <= bmax for bmin, bmax in zip(*intersection_bounds)):
                bounds.append(intersection_bounds)

    return bounds


def _error_empty_surface_monitor(self: Any) -> Self:
    """Error if any surface monitor does not at least cross a bounding box of a PEC/LossyMetal structure."""
    for monitor_ind, mnt in enumerate(self.monitors):
        if isinstance(mnt, get_args(SurfaceMonitorType)):
            bounds = self._get_surface_monitor_bounds(
                self.center, self.size, mnt, self.medium, self.structures
            )
            if len(bounds) == 0:
                self._raise_validation_error_at_loc(
                    f"Surface monitor {mnt.name} does not cross any PEC or lossy metal "
                    "(LossyMetalMedium with penetrable=False) surface.",
                    "monitors",
                    monitor_ind,
                )
    return self


def _error_surface_monitors_with_zero_size(self: Any) -> Self:
    """Error if simulation has surface monitors and the size of domain is zero along any dimension."""
    not_3d = any(dim == 0 for dim in self.size)
    if not_3d:
        for monitor_ind, mnt in enumerate(self.monitors):
            if isinstance(mnt, get_args(SurfaceMonitorType)):
                self._raise_validation_error_at_loc(
                    "Simulation domain has size zero along at least one dimension; surface monitors are not allowed in this case.",
                    "monitors",
                    monitor_ind,
                )
    return self


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
            not isinstance(monitor, FieldTimeMonitor | AuxFieldTimeMonitor)
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
    medium_set = Scene.intersecting_media(monitor, self.structures)
    if len(medium_set) > 1:
        raise SetupError(  # post-init-tidy3d-error: ignore
            f"Monitor '{monitor.name}' intersects more than one medium."
        )
    medium = medium_set.pop() if medium_set else self.medium
    return medium
