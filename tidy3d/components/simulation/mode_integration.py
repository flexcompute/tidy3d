"""Mode-plane finalization and validation shared by simulation and mode workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BlochBoundary, ModeABCBoundary
from tidy3d.components.geometry.base import Box, GeometryGroup
from tidy3d.components.geometry.utils import find_snap_location
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.medium import PECMedium
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.mode.geometry import (
    effective_mode_plane,
    rotation_kwargs,
    rotation_translate_kwargs,
    rotation_validation_freqs,
    snapped_mode_domain,
)
from tidy3d.components.mode.validation import (
    make_rotated_structures,
    validate_microwave_mode_spec,
    validate_mode_plane_radius,
    validate_plane_rotation_media,
    warn_thick_pml,
)
from tidy3d.components.monitor import (
    AbstractModeMonitor,
    FieldMonitor,
    ModeMonitor,
    ModeTimeMonitor,
)
from tidy3d.components.scene import Scene
from tidy3d.components.source.field import AbstractModeSource
from tidy3d.components.source.frame import PECFrame
from tidy3d.components.structure import Structure
from tidy3d.exceptions import (
    SetupError,
    Tidy3dError,
    format_chained_exception_message,
)
from tidy3d.log import log

from .materials import _medium_can_be_lossy

if TYPE_CHECKING:
    from pydantic import NonNegativeInt

    from tidy3d.compat import Self
    from tidy3d.components.boundary import InternalAbsorber
    from tidy3d.components.medium import MediumType
    from tidy3d.components.source.field import ModeSource
    from tidy3d.components.types import Axis

from . import constants


def _make_pec_frame(
    self: Any,
    obj: AbstractModeSource | InternalAbsorber,
    name: str | None = None,
) -> Structure:
    """Make the PEC frame around a mode source or internal absorber.

    The frame is an open tube: four lateral walls with both axis caps removed. An internal
    absorber is additionally closed by a PEC backing plate on its non-absorbing side, which
    the solver imposes as a boundary condition on that plane rather than as a structure.
    """

    # get pec frame bounding box and object's axis
    (box, axis, _) = self._pec_frame_box(obj)

    surfaces = Box.surfaces(box.size, box.center)
    del surfaces[2 * axis : 2 * axis + 2]

    return Structure(
        geometry=GeometryGroup(
            geometries=surfaces,
        ),
        medium=PECMedium(),
        name=name,
    )


def _pec_frame_span_inds(
    self: Any,
    obj: AbstractModeSource | InternalAbsorber,
) -> tuple[np.ndarray, int, str]:
    """Return grid-boundary index ranges ``[[beg, end], ...]`` the PEC frame covers,
    its frame axis, and the object's direction.

    Tangential axes use the shared snapped mode domain so the returned indices
    match where the mode-solver PEC boundaries are actually placed; the injection
    axis uses ``discretize_inds`` extended by ``frame.length`` cells for mode sources.
    """
    direction = obj.direction
    if isinstance(obj, AbstractModeSource):
        axis = obj.injection_axis
    else:
        axis = obj.size.index(0.0)

    snapped = snapped_mode_domain(self.grid, obj, axis)
    coords = self.grid.boundaries.to_list

    span_inds = np.zeros((3, 2), dtype=int)
    for dim in range(3):
        if dim == axis:
            continue
        span_inds[dim] = [
            find_snap_location(coords[dim], snapped.bounds[0][dim], "lower"),
            find_snap_location(coords[dim], snapped.bounds[1][dim], "upper"),
        ]

    ind_min, ind_max = self.grid.discretize_inds(obj, relax_precision=True)[axis]
    if isinstance(obj, AbstractModeSource):
        length = obj.frame.length
        if direction == "+":
            ind_max += length - 1
        else:
            ind_min -= length - 1
    span_inds[axis] = [ind_min, ind_max]

    return span_inds, axis, direction


def _pec_frame_box(self: Any, obj: AbstractModeSource | InternalAbsorber) -> tuple[Box, int, str]:
    """Return pec bounding box, frame axis and object's direction."""
    span_inds, axis, direction = self._pec_frame_span_inds(obj)
    coords = self.grid.boundaries.to_list
    box_bounds = [
        [coords[dim][span_inds[dim][0]], coords[dim][span_inds[dim][1]]] for dim in range(3)
    ]
    return Box.from_bounds(*np.transpose(box_bounds)), axis, direction


@cached_property
def _modal_plane_frames(self: Any) -> list[Structure]:
    """Return frames to add around mode sources and internal absorbers."""

    pec_frames = [
        self._make_pec_frame(src, name=f"{constants.MODAL_PEC_FRAME_NAME_PREFIX}source_{src_index}")
        for src_index, src in enumerate(self.sources)
        if isinstance(src, AbstractModeSource) and isinstance(src.frame, PECFrame)
    ]

    for absorber_index, absorber in enumerate(self._shifted_internal_absorbers):
        pec_frames.append(
            self._make_pec_frame(
                absorber,
                name=f"{constants.MODAL_PEC_FRAME_NAME_PREFIX}absorber_{absorber_index}",
            )
        )

    return pec_frames


@cached_property
def _finalized(self: Any) -> Self:
    """Return the finalized version of the simulation setup. That is, including automatic frames around mode sources and internal absorbers, and 2d strutures converted into volumetric analogues."""
    if len(self._modal_plane_frames) == 0 and not self._contains_converted_volumetric_structures:
        return self
    return self.updated_copy(
        grid_spec=GridSpec.from_grid(self.grid),
        structures=self._finalized_volumetric_structures,
    )


@cached_property
def _finalized_volumetric_structures(self: Any) -> list[Structure]:
    """Volumetric structures in the simulation, including automatic frames around mode sources and internal absorbers, and 2d strutures converted into volumetric analogues."""
    modal_frames = self._modal_plane_frames
    if not self._contains_converted_volumetric_structures:
        return list(self.static_structures) + modal_frames
    return list(self.volumetric_structures) + modal_frames


@cached_property
def _finalized_optical_medium_map(self: Any) -> dict[MediumType, NonNegativeInt]:
    """Returns dict mapping medium to index in material in finalized simulation.

    Returns
    -------
    Dict[:class:`.AbstractMedium`, int]
        Mapping between distinct mediums to index in finalized simulation.
    """
    medium_set = {structure._optical_medium for structure in self._finalized_volumetric_structures}
    medium_set.add(Structure._get_optical_medium(self.medium))
    return {medium: index for index, medium in enumerate(medium_set)}


def _validate_finalized(self: Any) -> None:
    """Validate that after adding pec frames simulation setup is still valid."""

    try:
        _ = self._finalized
    except Exception as e:
        raise Tidy3dError(
            format_chained_exception_message(
                "Simulation fails after requested mode source PEC frames are added.", e
            )
        ) from e


def _validate_no_bloch_with_modal_decomposition(self: Any) -> Self:
    """Reject Bloch boundaries combined with ``ModeTimeMonitor`` or multi-mode
    ``ModeABCBoundary``. ``Periodic`` boundaries and single-mode
    ``ModeABCBoundary`` remain supported.
    """
    if not any(isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list):
        return self
    message = (
        "Bloch boundaries are not supported in combination with "
        "'ModeTimeMonitor' or multi-mode 'ModeABCBoundary'. "
        "Use 'Periodic' boundaries instead, or remove the "
        "modal-decomposition feature."
    )
    for idx, monitor in enumerate(self.monitors):
        if isinstance(monitor, ModeTimeMonitor):
            self._raise_validation_error_at_loc(message, "monitors", idx)
    for idx, absorber in enumerate(self.internal_absorbers or []):
        boundary_spec = absorber.boundary_spec
        if isinstance(boundary_spec, ModeABCBoundary) and boundary_spec._num_absorbed_modes > 1:
            self._raise_validation_error_at_loc(message, "internal_absorbers", idx)
    return self


def _validate_internal_absorber_placement(self: Any) -> Self:
    """Require an ``InternalAbsorber`` to sit strictly inside the domain along its normal.

    Every absorber is closed by a PEC backing plate one cell to its non-absorbing side, so an
    absorber on a domain face puts that plate outside the domain. A ``ModeABCBoundary`` also
    solves modes on a plane, which needs grid cells around it: single-mode absorption solves on
    ``boundary_spec.plane`` as given, while multi-mode absorption solves one cell upstream of the
    absorber -- below itself for a ``"+"`` absorber, above for a ``"-"`` one -- and so needs that
    cell to exist AND to be interior, which is why it asks for
    ``MULTIMODE_MODE_ABC_NORMAL_CLEARANCE_CELLS`` cells rather than one.
    """
    for idx, absorber in enumerate(self._shifted_internal_absorbers):
        span_inds, axis, direction = self._pec_frame_span_inds(absorber)
        dim = "xyz"[axis]
        sim_min = self.center[axis] - self.size[axis] / 2
        sim_max = self.center[axis] + self.size[axis] / 2
        if sim_max <= sim_min:
            continue

        # The PEC backing plate closes the frame on the absorber's non-absorbing side: the cell
        # just past the frame for a "+" absorber, the frame's own cell for a "-" one. On a domain
        # face the "+" plate lands outside the grid, where the solver has only a ghost cell.
        num_cells = int(self.grid.num_cells[axis])
        plate_cell = int(span_inds[axis][1]) if direction == "+" else int(span_inds[axis][0])
        if not 0 <= plate_cell < num_cells:
            self._raise_validation_error_at_loc(
                f"An 'InternalAbsorber' must leave room for its PEC backing plate, which sits on "
                f"its non-absorbing side, but this absorber is against the '{dim}' domain "
                "boundary and the plate would fall outside the simulation. Move it one cell "
                "further in, or use a domain boundary condition to absorb at the edge.",
                "internal_absorbers",
                idx,
            )

        boundary_spec = absorber.boundary_spec
        if not isinstance(boundary_spec, ModeABCBoundary):
            continue

        if boundary_spec._num_absorbed_modes > 1:
            clearance = constants.MULTIMODE_MODE_ABC_NORMAL_CLEARANCE_CELLS
            if direction == "+":
                gap, side = int(span_inds[axis][0]), f"-{dim}"
            else:
                gap = num_cells - int(span_inds[axis][1])
                side = f"+{dim}"
            if gap < clearance:
                self._raise_validation_error_at_loc(
                    f"A multi-mode 'ModeABCBoundary' needs at least {clearance} grid cells "
                    f"between its absorber and the '{side}' domain boundary, but this absorber "
                    f"has {gap}. Move it further into the domain, enlarge the simulation along "
                    f"'{dim}', or absorb a single mode.",
                    "internal_absorbers",
                    idx,
                )
        else:
            plane_pos = boundary_spec.plane.center[axis]
            if not sim_min < plane_pos < sim_max:
                self._raise_validation_error_at_loc(
                    "The mode plane of a 'ModeABCBoundary' absorber lies on the simulation "
                    f"boundary along its normal direction, at {plane_pos} with simulation "
                    f"bounds ({sim_min}, {sim_max}). Please move the plane inside the simulation.",
                    "internal_absorbers",
                    idx,
                )
    return self


def _validate_mode_objects(self: Any) -> None:
    """Apply the mode-solver setup checks to each modal source and monitor."""

    def validate_mode_sort_spec_bounding_box(
        mode_obj: ModeSource | AbstractModeMonitor, validation_loc: tuple[str, int]
    ) -> None:
        """Validate a fill-fraction box against the effective mode plane."""
        sort_spec = getattr(mode_obj.mode_spec, "sort_spec", None)
        if sort_spec is None:
            return

        mode_plane_bounds = Box.bounds_intersection(mode_obj.bounds, self.bounds)
        normal_axis = mode_obj.geometry.zero_dims[0]
        if not sort_spec._bounding_box_intersects_tangentially(mode_plane_bounds, normal_axis):
            self._raise_validation_error_at_loc(
                "'ModeSortSpec.bounding_box' must intersect the effective mode plane along "
                "both tangential axes. Please move or resize the bounding box in the "
                "tangential directions.",
                *validation_loc,
                "mode_spec",
                "sort_spec",
                "bounding_box",
            )

    def validate_mode_object(mode_obj: ModeSource | AbstractModeMonitor, msg_prefix: str) -> None:
        # Warn if pml is too thick
        warn_thick_pml(
            simulation=self,
            plane=mode_obj.geometry,
            mode_spec=mode_obj.mode_spec,
            msg_prefix=msg_prefix,
        )
        # Error if mode plane radius is too small
        validate_mode_plane_radius(
            mode_spec=mode_obj.mode_spec,
            plane=mode_obj.geometry,
            sim_geom=self.geometry,
        )
        # Test if structures can be rotated if ``angle_rotation=True``
        theta = mode_obj.mode_spec.angle_theta
        if np.abs(theta) > 0 and mode_obj.mode_spec.angle_rotation:
            structs_in = Scene.intersecting_structures(mode_obj.geometry, self.structures)
            total_structures = [
                self.scene.background_structure,
                *list(self.volumetric_structures),
            ]
            mediums_in = list(Scene.intersecting_media(mode_obj.geometry, total_structures))
            translate_kwargs = rotation_translate_kwargs(mode_obj.geometry, mode_obj.mode_spec)
            rotate_kwargs_value = rotation_kwargs(mode_obj.geometry, mode_obj.mode_spec)
            validation_freqs = rotation_validation_freqs(mode_obj)
            validate_plane_rotation_media(
                mediums=mediums_in,
                rotate_kwargs=rotate_kwargs_value,
                freqs=validation_freqs,
            )
            make_rotated_structures(
                structs_in,
                translate_kwargs,
                rotate_kwargs_value,
                validation_freqs,
            )
        # Validate microwave mode spec with mode solver setup
        if isinstance(mode_obj.mode_spec, MicrowaveModeSpec):
            validate_microwave_mode_spec(
                mode_spec=mode_obj.mode_spec,
                plane=effective_mode_plane(mode_obj.geometry, self.geometry),
            )

    for imnt, monitor in enumerate(self.monitors):
        if isinstance(monitor, (AbstractModeMonitor, ModeTimeMonitor)):
            validate_mode_sort_spec_bounding_box(monitor, ("monitors", imnt))
            try:
                validate_mode_object(mode_obj=monitor, msg_prefix=f"'monitors[{imnt}]'")
            except Exception as e:
                self._raise_validation_error_at_loc(
                    format_chained_exception_message(
                        f"Monitor at 'monitors[{imnt}]' failed validation", e
                    ),
                    "monitors",
                    imnt,
                )

    for isrc, source in enumerate(self.sources):
        if isinstance(source, AbstractModeSource):
            validate_mode_sort_spec_bounding_box(source, ("sources", isrc))
            try:
                validate_mode_object(mode_obj=source, msg_prefix=f"'sources[{isrc}]'")
            except Exception as e:
                self._raise_validation_error_at_loc(
                    format_chained_exception_message(
                        f"Source at 'sources[{isrc}]' failed validation", e
                    ),
                    "sources",
                    isrc,
                )


def _validate_modes_size(self: Any) -> None:
    """Warn if mode sources or monitors have a large number of points."""

    def warn_mode_size(
        monitor: AbstractModeMonitor | ModeTimeMonitor, msg_header: str, custom_loc: list
    ) -> None:
        """Warn if a mode component has a large number of points."""
        num_cells = np.prod(self.discretize_monitor(monitor).num_cells)
        if num_cells > constants.WARN_MODE_NUM_CELLS:
            consolidated_logger.warning(
                msg_header + f"has a large number ({num_cells:1.2e}) of grid points. "
                "This can lead to solver slow-down and increased cost. "
                "Consider making the size of the component smaller, as long as the modes "
                "of interest decay by the plane boundaries.",
                custom_loc=custom_loc,
            )

    with log as consolidated_logger:
        for src_ind, source in enumerate(self.sources):
            if isinstance(source, AbstractModeSource):
                # Make a monitor so we can call ``discretize_monitor``
                monitor = FieldMonitor(
                    center=source.center,
                    size=source.size,
                    name="tmp",
                    freqs=[source.source_time._freq0],
                    colocate=False,
                )
                msg_header = f"Mode source at sources[{src_ind}] "
                custom_loc = ["sources", src_ind]
                warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    with log as consolidated_logger:
        for mnt_ind, monitor in enumerate(self.monitors):
            if isinstance(monitor, (AbstractModeMonitor, ModeTimeMonitor)):
                msg_header = f"Mode monitor '{monitor.name}' "
                custom_loc = ["monitors", mnt_ind]
                warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)


def _validate_num_cells_in_mode_objects(self: Any) -> None:
    """Raise an error if mode sources or monitors intersect with a very small number
    of grid cells in their transverse dimensions."""

    def check_num_cells(
        mode_object: tuple[ModeSource, ModeMonitor], normal_axis: Axis, msg_header: str
    ) -> None:
        disc_grid = self.discretize(mode_object)
        _, check_axes = Box.pop_axis([0, 1, 2], axis=normal_axis)
        for axis in check_axes:
            sim_size = self.size[axis]
            dim_cells = disc_grid.num_cells[axis]
            if sim_size > 0 and dim_cells <= 2:
                small_dim = "xyz"[axis]
                raise SetupError(
                    msg_header + f"is too small along the "
                    f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                    f"Increase the size of the object along '{small_dim}'."
                )

    for source in self.sources:
        if isinstance(source, AbstractModeSource):
            msg_header = f"Mode source '{source.name}' "
            check_num_cells(source, source.injection_axis, msg_header)

    for monitor in self.monitors:
        if isinstance(monitor, (ModeMonitor, ModeTimeMonitor)):
            msg_header = f"Mode monitor '{monitor.name}' "
            check_num_cells(monitor, monitor.normal_axis, msg_header)


@cached_property
def complex_fields(self: Any) -> bool:
    """Whether complex fields are used in the simulation.

    Triggers on Bloch boundaries, complex-fields nonlinear models, or a
    ``ModeTimeMonitor`` on a simulation that contains a lossy medium.

    Returns
    -------
    bool
        Whether the time-stepping fields are real or complex.
    """
    if any(isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list):
        return True
    for medium in self.scene.mediums:
        if medium.nonlinear_spec is not None:
            if any(model.complex_fields for model in medium._nonlinear_models):
                return True
    if self._has_lossy_mode_decomposition_feature:
        return True
    return False


@cached_property
def _has_lossy_mode_decomposition_feature(self: Any) -> bool:
    """True iff the simulation contains a ``ModeTimeMonitor`` or multi-mode
    ``ModeABCBoundary`` and any sim medium can be lossy.
    """
    has_mtm = any(isinstance(m, ModeTimeMonitor) for m in self.monitors)
    # Single-element ``mode_index`` (``=i`` or ``=(i,)``) absorbs only one mode, so no
    # decomposition runs; gate on the count to avoid enabling ``complex_fields`` (~2x
    # cost) when no decomposition is performed.
    has_multimode_abc = any(
        isinstance(ia.boundary_spec, ModeABCBoundary) and ia.boundary_spec._num_absorbed_modes > 1
        for ia in (self.internal_absorbers or [])
    )
    if not (has_mtm or has_multimode_abc):
        return False
    # `scene.mediums` covers structure mediums; the background medium is
    # held separately on the simulation, so include it explicitly.
    if _medium_can_be_lossy(self.medium):
        return True
    return any(_medium_can_be_lossy(medium) for medium in self.scene.mediums)
