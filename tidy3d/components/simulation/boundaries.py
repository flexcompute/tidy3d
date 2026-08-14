"""Boundary validation, PML geometry, and symmetry helpers for Yee-grid simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import (
    model_validator,
)

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import (
    CLIPPING_MARGIN,
    PML,
    ABCBoundary,
    Absorber,
    AbsorberSpec,
    BlochBoundary,
    Boundary,
    ModeABCBoundary,
    Periodic,
    StablePML,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.utils import _shift_object
from tidy3d.components.medium import AnisotropicMedium, FullyAnisotropicMedium
from tidy3d.components.monitor import DiffractionMonitor
from tidy3d.components.scene import Scene
from tidy3d.components.source.field import AbstractModeSource, FixedAngleSpec, PlaneWave
from tidy3d.components.source.frame import PECFrame
from tidy3d.components.structure import Structure
from tidy3d.components.validators import named_obj_descr
from tidy3d.constants import C_0
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.compat import Self
    from tidy3d.components.boundary import BoundaryEdgeType, BoundarySpec, InternalAbsorber
    from tidy3d.components.medium import MediumType, MediumType3D
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.types import Axis

from . import constants


def validate_boundaries_for_zero_dims(
    warn_on_change: bool = True,
) -> Callable[[Any], Any]:
    """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""

    @model_validator(mode="after")
    def boundaries_for_zero_dims(self: Any) -> Any:
        """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""
        val = self.boundary_spec
        boundaries = val.to_list
        size = self.size
        symmetry = self.symmetry
        axis_names = "xyz"

        for dim, (boundary, symmetry_dim, size_dim) in enumerate(zip(boundaries, symmetry, size)):
            if size_dim == 0:
                axis = axis_names[dim]
                num_absorbing_bdries = sum(
                    isinstance(bnd, AbsorberSpec | ABCBoundary | ModeABCBoundary)
                    for bnd in boundary
                )
                num_bloch_bdries = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)

                if num_absorbing_bdries > 0:
                    pbc = Boundary(minus=Periodic(), plus=Periodic())
                    val = val.updated_copy(**{axis: pbc})
                    if warn_on_change:
                        log.warning(
                            f"The simulation has zero size along the {axis} axis, so "
                            "using a PML or absorbing boundary along that axis is incorrect. "
                            f"Use either 'Periodic' or 'BlochBoundary' along {axis}. "
                            "Using 'Periodic' boundary by default."
                        )

                if num_bloch_bdries > 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, "
                        "using a Bloch boundary along such an axis is not supported because of "
                        "the Bloch vector definition in units of '2 * pi / (size along dimension)'. Use a small "
                        "but nonzero size along the dimension instead.",
                        "boundary_spec",
                        axis,
                    )

                if symmetry_dim != 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, so "
                        "using symmetry along that axis is incorrect. Use 'PECBoundary' "
                        "or 'PMCBoundary' to select source polarization if needed and set "
                        f"Simulation.symmetry to 0 along {axis}.",
                        "symmetry",
                        dim,
                    )

                if boundary[0] != boundary[1]:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis. "
                        f"The boundary condition for {axis} plus and {axis} "
                        "minus must be the same.",
                        "boundary_spec",
                        axis,
                    )

        # Update boundary_spec if it was modified
        if val != self.boundary_spec:
            object.__setattr__(self, "boundary_spec", val)

        return self

    return boundaries_for_zero_dims


def _validate_boundary_spec_symmetry(self: Any) -> Self:
    """Error if symmetry is imposed along an axis but the boundary conditions are not the same
    on both sides."""

    def equivalent(plus: BoundarySpec, minus: BoundarySpec) -> bool:
        """Returns whether two boundary conditions are physically identical."""
        # Make copies of `plus` and `minus` with the `name` attribute set to "".
        plus_cpy = plus.updated_copy(name="")
        minus_cpy = minus.updated_copy(name="")
        return plus_cpy == minus_cpy

    bs = self.boundary_spec
    boundaries = [bs.x, bs.y, bs.z]
    for ax, symmetry, ax_bounds in zip("xyz", self.symmetry, boundaries):
        if symmetry != 0 and not equivalent(ax_bounds.plus, ax_bounds.minus):
            self._raise_validation_error_at_loc(
                f"Symmetry '{symmetry}' along axis {ax} requires the same boundary "
                f"condition on both sides of the axis.",
                "boundary_spec",
                ax,
            )
    return self


@cached_property
def _shifted_internal_absorbers(self: Any) -> list[InternalAbsorber]:
    """List of absorber shifted to their actual locations based on their grid_shift's."""

    return [
        _shift_object(
            obj=absorber,
            grid=self.grid,
            bounds=self.bounds,
            direction=absorber.direction,
            shift=absorber.grid_shift,
        )
        for absorber in self.internal_absorbers
    ]


@cached_property
def bounds_pml(self: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Simulation bounds including the PML regions."""
    log.warning(
        "'Simulation.bounds_pml' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.simulation_bounds' instead."
    )
    return self.simulation_bounds


@cached_property
def simulation_bounds(self: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Simulation bounds including the PML regions."""
    pml_thick = self.pml_thicknesses
    bounds_in = self.bounds
    bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
    bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

    return (bounds_min, bounds_max)


def _make_pml_boxes(self: Any, normal_axis: Axis) -> list[Box]:
    """make a list of Box objects representing the pml to plot on plane."""
    pml_boxes = []
    pml_thicks = self.pml_thicknesses
    for pml_axis, num_layers_dim in enumerate(self.num_pml_layers):
        if pml_axis == normal_axis:
            continue
        for sign, pml_height, num_layers in zip((-1, 1), pml_thicks[pml_axis], num_layers_dim):
            if num_layers == 0:
                continue
            pml_box = self._make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
            pml_boxes.append(pml_box)
    return pml_boxes


def _make_pml_box(self: Any, pml_axis: Axis, pml_height: float, sign: int) -> Box:
    """Construct a :class:`.Box` representing an arborbing boundary to be plotted."""
    rmin, rmax = (list(bounds) for bounds in self.simulation_bounds)
    if sign == -1:
        rmax[pml_axis] = rmin[pml_axis] + pml_height
    else:
        rmin[pml_axis] = rmax[pml_axis] - pml_height
    pml_box = Box.from_bounds(rmin=rmin, rmax=rmax)

    # if any dimension of the sim has size 0, set the PML to a very small size along that dim
    new_size = list(pml_box.size)
    for dim_index, sim_size in enumerate(self.size):
        if sim_size == 0.0:
            new_size[dim_index] = constants.PML_HEIGHT_FOR_0_DIMS
    pml_box = pml_box.updated_copy(size=tuple(new_size))

    return pml_box


@cached_property
def pml_thicknesses(self: Any) -> list[tuple[float, float]]:
    """Thicknesses (um) of absorbers in all three axes and directions (-, +)

    Returns
    -------
    list[Tuple[float, float]]
        List containing the absorber thickness (micron) in - and + boundaries.
    """
    num_layers = self.num_pml_layers
    pml_thicknesses = []
    for num_layer, boundaries in zip(num_layers, self.grid.boundaries.to_list):
        thick_l = boundaries[num_layer[0]] - boundaries[0]
        thick_r = boundaries[-1] - boundaries[-1 - num_layer[1]]
        pml_thicknesses.append((thick_l, thick_r))

    return pml_thicknesses


def _pml_extrusion_clipping_bound_ind(self: Any, axis: int, side: int) -> int | None:
    """Grid-boundary index where the PML extrusion clipping inset reaches into the domain.

    The "extrusion region" spans from the outer simulation boundary through the PML and an
    additional :data:`CLIPPING_MARGIN` cells of interior. This method returns the inner-most
    boundary index of that region (i.e., ``num_layers + CLIPPING_MARGIN`` on the minus side
    and the analogous index counting from the far end on the plus side).

    Returns ``None`` when there are no absorber layers on that side or the computed index
    falls outside the grid. This helper does not check whether ``extrude_structures`` is
    actually enabled — callers that only care when extrusion is active must gate separately.
    """
    n_layers = self.num_pml_layers[axis][side]
    if n_layers == 0:
        return None
    n_bounds = len(self.grid.boundaries.to_list[axis])
    if side == 0:
        idx = n_layers + CLIPPING_MARGIN
        return idx if idx < n_bounds else None
    idx = n_bounds - 1 - n_layers - CLIPPING_MARGIN
    return idx if idx >= 0 else None


@cached_property
def _periodic(self: Any) -> tuple[bool, bool, bool]:
    """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
    We check on both sides but in practice there should be no cases in which a periodic/Bloch
    BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
    periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
    boundary with another PEC/PMC plane upon initialization."""
    periodic = []
    for bcs_1d in self.boundary_spec.to_list:
        periodic.append(all(isinstance(bcs, Periodic | BlochBoundary) for bcs in bcs_1d))
    return periodic


@cached_property
def num_pml_layers(self: Any) -> list[tuple[float, float]]:
    """Number of absorbing layers in all three axes and directions (-, +).

    Returns
    -------
    list[tuple[float, float]]
        List containing the number of absorber layers in - and + boundaries.
    """
    num_layers = [[0, 0], [0, 0], [0, 0]]

    for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
        for idx_j, boundary in enumerate(boundary1d):
            if isinstance(boundary, PML | StablePML | Absorber):
                num_layers[idx_i][idx_j] = boundary.num_layers

    return num_layers


def _structures_not_at_edges(self: Any) -> Self:
    """Override :class:`.AbstractSimulation` validator for :class:`.Simulation`.

    The edge check is handled by :meth:`._validate_structures_not_at_edges`, which is called
    from :meth:`._validate_scene` and can consider `boundary_spec` extrusion settings.
    """
    return self


def _bloch_with_symmetry(self: Any) -> Self:
    """Error if a Bloch boundary is applied with symmetry"""
    val = self.boundary_spec
    boundaries = val.to_list
    symmetry = self.symmetry
    for dim, boundary in enumerate(boundaries):
        num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
        if num_bloch > 0 and symmetry[dim] != 0:
            self._raise_validation_error_at_loc(
                f"Bloch boundaries cannot be used with a symmetry along dimension {dim}.",
                "boundary_spec",
                "xyz"[dim],
            )
    return self


def _bloch_boundaries_diff_mnt(self: Any) -> Self:
    """Error if there are diffraction monitors incompatible with boundary conditions."""

    monitors = self.monitors

    if not monitors or not any(isinstance(mnt, DiffractionMonitor) for mnt in monitors):
        return self

    boundaries = self.boundary_spec.to_list
    sources = self.sources
    size = self.size
    sim_medium = self.medium
    structures = self.structures
    for source_ind, source in enumerate(sources):
        if not isinstance(source, PlaneWave):
            continue

        if isinstance(source.angular_spec, FixedAngleSpec):
            continue

        _, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
        medium_set = Scene.intersecting_media(source, structures)
        medium = medium_set.pop() if medium_set else sim_medium

        for tan_dir in tan_dirs:
            boundary = boundaries[tan_dir]

            # check the Bloch boundary + angled plane wave case
            num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
            if num_bloch > 0:
                self._check_bloch_vec(
                    source=source,
                    source_ind=source_ind,
                    bloch_vec=boundary[0].bloch_vec,
                    dim=tan_dir,
                    medium=medium,
                    domain_size=size[tan_dir],
                    has_diff_mnt=True,
                )
    return self


def _validate_frequency_mode_abc(self: Any) -> Self:
    """Warn if ModeABCBoundary expects a frequency from a source, but there are multiple sources with different central frequencies."""

    def boundary_needs_freq(
        boundary: ModeABCBoundary | ABCBoundary | BoundaryEdgeType,
    ) -> bool:
        return (isinstance(boundary, ModeABCBoundary) and boundary.freq_spec is None) or (
            isinstance(boundary, ABCBoundary)
            and (
                (boundary.conductivity is not None and boundary.conductivity != 0)
                or (boundary.permittivity is None and boundary.conductivity is None)
            )
        )

    # check domain boundaries
    boundaries = self.boundary_spec.to_list
    need_wavelength = any(boundary_needs_freq(edge) for edge in np.ravel(boundaries))

    # check dinternal absorbers
    need_wavelength = need_wavelength or any(
        boundary_needs_freq(abc.boundary_spec) for abc in self.internal_absorbers
    )

    if need_wavelength:
        self._check_source_freq_available(
            no_source_error=(
                "At least one 'ModeABCBoundary'/'ABCBoundary' needs specification of frequency at which the absorbed mode must be evaluated. "
                "Add at least one source or use parameter 'frequency' for 'ModeABCBoundary'."
            ),
            no_source_loc=("sources",),
            multi_freq_warning=(
                "At least one 'ModeABCBoundary' does not specify frequency at which the absorbed mode must be evaluated. "
                "The central frequency of the first source will be used."
            ),
        )

    return self


def _validate_absorber_in_zero_dims(self: Any) -> Self:
    """Error if internal absorber is oriented along zero size dim."""
    val = self.internal_absorbers
    if val is None:
        return val

    sim_size = self.size
    for abc_index, abc in enumerate(val):
        if sim_size[abc._normal_axis] == 0:
            self._raise_validation_error_at_loc(
                "Port absorbers are not allowed to be oriented along simulation zero size dimensions.",
                "internal_absorbers",
                abc_index,
            )

    return self


@classmethod
def _get_mediums_on_abc(
    cls: Any,
    boundary_spec: BoundarySpec,
    sim_structure: Structure,
    structures: tuple[Structure, ...],
) -> tuple[
    list[MediumType3D],
    list[MediumType3D],
    list[MediumType3D],
    list[MediumType3D],
    list[MediumType3D],
    list[MediumType3D],
]:
    """For each ABC boundary that needs an automatic medium detection (permittivity=None)
    determine mediums it crosses.
    """

    # list of structures including background as a Box()
    surface_box = sim_structure.geometry
    # expand zero dimensions to make sure surface are extracted correctly and treatment is uniform
    surface_box = surface_box.updated_copy(size=[1e-6 if s == 0 else s for s in surface_box.size])
    surfaces = Box.surfaces(center=surface_box.center, size=surface_box.size)

    total_structures = [sim_structure, *list(structures)]

    mediums = []
    for boundary, surface in zip(np.ravel(boundary_spec.to_list), surfaces):
        if isinstance(boundary, ABCBoundary) and boundary.permittivity is None:
            mediums.append(Scene.intersecting_media(surface, total_structures))
        else:
            mediums.append(None)

    return mediums


def _abc_boundaries_homogeneous(self: Any) -> Self:
    """Error if abc boundaries intersect multiple mediums or anisotropic mediums."""
    val = self.boundary_spec
    if val is None:
        return val

    sim_structure = Structure(
        geometry=Box(size=self.size, center=self.center),
        medium=self.medium,
    )

    mediums_all_sides = self._get_mediums_on_abc(
        boundary_spec=val,
        sim_structure=sim_structure,
        structures=self.structures or [],
    )
    boundary_locs = [
        ("x", "minus"),
        ("x", "plus"),
        ("y", "minus"),
        ("y", "plus"),
        ("z", "minus"),
        ("z", "plus"),
    ]

    with log as consolidated_logger:
        for (axis_name, side_name), mediums in zip(boundary_locs, mediums_all_sides):
            if mediums is not None:
                # make sure there is no more than one medium in the returned list
                if len(mediums) > 1:
                    self._raise_validation_error_at_loc(
                        f"{len(mediums)} different mediums detected on an 'ABCBoundary'. Boundary must be homogeneous."
                        "Alternatively, effective permeability and conductivity can be directly provided as "
                        "parameters for an 'ABCBoundary', in which case this medium check is skipped.",
                        "boundary_spec",
                        axis_name,
                        side_name,
                    )
                # 0 medium, something is wrong
                if len(mediums) < 1:
                    self._raise_validation_error_at_loc(
                        "No medium detected on plane containing 'ABCBoundary', "
                        "indicating an unexpected error. Please create a github issue so "
                        "that the problem can be investigated.",
                        "boundary_spec",
                        axis_name,
                        side_name,
                    )
                # 1 medium, check if the medium is spatially uniform
                if not list(mediums)[0].is_spatially_uniform:
                    consolidated_logger.warning(
                        "Nonuniform custom medium detected on an 'ABCBoundary'. "
                        "Boundary must be homogeneous. Make sure custom medium is uniform on the boundary.",
                    )

                if isinstance(list(mediums)[0], AnisotropicMedium | FullyAnisotropicMedium):
                    self._raise_validation_error_at_loc(
                        "An anisotropic medium is detected on an 'ABCBoundary'. "
                        "Boundary medium must be homogeneous and isotropic.",
                        "boundary_spec",
                        axis_name,
                        side_name,
                    )

    return self


def _validate_no_structures_pml(self: Any) -> None:
    """Ensure no structures terminate / have bounds inside of PML."""

    pml_thicks = np.array(self.pml_thicknesses).T
    sim_bounds = self.bounds
    bound_spec = self.boundary_spec.to_list

    with log as consolidated_logger:
        for i, structure in enumerate(self.static_structures):
            geo_bounds = structure.geometry.bounds
            warn = False  # will only warn once per structure
            for sim_bound, geo_bound, pml_thick, bound_dim, pm_val in zip(
                sim_bounds, geo_bounds, pml_thicks, bound_spec, (-1, 1)
            ):
                for sim_pos, geo_pos, pml, bound_edge in zip(
                    sim_bound, geo_bound, pml_thick, bound_dim
                ):
                    sim_pos_pml = sim_pos + pm_val * pml
                    in_pml_plus = (pm_val > 0) and (sim_pos < geo_pos <= sim_pos_pml)
                    in_pml_mnus = (pm_val < 0) and (sim_pos > geo_pos >= sim_pos_pml)
                    if (
                        not isinstance(bound_edge, Absorber)
                        and (in_pml_plus or in_pml_mnus)
                        and (
                            not hasattr(bound_edge, "extrude_structures")
                            or not bound_edge.extrude_structures
                        )
                    ):
                        warn = True
            if warn:
                obj_descr = named_obj_descr(structure, "structures", i)
                consolidated_logger.warning(
                    f"A bound of {obj_descr} was detected as being "
                    "within the simulation PML. We recommend extending structures to "
                    "infinity or completely outside of the simulation PML to avoid "
                    "unexpected effects when the structures are not translationally "
                    "invariant within the PML.",
                    custom_loc=["structures", i],
                )


def _validate_no_structures_close_to_pml(self: Any) -> None:
    """Warn if structures are too close to PML boundaries and may be automatically extruded."""
    if not self.structures or not self.sources:
        return

    sim_bound_min, sim_bound_max = self.bounds
    boundaries = self.boundary_spec.to_list

    # Access grid - this will compute it once and cache it
    grid_boundaries = self.grid.boundaries.to_list
    num_pml_layers = self.num_pml_layers

    def is_within_clipping_margin(axis_idx: int, struct_val: float, side_idx: int) -> bool:
        """Check if ``struct_val`` falls inside the ``CLIPPING_MARGIN`` inset, i.e. between
        the absorber-domain interface and the inner edge of the extrusion region."""
        clipping_bound_idx = self._pml_extrusion_clipping_bound_ind(axis_idx, side_idx)
        if clipping_bound_idx is None:
            return False
        grid_axis = grid_boundaries[axis_idx]
        num_layers = num_pml_layers[axis_idx][side_idx]
        if side_idx == 0:
            absorber_start_coord = grid_axis[num_layers]
            clipping_bound_coord = grid_axis[clipping_bound_idx]
            return absorber_start_coord <= struct_val <= clipping_bound_coord
        absorber_start_coord = grid_axis[len(grid_axis) - num_layers - 1]
        clipping_bound_coord = grid_axis[clipping_bound_idx]
        return clipping_bound_coord <= struct_val <= absorber_start_coord

    with log as consolidated_logger:

        def warn(structure: Structure, istruct: int, side: str, extrusion_flag: bool) -> None:
            """Warn when a structure is within half a wavelength of a PML boundary.
            If ``extrusion_flag`` is True, warns about automatic extrusion. Otherwise, warns about
            potential inaccuracies and suggests increasing the gap or extending the structure.
            """
            obj_descr = named_obj_descr(structure, "structures", istruct)

            if extrusion_flag:
                consolidated_logger.warning(
                    f"Structure: {obj_descr} was detected as being less "
                    f"than half of a central wavelength from a PML on side {side}. "
                    "The structure will be automatically extruded to the end of the PML region "
                    "to ensure translational invariance.",
                    custom_loc=["structures", istruct],
                )
            else:
                consolidated_logger.warning(
                    f"Structure: {obj_descr} was detected as being less "
                    f"than half of a central wavelength from a PML on side {side}. "
                    "To avoid inaccurate results or divergence, please increase gap between "
                    "any structures and PML or fully extend structure through the pml.",
                    custom_loc=["structures", istruct],
                )

        for istruct, structure in enumerate(self.structures):
            struct_bound_min, struct_bound_max = structure.geometry.bounds

            for source in self.sources:
                lambda0 = C_0 / source.source_time._freq0

                # Check both min (side_idx=0) and max (side_idx=1) sides
                for side_idx in [0, 1]:
                    sim_bound_side = sim_bound_min if side_idx == 0 else sim_bound_max
                    struct_bound_side = struct_bound_min if side_idx == 0 else struct_bound_max
                    side_suffix = "-min" if side_idx == 0 else "-max"

                    zipped = zip(
                        ["x", "y", "z"],
                        [0, 1, 2],
                        sim_bound_side,
                        struct_bound_side,
                        boundaries,
                    )
                    for axis, axis_idx, sim_val, struct_val, boundary in zipped:
                        # The test is required only for PML and stable PML
                        if not isinstance(boundary[side_idx], PML | StablePML):
                            continue
                        # Min side: struct_val > sim_val, Max side: struct_val < sim_val
                        if (
                            boundary[side_idx].num_layers > 0
                            and (struct_val > sim_val if side_idx == 0 else struct_val < sim_val)
                            and abs(sim_val - struct_val) < lambda0 / 2
                        ):
                            extrusion_flag = boundary[
                                side_idx
                            ].extrude_structures and is_within_clipping_margin(
                                axis_idx, struct_val, side_idx
                            )
                            warn(structure, istruct, axis + side_suffix, extrusion_flag)


def _validate_pec_frame_not_in_pml_extrusion(self: Any) -> None:
    """Error if an automatically added PEC frame overlaps the PML extrusion region.

    Works in grid-boundary index space: each PEC frame spans ``[beg, end]`` along every axis,
    and each PML side with ``extrude_structures`` enabled forbids the index range covering
    the PML plus an additional ``CLIPPING_MARGIN`` cells of interior (the clipping inset).
    Touching counts as overlap.
    """
    # Collect auto-added PEC frame index spans alongside the field/loc they originate from.
    frames: list[tuple[np.ndarray, str, int, str]] = []
    for src_idx, src in enumerate(self.sources):
        if isinstance(src, AbstractModeSource) and isinstance(src.frame, PECFrame):
            span_inds, _, _ = self._pec_frame_span_inds(src)
            descr = f"mode source '{src.name}'" if src.name else f"mode source at index {src_idx}"
            frames.append((span_inds, "sources", src_idx, descr))
    for abs_idx, absorber in enumerate(self._shifted_internal_absorbers):
        span_inds, _, _ = self._pec_frame_span_inds(absorber)
        frames.append(
            (span_inds, "internal_absorbers", abs_idx, f"internal absorber at index {abs_idx}")
        )
    if not frames:
        return

    boundaries = self.boundary_spec.to_list
    grid_boundaries = self.grid.boundaries.to_list

    for axis in range(3):
        n_bounds = len(grid_boundaries[axis])
        for side in (0, 1):
            bnd = boundaries[axis][side]
            if not isinstance(bnd, AbsorberSpec) or not bnd.extrude_structures:
                continue
            clip_ind = self._pml_extrusion_clipping_bound_ind(axis, side)
            if clip_ind is None:
                continue
            ext_lo, ext_hi = (0, clip_ind) if side == 0 else (clip_ind, n_bounds - 1)
            for span_inds, field, loc, descr in frames:
                beg, end = span_inds[axis]
                if beg <= ext_hi and end >= ext_lo:
                    axis_label = "xyz"[axis]
                    side_label = f"{'-+'[side]}{axis_label}"
                    self._raise_validation_error_at_loc(
                        f"The automatically added PEC frame for {descr} overlaps the "
                        f"{bnd.type} extrusion region on the '{side_label}' boundary. "
                        f"The extrusion region extends {CLIPPING_MARGIN} grid cells beyond "
                        f"the {bnd.type} into the simulation domain; increase the simulation "
                        f"size along '{axis_label}', move the source/absorber away from "
                        f"that boundary, or disable 'extrude_structures' on that side.",
                        field,
                        loc,
                    )


def _validate_internal_abc_no_fully_anisotropic(self: Any) -> Self:
    """Error if internal absorber intersect fully anisotropic mediums."""

    total_structures = [self.scene.background_structure, *list(self.structures)]

    for abc_index, abc in enumerate(self._shifted_internal_absorbers):
        mediums = Scene.intersecting_media(abc, tuple(total_structures))

        if any(isinstance(med, FullyAnisotropicMedium) for med in mediums):
            self._raise_validation_error_at_loc(
                "A 'InternalAbsorber' cannot cross a 'FullyAnisotropicMedium'.",
                "internal_absorbers",
                abc_index,
            )
    return self


def _num_non_pml_cells(self: Any) -> int:
    """Number of grid cells in the simulation domain excluding PML/absorber layers."""
    non_pml_cells_dim = []
    for num_cells_dim, num_pml_layers_dim in zip(self.grid.num_cells, self.num_pml_layers):
        num_pml_cells_dim = num_pml_layers_dim[0] + num_pml_layers_dim[1]
        non_pml_cells_dim.append(num_cells_dim - num_pml_cells_dim)
    return int(np.prod(non_pml_cells_dim))


@staticmethod
def _check_bloch_vec(
    source: SourceType,
    source_ind: int,
    bloch_vec: float,
    dim: Axis,
    medium: MediumType,
    domain_size: float,
    has_diff_mnt: bool = False,
) -> None:
    """Helper to check if a given Bloch vector is consistent with a given source."""

    # make a dummy Bloch boundary to check for correctness
    dummy_bnd = BlochBoundary.from_source(
        source=source, domain_size=domain_size, axis=dim, medium=medium
    )
    expected_bloch_vec = dummy_bnd.bloch_vec

    if bloch_vec != expected_bloch_vec:
        test_val = np.real(expected_bloch_vec - bloch_vec)

        test_val_is_int = np.isclose(test_val, np.round(test_val))
        src_name = f" '{source.name}'" if source.name else ""

        if has_diff_mnt and test_val_is_int and not np.isclose(test_val, 0):
            # the given Bloch vector is offset by an integer
            log.warning(
                f"The wave vector of source{src_name} along dimension "
                f"'{dim}' is equal to the Bloch vector of the simulation "
                "boundaries along that dimension plus an integer reciprocal "
                "lattice vector. If using a 'DiffractionMonitor', diffraction "
                "order 0 will not correspond to the angle of propagation "
                "of the source. Consider using 'BlochBoundary.from_source()'.",
                custom_loc=["boundary_spec", "xyz"[dim]],
            )

        if not test_val_is_int:
            # the given Bloch vector is neither equal to the expected value, nor
            # off by an integer
            log.warning(
                f"The Bloch vector along dimension '{dim}' may be incorrectly "
                f"set with respect to the source{src_name}. The absolute "
                "difference between the expected and provided values in "
                "bandstructure units, up to an integer offset, is greater than "
                "1e-6. Consider using ``BlochBoundary.from_source()``, or "
                "double-check that it was defined correctly.",
                custom_loc=["boundary_spec", "xyz"[dim]],
            )
