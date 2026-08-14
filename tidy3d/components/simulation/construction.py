"""Scene construction, subsectioning, and padding helpers for Yee-grid simulations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from tidy3d.components.boundary import PML, Absorber, Boundary, Periodic, StablePML
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.utils import filter_intersecting_geometries
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.medium import AbstractCustomMedium
from tidy3d.constants import fp_eps
from tidy3d.exceptions import (
    SetupError,
)
from tidy3d.log import log

if TYPE_CHECKING:
    from typing import Literal

    from pydantic import NonNegativeFloat

    from tidy3d.compat import Self
    from tidy3d.components.boundary import BoundarySpec, InternalAbsorber
    from tidy3d.components.scene import Scene
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.types import Symmetry
    from tidy3d.components.types.monitor import MonitorType


def subsection(
    self: Any,
    region: Box,
    boundary_spec: BoundarySpec = None,
    grid_spec: GridSpec | Literal["identical"] = None,
    symmetry: tuple[Symmetry, Symmetry, Symmetry] | None = None,
    warn_symmetry_expansion: bool = True,
    sources: tuple[SourceType, ...] | None = None,
    monitors: tuple[MonitorType, ...] | None = None,
    remove_outside_structures: bool = True,
    remove_outside_grid_spec: bool = False,
    remove_outside_custom_mediums: bool = False,
    include_pml_cells: bool = False,
    validate_geometries: bool = True,
    deep_copy: bool = True,
    internal_absorbers: tuple[InternalAbsorber, ...] | None = None,
    **kwargs: Any,
) -> Self:
    """Generate a simulation instance containing only the ``region``.

    Parameters
    ----------
    region : :class:`.Box`
        New simulation domain.
    boundary_spec : :class:`.BoundarySpec` = None
        New boundary specification. If ``None``, then it is inherited from the original
        simulation.
    grid_spec : :class:`.GridSpec` = None
        New grid specification. If ``None``, then it is inherited from the original
        simulation. If ``identical``, then the original grid is transferred directly as a
        :class:`.CustomGrid`. Note that in the latter case the region of the new simulation is
        snapped to the original grid lines.
    symmetry : tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
        New simulation symmetry. If ``None``, then it is inherited from the original
        simulation. Note that in this case the size and placement of new simulation domain
        must be commensurate with the original symmetry.
    warn_symmetry_expansion : bool = True
        Whether to warn when the subsection is expanded to preserve symmetry.
    sources : tuple[SourceType, ...] = None
        New list of sources. If ``None``, then the sources intersecting the new simulation
        domain are inherited from the original simulation.
    monitors : tuple[MonitorType, ...] = None
        New list of monitors. If ``None``, then the monitors intersecting the new simulation
        domain are inherited from the original simulation.
    remove_outside_structures : bool = True
        Remove structures outside of the new simulation domain.
    remove_outside_grid_spec : bool = False
        Prune or clip ``override_structures``, ``layer_refinement_specs``, and
        ``snapping_points`` in ``grid_spec`` to the requested region. Only
        applies when at least one axis uses :class:`.AutoGrid` or
        :class:`.QuasiUniformGrid`.
    remove_outside_custom_mediums : bool = True
        Remove custom medium data outside of the new simulation domain.
    include_pml_cells : bool = False
        Keep PML cells in simulation boundaries. Note that retained PML cells will be converted
        to regular cells, and the simulation domain boundary will be moved accordingly.
    validate_geometries: bool = True
        If ``False``, skip validation for the geometries in the resulting simulation object.
        Simulation validators remain but only use the bounding box of the existing geometries.
        Used internally.
    deep_copy: bool = True
        Recursively copy all nested objects in the generated simulation object.
    internal_absorbers : Tuple[InternalAbsorber, ...] = None
        New list of internal absorbers. If ``None``, then the absorbers intersecting the new simulation
        domain are inherited from the original simulation.
    **kwargs
        Other arguments passed to new simulation instance.
    """

    # must intersect the original domain
    if not self.intersects(region):
        raise SetupError("Requested region does not intersect simulation domain")

    # restrict to the original simulation domain
    if include_pml_cells:
        new_bounds = Box.bounds_intersection(self.simulation_bounds, region.bounds)
    else:
        new_bounds = Box.bounds_intersection(self.bounds, region.bounds)
    new_bounds = [list(new_bounds[0]), list(new_bounds[1])]

    # grid spec inheritace
    if grid_spec is None:
        grid_spec = self.grid_spec
    elif isinstance(grid_spec, str) and grid_spec == "identical":
        # create a custom grid from existing one
        grids_1d = self.grid.boundaries.to_list
        grid_spec = GridSpec.from_grid(self.grid)

        # adjust region bounds to perfectly coincide with the grid
        # note, sometimes (when a box already seems to perfrecty align with the grid)
        # this causes the new region to expand one more pixel because of numerical roundoffs
        # To help to avoid that we shrink new region by a small amount.
        center = [(bmin + bmax) / 2 for bmin, bmax in zip(*new_bounds)]
        size = [max(0.0, bmax - bmin - 2 * fp_eps) for bmin, bmax in zip(*new_bounds)]
        aux_box = Box(center=center, size=size)
        grid_inds = self.grid.discretize_inds(box=aux_box)

        for dim in range(3):
            # preserve zero size dimensions
            if new_bounds[0][dim] != new_bounds[1][dim]:
                new_bounds[0][dim] = grids_1d[dim][grid_inds[dim][0]]
                new_bounds[1][dim] = grids_1d[dim][grid_inds[dim][1]]

    # if symmetry is not overriden we inherit it from the original simulation where is needed
    if symmetry is None:
        # start with no symmetry
        symmetry = [0, 0, 0]

        # now check in each dimension whether we cross symmetry plane
        for dim in range(3):
            if self.symmetry[dim] != 0:
                crosses_symmetry = (
                    new_bounds[0][dim] < self.center[dim] and new_bounds[1][dim] > self.center[dim]
                )

                # inherit symmetry only if we cross symmetry plane, otherwise we don't impose
                # symmetry even if the original simulation had symmetry
                if crosses_symmetry:
                    symmetry[dim] = self.symmetry[dim]
                    center = (new_bounds[0][dim] + new_bounds[1][dim]) / 2

                    if not math.isclose(center, self.center[dim]):
                        if warn_symmetry_expansion:
                            log.warning(
                                f"The original simulation is symmetric along {'xyz'[dim]} direction. "
                                "The requested new simulation region does cross the symmetry plane but is "
                                "not symmetric with respect to it. To preserve correct symmetry, "
                                "the requested simulation region is expanded symmetrically."
                            )
                        new_bounds[0][dim] = 2 * self.center[dim] - new_bounds[1][dim]

    # symmetry and grid spec treatments could change new simulation bounds
    # thus, recreate a box instance
    new_box = Box.from_bounds(*new_bounds)

    if remove_outside_grid_spec:
        grid_spec = grid_spec._localized_copy(region=region)

    # Filter structures to those intersecting the subsection region using recursive
    # geometry pruning, then replace each structure's geometry with the pruned version.
    if remove_outside_structures:
        pruned_geometries = filter_intersecting_geometries(
            [strc.geometry for strc in self.structures], new_box
        )
        new_structures = [
            strc.updated_copy(geometry=geometry, deep=False)
            for strc, geometry in zip(self.structures, pruned_geometries)
            if geometry is not None
        ]
    else:
        new_structures = list(self.structures)

    # If ``validate_geometries=False``, use aux structures whose geometry is replaced by its bounding box
    # so that other validations are still performed.
    aux_new_structures = new_structures
    if not validate_geometries:
        aux_new_structures = [
            strc.updated_copy(geometry=strc.geometry.bounding_box, deep=deep_copy)
            for strc in new_structures
        ]

    new_lumped_elements = [
        elem for elem in self.lumped_elements if new_box.intersects(elem.to_geometry())
    ]

    if sources is None:
        sources = [src for src in self.sources if new_box.intersects(src)]

    if internal_absorbers is None:
        internal_absorbers = [
            abc for abc in self._shifted_internal_absorbers if new_box.intersects(abc)
        ]

    if monitors is None:
        monitors = [mnt for mnt in self.monitors if new_box.intersects(mnt)]

    if boundary_spec is None:
        boundary_spec = self.boundary_spec

    # set boundary conditions in zero-size dimension to periodic
    for dim in range(3):
        if new_bounds[0][dim] == new_bounds[1][dim] and not isinstance(
            boundary_spec.to_list[dim][0], Periodic
        ):
            axis_name = "xyz"[dim]
            log.warning(
                f"The resulting simulation subsection has size zero along axis '{axis_name}'. "
                "Periodic boundary conditions are automatically set along this dimension."
            )
            boundary_spec = boundary_spec.updated_copy(**{"xyz"[dim]: Boundary.periodic()})

    # reduction of custom medium data
    new_sim_medium = self.medium
    if remove_outside_custom_mediums:
        # check for special treatment in case of PML
        if any(
            any(isinstance(edge, PML | StablePML | Absorber) for edge in boundary)
            for boundary in boundary_spec.to_list
        ):
            # if we need to cut out outside custom medium we have to be careful about PML/Absorber
            # we should include data in PML so that there is no artificial reflection at PML boundaries

            # to do this, we first create an auxiliary simulation
            aux_sim = self.updated_copy(
                center=new_box.center,
                size=new_box.size,
                grid_spec=grid_spec,
                boundary_spec=boundary_spec,
                monitors=(),
                sources=tuple(sources),  # need wavelength in case of auto grid
                symmetry=tuple(symmetry),
                structures=tuple(aux_new_structures),
                deep=deep_copy,
            )

            # then use its bounds as region for data cut off
            new_bounds = aux_sim.simulation_bounds

            # Note that this is not a full proof strategy. For example, if grid_spec is AutoGrid
            # then after outside custom medium data is removed the grid sizes and, thus,
            # pml extents can change as well

        # now cut out custom medium data
        new_structures_reduced_data = []
        aux_new_structures_reduced_data = []

        for structure in new_structures:
            medium = structure.medium
            if isinstance(medium, AbstractCustomMedium):
                new_structure_bounds = Box.bounds_intersection(
                    new_bounds, structure.geometry.bounds
                )
                new_medium = medium.sel_inside(bounds=new_structure_bounds)
                # if skip geometry validation, structure validation is performed in aux structure below
                new_structure = structure.updated_copy(
                    medium=new_medium, deep=deep_copy, validate=validate_geometries
                )
                new_structures_reduced_data.append(new_structure)
                if not validate_geometries:
                    aux_new_structure = new_structure.updated_copy(
                        geometry=new_structure.geometry.bounding_box,
                        deep=deep_copy,
                        validate=True,
                    )
                    aux_new_structures_reduced_data.append(aux_new_structure)
            else:
                new_structures_reduced_data.append(structure)
                if not validate_geometries:
                    aux_new_structures_reduced_data.append(
                        structure.updated_copy(
                            geometry=structure.geometry.bounding_box, deep=deep_copy
                        )
                    )

        new_structures = new_structures_reduced_data
        aux_new_structures = new_structures_reduced_data
        if not validate_geometries:
            aux_new_structures = aux_new_structures_reduced_data

        if isinstance(self.medium, AbstractCustomMedium):
            new_sim_medium = self.medium.sel_inside(bounds=new_bounds)

    # finally, create an updated copy with all modifications
    new_sim_dict = dict(
        center=new_box.center,
        size=new_box.size,
        medium=new_sim_medium,
        grid_spec=grid_spec,
        boundary_spec=boundary_spec,
        monitors=tuple(monitors),
        sources=tuple(sources),
        symmetry=tuple(symmetry),
        structures=tuple(aux_new_structures),
        lumped_elements=tuple(new_lumped_elements),
        internal_absorbers=tuple(internal_absorbers),
        **kwargs,
    )

    if validate_geometries:
        return self.updated_copy(**new_sim_dict, deep=deep_copy)
    # 1) Perform validators not directly related to geometries
    new_sim = self.updated_copy(**new_sim_dict, deep=deep_copy, validate=True)
    # 2) Assemble the full simulation without validation
    return new_sim.updated_copy(structures=tuple(new_structures), deep=deep_copy, validate=False)


def _invalidate_solver_cache(self: Any) -> None:
    """Clear cached attributes that become stale when subpixel changes."""
    self._cached_properties.pop("_mode_solver", None)


@classmethod
def from_scene(cls: Any, scene: Scene, **kwargs: Any) -> Self:
    """Create a simulation from a :class:`.Scene` instance. Must provide additional parameters
    to define a valid simulation (for example, ``run_time``, ``grid_spec``, etc).

    Parameters
    ----------
    scene : :class:`.Scene`
        Size of object in x, y, and z directions.
    **kwargs
        Other arguments passed to new simulation instance.

    Example
    -------
    >>> from tidy3d import Scene, Medium, Box, Structure, GridSpec, Simulation
    >>> box = Structure(
    ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
    ...     medium=Medium(permittivity=5),
    ... )
    >>> scene = Scene(
    ...     structures=[box],
    ...     medium=Medium(permittivity=3),
    ... )
    >>> sim = Simulation.from_scene(
    ...     scene=scene,
    ...     center=(0, 0, 0),
    ...     size=(5, 6, 7),
    ...     run_time=1e-12,
    ...     grid_spec=GridSpec.uniform(dl=0.4),
    ... )
    """
    return cls(
        structures=scene.structures,
        medium=scene.medium,
        **kwargs,
    )


def padded_copy(
    self: Any,
    x: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
    y: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
    z: tuple[NonNegativeFloat, NonNegativeFloat] | None = None,
) -> Self:
    """Created a copy of simulation with padded simulation domain.

    Parameters
    ----------
    x : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
        Padding sizes at the left and right boundaries of the simulation along x-axis.
    y : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
        Padding sizes at the left and right boundaries of the simulation along y-axis.
    z : Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None
        Padding sizes at the left and right boundaries of the simulation along z-axis.

    Returns
    -------
    Simulation
        Simulation with padded simulation domain.
    """
    # get simulation bounding box and pad it
    box = Box(center=self.center, size=self.size)
    padded_box = box.padded_copy(x, y, z)

    return self.updated_copy(size=padded_box.size, center=padded_box.center)


def uniformly_padded_copy(self: Any, padding: NonNegativeFloat) -> Self:
    """Create copy of simulation with uniformly padded simulation domain.

    Parameters
    ----------
    padding : NonNegativeFloat
        Padding size applied uniformly at all simulation boundaries.

    Returns
    -------
    Simulation
        Simulation with uniformly padded simulation domain.
    """
    if padding < 0:
        raise ValueError(f"Padding must be non-negative. Got {padding}.")

    padding_tuple = (padding, padding)
    return self.padded_copy(x=padding_tuple, y=padding_tuple, z=padding_tuple)
