"""Defines heat simulation class"""
from __future__ import annotations

from typing import Tuple, List, Dict
from matplotlib import cm

import pydantic.v1 as pd

from .boundary import TemperatureBC, HeatFluxBC, ConvectionBC
from .boundary import HeatBoundarySpec
from .source import HeatSourceType, UniformHeatSource
from .monitor import HeatMonitorType
from .grid import HeatGridType
from .viz import HEAT_BC_COLOR_TEMPERATURE, HEAT_BC_COLOR_FLUX, HEAT_BC_COLOR_CONVECTION
from .viz import plot_params_heat_bc, plot_params_heat_source, HEAT_SOURCE_CMAP

from ..base_sim.simulation import AbstractSimulation
from ..base import cached_property
from ..types import Ax, Shapely, TYPE_TAG_STR, ScalarSymmetry, Bound
from ..viz import add_ax_if_none, equal_aspect, PlotParams
from ..structure import Structure
from ..geometry.base import Box, GeometryGroup
from ..geometry.primitives import Sphere, Cylinder
from ..geometry.polyslab import PolySlab
from ..scene import Scene
from ..heat_spec import SolidSpec

from ..bc_placement import StructureBoundary, StructureStructureInterface
from ..bc_placement import StructureSimulationBoundary, SimulationBoundary
from ..bc_placement import MediumMediumInterface

from ...exceptions import SetupError
from ...constants import inf, VOLUMETRIC_HEAT_RATE

from ...log import log

HEAT_BACK_STRUCTURE_STR = "<<<HEAT_BACKGROUND_STRUCTURE>>>"

HeatSingleGeometryType = (Box, Cylinder, Sphere, PolySlab)


class HeatSimulation(AbstractSimulation):
    """Contains all information about heat simulation.

    Example
    -------
    >>> from tidy3d import Medium, SolidSpec, FluidSpec, UniformUnstructuredGrid, TemperatureMonitor
    >>> heat_sim = HeatSimulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(
    ...                 permittivity=2.0, heat_spec=SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=Medium(permittivity=3.0, heat_spec=FluidSpec()),
    ...     grid_spec=UniformUnstructuredGrid(dl=0.1),
    ...     sources=[UniformHeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         HeatBoundarySpec(
    ...             placement=StructureBoundary(structure="box"),
    ...             condition=TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[TemperatureMonitor(size=(1, 2, 3), name="sample")],
    ... )
    """

    boundary_spec: Tuple[HeatBoundarySpec, ...] = pd.Field(
        (),
        title="Boundary Condition Specifications",
        description="List of boundary condition specifications.",
    )

    sources: Tuple[HeatSourceType, ...] = pd.Field(
        (),
        title="Heat Sources",
        description="List of heat sources.",
    )

    monitors: Tuple[HeatMonitorType, ...] = pd.Field(
        (),
        title="Monitors",
        description="Monitors in the simulation.",
    )

    grid_spec: HeatGridType = pd.Field(
        title="Grid Specification",
        description="Grid specification for heat simulation.",
        discriminator=TYPE_TAG_STR,
    )

    symmetry: Tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (symmetry off) or ``1`` (symmetry on).",
    )

    @pd.validator("structures", always=True)
    def check_unsupported_geometries(cls, val):
        """Error if structures contain unsupported yet geometries."""
        for structure in val:
            if isinstance(structure.geometry, GeometryGroup):
                geometries = structure.geometry.geometries
            else:
                geometries = [structure.geometry]
            for geom in geometries:
                if isinstance(geom, (GeometryGroup)):
                    raise SetupError(
                        "'HeatSimulation' does not currently support recursive 'GeometryGroup's."
                    )
                if not isinstance(geom, HeatSingleGeometryType):
                    geom_names = [f"'{cl.__name__}'" for cl in HeatSingleGeometryType]
                    raise SetupError(
                        "'HeatSimulation' does not currently support geometries of type "
                        f"'{geom.type}'. Allowed geometries are "
                        f"{', '.join(geom_names)}, "
                        "and non-recursive 'GeometryGroup'."
                    )
        return val

    @pd.validator("size", always=True)
    def check_zero_dim_domain(cls, val, values):
        """Error if heat domain have zero dimensions."""

        if any(length == 0 for length in val):
            raise SetupError(
                "'HeatSimulation' does not currently support domains with dimensions of zero size."
            )

        return val

    @pd.validator("boundary_spec", always=True)
    def names_exist_bcs(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media."""

        structures = values.get("structures")
        structures_names = {s.name for s in structures}
        mediums_names = {s.medium.name for s in structures}
        mediums_names.add(values.get("medium").name)

        for bc_ind, bc_spec in enumerate(val):
            bc_place = bc_spec.placement
            if isinstance(bc_place, (StructureBoundary, StructureSimulationBoundary)):
                if bc_place.structure not in structures_names:
                    raise SetupError(
                        f"Structure '{bc_place.structure}' provided in "
                        f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}')"
                        "is not found among simulation structures."
                    )
            if isinstance(bc_place, (StructureStructureInterface)):
                for struct_name in bc_place.structures:
                    if struct_name and struct_name not in structures_names:
                        raise SetupError(
                            f"Structure '{struct_name}' provided in "
                            f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation structures."
                        )
            if isinstance(bc_place, (MediumMediumInterface)):
                for med_name in bc_place.mediums:
                    if med_name not in mediums_names:
                        raise SetupError(
                            f"Material '{med_name}' provided in "
                            f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation mediums."
                        )
        return val

    @pd.validator("grid_spec", always=True)
    def names_exist_grid_spec(cls, val, values):
        """Warn if UniformUnstructuredGrid points at a non-existing structure."""

        structures = values.get("structures")
        structures_names = {s.name for s in structures}

        for structure_name in val.non_refined_structures:
            if structure_name not in structures_names:
                log.warning(
                    f"Structure '{structure_name}' listed as a non-refined structure in "
                    "'HeatSimulation.grid_spec' is not present in 'HeatSimulation.structures'"
                )

        return val

    @pd.validator("sources", always=True)
    def names_exist_sources(cls, val, values):
        """Error if a heat source point to non-existing structures."""
        structures = values.get("structures")
        structures_names = {s.name for s in structures}

        for source in val:
            for name in source.structures:
                if name not in structures_names:
                    raise SetupError(
                        f"Structure '{name}' provided in a '{source.type}' "
                        "is not found among simulation structures."
                    )
        return val

    """ Post-init validators """

    def _post_init_validators(self) -> None:
        """Call validators taking z`self` that get run after init."""
        self._warn_multiple_zones()

    def _warn_multiple_zones(self):
        """Warn about current restriction on number of adjacent zones."""

        struc_src_map = {}
        for source in self.sources:
            for name in source.structures:
                struc_src_map[name] = source

        unique_solid_zones = {
            (struc.medium.heat_spec, struc_src_map.get(struc.name, None))
            for struc in self.structures
            if isinstance(struc.medium.heat_spec, SolidSpec)
        }

        if isinstance(self.medium.heat_spec, SolidSpec):
            unique_solid_zones.add((self.medium.heat_spec, None))

        if len(unique_solid_zones) > 2:
            log.warning(
                "More than 2 different solid zones (zone = medium + source) are detected in the heat "
                "simulation. Make sure no more than 2 solid zones are adjacent to each other anywhere "
                "in the simulation domain. The simulation results may be inaccurate otherwise. "
                "This restriction will be removed in the upcoming Tidy3D versions."
            )

    @equal_aspect
    @add_ax_if_none
    def plot_heat_conductivity(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        colorbar: str = "conductivity",
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        colorbar: str = "conductivity"
            Display colorbar for thermal conductivity ("conductivity") or heat source rate
            ("source").
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        cbar_cond = colorbar == "conductivity"

        ax = self.scene.plot_heat_conductivity(
            ax=ax, x=x, y=y, z=z, cbar=cbar_cond, alpha=alpha, hlim=hlim, vlim=vlim
        )
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, alpha=source_alpha, hlim=hlim, vlim=vlim)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, alpha=monitor_alpha, hlim=hlim, vlim=vlim)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        if colorbar == "source":
            self._add_heat_source_cbar(ax=ax)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's boundary conditions on a plane defined by one nonzero x,y,z
        coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get structure list
        structures = [self.simulation_structure]
        structures += list(self.structures)

        # construct slicing plane
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        # get boundary conditions in the plane
        boundaries = self._construct_heat_boundaries(
            structures=structures,
            plane=plane,
            boundary_spec=self.boundary_spec,
        )

        # plot boundary conditions
        for (bc_spec, shape) in boundaries:
            ax = self._plot_boundary_condition(shape=shape, boundary_spec=bc_spec, ax=ax)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)

        return ax

    def _get_bc_plot_params(self, boundary_spec: HeatBoundarySpec) -> PlotParams:
        """Constructs the plot parameters for given boundary conditions."""

        plot_params = plot_params_heat_bc
        condition = boundary_spec.condition

        if isinstance(condition, TemperatureBC):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(condition, HeatFluxBC):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(condition, ConvectionBC):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_CONVECTION)

        return plot_params

    def _plot_boundary_condition(
        self, shape: Shapely, boundary_spec: HeatBoundarySpec, ax: Ax
    ) -> Ax:
        """Plot a structure's cross section shape for a given boundary condition."""
        plot_params_bc = self._get_bc_plot_params(boundary_spec=boundary_spec)
        ax = self.plot_shape(shape=shape, plot_params=plot_params_bc, ax=ax)
        return ax

    @staticmethod
    def _structure_to_bc_spec_map(
        plane: Box, structures: Tuple[Structure, ...], boundary_spec: Tuple[HeatBoundarySpec, ...]
    ) -> Dict[str, HeatBoundarySpec]:
        """Construct structure name to bc spec inverse mapping. One structure may correspond to
        multiple boundary conditions."""

        named_structures_present = {structure.name for structure in structures if structure.name}

        struct_to_bc_spec = {}
        for bc_spec in boundary_spec:
            bc_place = bc_spec.placement
            if (
                isinstance(bc_place, (StructureBoundary, StructureSimulationBoundary))
                and bc_place.structure in named_structures_present
            ):
                if bc_place.structure in struct_to_bc_spec:
                    struct_to_bc_spec[bc_place.structure] += [bc_spec]
                else:
                    struct_to_bc_spec[bc_place.structure] = [bc_spec]

            if isinstance(bc_place, StructureStructureInterface):
                for structure in bc_place.structures:
                    if structure in named_structures_present:
                        if structure in struct_to_bc_spec:
                            struct_to_bc_spec[structure] += [bc_spec]
                        else:
                            struct_to_bc_spec[structure] = [bc_spec]

            if isinstance(bc_place, SimulationBoundary):
                struct_to_bc_spec[HEAT_BACK_STRUCTURE_STR] = [bc_spec]

        return struct_to_bc_spec

    @staticmethod
    def _medium_to_bc_spec_map(
        plane: Box, structures: Tuple[Structure, ...], boundary_spec: Tuple[HeatBoundarySpec, ...]
    ) -> Dict[str, HeatBoundarySpec]:
        """Construct medium name to bc spec inverse mapping. One medium may correspond to
        multiple boundary conditions."""

        named_mediums_present = {
            structure.medium.name for structure in structures if structure.medium.name
        }

        med_to_bc_spec = {}
        for bc_spec in boundary_spec:
            bc_place = bc_spec.placement
            if isinstance(bc_place, MediumMediumInterface):
                for med in bc_place.mediums:
                    if med in named_mediums_present:
                        if med in med_to_bc_spec:
                            med_to_bc_spec[med] += [bc_spec]
                        else:
                            med_to_bc_spec[med] = [bc_spec]

        return med_to_bc_spec

    @staticmethod
    def _construct_forward_boundaries(
        shapes: Tuple[Tuple[str, str, Shapely, Tuple[float, float, float, float]], ...],
        struct_to_bc_spec: Dict[str, HeatBoundarySpec],
        med_to_bc_spec: Dict[str, HeatBoundarySpec],
        background_structure_shape: Shapely,
    ) -> Tuple[Tuple[HeatBoundarySpec, Shapely], ...]:
        """Construct Simulation, StructureSimulation, Structure, and MediumMedium boundaries."""

        # forward foop to take care of Simulation, StructureSimulation, Structure,
        # and MediumMediums
        boundaries = []  # bc_spec, structure name, shape, bounds
        background_shapes = []
        for name, medium, shape, bounds in shapes:

            # intersect existing boundaries (both structure based and medium based)
            for index, (_bc_spec, _name, _bdry, _bounds) in enumerate(boundaries):

                # simulation bc is overriden only by StructureSimulationBoundary
                if isinstance(_bc_spec.placement, SimulationBoundary):
                    if name not in struct_to_bc_spec:
                        continue
                    if any(
                        not isinstance(bc_spec.placement, StructureSimulationBoundary)
                        for bc_spec in struct_to_bc_spec[name]
                    ):
                        continue

                if Box._do_not_intersect(bounds, _bounds, shape, _bdry):
                    continue

                diff_shape = _bdry - shape

                boundaries[index] = (_bc_spec, _name, diff_shape, diff_shape.bounds)

            # create new structure based boundary

            if name in struct_to_bc_spec:
                for bc_spec in struct_to_bc_spec[name]:

                    if isinstance(bc_spec.placement, StructureBoundary):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries.append((bc_spec, name, bdry, bdry.bounds))

                    if isinstance(bc_spec.placement, SimulationBoundary):
                        boundaries.append((bc_spec, name, shape.exterior, shape.exterior.bounds))

                    if isinstance(bc_spec.placement, StructureSimulationBoundary):
                        bdry = background_structure_shape.exterior
                        bdry = bdry.intersection(shape)
                        boundaries.append((bc_spec, name, bdry, bdry.bounds))

            # create new medium based boundary, and cut or merge relevant background shapes

            # loop through background_shapes (note: all background are non-intersecting or merged)
            # this is similar to _filter_structures_plane but only mediums participating in BCs
            # are tracked
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

                if Box._do_not_intersect(bounds, _bounds, shape, _shape):
                    continue

                diff_shape = _shape - shape

                # different medium, remove intersection from background shape
                if medium != _medium and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

                    # in case when there is a bc between two media
                    # create a new boudnary segment
                    for bc_spec in med_to_bc_spec[_medium.name]:
                        if medium.name in bc_spec.placement.mediums:
                            bdry = shape.exterior.intersection(_shape)
                            bdry = bdry.intersection(background_structure_shape)
                            boundaries.append((bc_spec, name, bdry, bdry.bounds))

                # same medium, add diff shape to this shape and mark background shape for removal
                # note: this only happens if this medium is listed in BCs
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            # but only if this medium is listed in BCs
            if medium.name in med_to_bc_spec:
                background_shapes.append((medium, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out empty geometries
        boundaries = [(bc_spec, bdry) for (bc_spec, name, bdry, _) in boundaries if bdry]

        return boundaries

    @staticmethod
    def _construct_reverse_boundaries(
        shapes: Tuple[Tuple[str, str, Shapely, Bound], ...],
        struct_to_bc_spec: Dict[str, HeatBoundarySpec],
        background_structure_shape: Shapely,
    ) -> Tuple[Tuple[HeatBoundarySpec, Shapely], ...]:
        """Construct StructureStructure boundaries."""

        # backward foop to take care of StructureStructure
        # we do it in this way because we define the boundary between
        # two overlapping structures A and B, where A comes before B, as
        # boundary(B) intersected by A
        # So, in this loop as we go backwards through the structures we:
        # - (1) when come upon B, create boundary(B)
        # - (2) cut away from it by other structures
        # - (3) when come upon A, intersect it with A and mark it as complete,
        #   that is, no more further modifications
        boundaries_reverse = []

        for name, _, shape, bounds in shapes[:0:-1]:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries
            for index, (_bc_spec, _name, _bdry, _bounds, _completed) in enumerate(
                boundaries_reverse
            ):

                if not _completed:

                    if Box._do_not_intersect(bounds, _bounds, shape, _bdry):
                        continue

                    # event (3) from above
                    if name in _bc_spec.placement.structures:
                        new_bdry = _bdry.intersection(shape)
                        boundaries_reverse[index] = (
                            _bc_spec,
                            _name,
                            new_bdry,
                            new_bdry.bounds,
                            True,
                        )

                    # event (2) from above
                    else:
                        new_bdry = _bdry - shape
                        boundaries_reverse[index] = (
                            _bc_spec,
                            _name,
                            new_bdry,
                            new_bdry.bounds,
                            _completed,
                        )

            # create new boundary (event (1) from above)
            if name in struct_to_bc_spec:
                for bc_spec in struct_to_bc_spec[name]:
                    if isinstance(bc_spec.placement, StructureStructureInterface):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries_reverse.append((bc_spec, name, bdry, bdry.bounds, False))

        # filter and append completed boundaries to main list
        filtered_boundaries = []
        for bc_spec, _, bdry, _, is_completed in boundaries_reverse:
            if bdry and is_completed:
                filtered_boundaries.append((bc_spec, bdry))

        return filtered_boundaries

    @staticmethod
    def _construct_heat_boundaries(
        structures: List[Structure],
        plane: Box,
        boundary_spec: List[HeatBoundarySpec],
    ) -> List[Tuple[HeatBoundarySpec, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_spec : List[HeatBoundarySpec]
            list of boundary conditions associated with structures.

        Returns
        -------
        List[Tuple[:class:`.HeatBoundarySpec`, shapely.geometry.base.BaseGeometry]]
            List of boundary lines and boundary conditions on the plane after merging.
        """

        # get structures in the plane and present named structures and media
        shapes = []  # structure name, structure medium, shape, bounds
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = plane.intersections_with(structure.geometry)

            # append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, structure.medium, shape, shape.bounds))

        background_structure_shape = shapes[0][2]

        # construct an inverse mapping structure -> bc for present structures
        struct_to_bc_spec = HeatSimulation._structure_to_bc_spec_map(
            plane=plane, structures=structures, boundary_spec=boundary_spec
        )

        # construct an inverse mapping medium -> bc for present mediums
        med_to_bc_spec = HeatSimulation._medium_to_bc_spec_map(
            plane=plane, structures=structures, boundary_spec=boundary_spec
        )

        # construct boundaries in 2 passes:

        # 1. forward foop to take care of Simulation, StructureSimulation, Structure,
        # and MediumMediums
        boundaries = HeatSimulation._construct_forward_boundaries(
            shapes=shapes,
            struct_to_bc_spec=struct_to_bc_spec,
            med_to_bc_spec=med_to_bc_spec,
            background_structure_shape=background_structure_shape,
        )

        # 2. reverse loop: construct structure-structure boundary
        struct_struct_boundaries = HeatSimulation._construct_reverse_boundaries(
            shapes=shapes,
            struct_to_bc_spec=struct_to_bc_spec,
            background_structure_shape=background_structure_shape,
        )

        return boundaries + struct_struct_boundaries

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # background can't have source, so no need to add background structure
        structures = self.structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        # distribute source where there are assigned
        structure_source_map = {}
        for source in self.sources:
            for name in source.structures:
                structure_source_map[name] = source

        source_list = [structure_source_map.get(structure.name, None) for structure in structures]

        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        source_shapes = self.scene._filter_structures_plane(
            structures=structures, plane=plane, property_list=source_list
        )

        source_min, source_max = self.source_bounds
        for (source, shape) in source_shapes:
            if source is not None:
                ax = self._plot_shape_structure_source(
                    alpha=alpha,
                    source=source,
                    source_min=source_min,
                    source_max=source_max,
                    shape=shape,
                    ax=ax,
                )

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)
        return ax

    def _add_heat_source_cbar(self, ax: Ax):
        """Add colorbar for heat sources."""
        source_min, source_max = self.source_bounds
        self.scene._add_cbar(
            vmin=source_min,
            vmax=source_max,
            label=f"Volumetric heat rate ({VOLUMETRIC_HEAT_RATE})",
            cmap=HEAT_SOURCE_CMAP,
            ax=ax,
        )

    @cached_property
    def source_bounds(self) -> Tuple[float, float]:
        """Compute range of heat sources present in the simulation."""

        rate_list = [
            source.rate for source in self.sources if isinstance(source, UniformHeatSource)
        ]
        rate_list.append(0)
        rate_min = min(rate_list)
        rate_max = max(rate_list)
        return rate_min, rate_max

    def _get_structure_source_plot_params(
        self,
        source: HeatSourceType,
        source_min: float,
        source_max: float,
        alpha: float = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in simulation.plot_eps()."""

        plot_params = plot_params_heat_source
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if isinstance(source, UniformHeatSource):
            rate = source.rate
            delta_rate = rate - source_min
            delta_rate_max = source_max - source_min + 1e-5
            rate_fraction = delta_rate / delta_rate_max
            cmap = cm.get_cmap(HEAT_SOURCE_CMAP)
            rgba = cmap(rate_fraction)
            plot_params = plot_params.copy(update={"edgecolor": rgba})

        return plot_params

    def _plot_shape_structure_source(
        self,
        source: HeatSourceType,
        shape: Shapely,
        source_min: float,
        source_max: float,
        ax: Ax,
        alpha: float = None,
    ) -> Ax:
        """Plot a structure's cross section shape for a given medium, grayscale for permittivity."""
        plot_params = self._get_structure_source_plot_params(
            source=source,
            source_min=source_min,
            source_max=source_max,
            alpha=alpha,
        )
        ax = self.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> HeatSimulation:
        """Create a simulation from a :class:.`Scene` instance. Must provide additional parameters
        to define a valid simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:.`Scene`
            Scene containing structures information.
        **kwargs
            Other arguments

        Example
        -------
        >>> from tidy3d import Scene, Medium, Box, Structure, UniformUnstructuredGrid
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> scene = Scene(
        ...     structures=[box],
        ...     medium=Medium(permittivity=3),
        ... )
        >>> sim = HeatSimulation.from_scene(
        ...     scene=scene,
        ...     center=(0, 0, 0),
        ...     size=(5, 6, 7),
        ...     grid_spec=UniformUnstructuredGrid(dl=0.4),
        ... )
        """

        return cls(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )
