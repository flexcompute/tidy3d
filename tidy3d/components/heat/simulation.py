"""Defines heat simulation class"""
from __future__ import annotations

from typing import Tuple, List
from matplotlib import cm

import pydantic as pd
from shapely.plotting import plot_line
from shapely import LineString, MultiLineString, GeometryCollection

from .boundary import TemperatureBC, HeatFluxBC, ConvectionBC
from .boundary import HeatBoundarySpec
from ..bc_placement import StructureBoundary, StructureStructureInterface
from ..bc_placement import StructureSimulationBoundary, SimulationBoundary
from ..bc_placement import MediumMediumInterface
from .source import HeatSourceType, UniformHeatSource
from .grid import HeatGridType
from .viz import HEAT_BC_COLOR_TEMPERATURE, HEAT_BC_COLOR_FLUX, HEAT_BC_COLOR_CONVECTION
from .viz import plot_params_heat_bc, plot_params_heat_source, HEAT_SOURCE_CMAP

from ..base import cached_property, Tidy3dBaseModel
from ..types import Ax, Shapely, TYPE_TAG_STR
from ..viz import add_ax_if_none, equal_aspect, PlotParams
from ..scene import Scene
from ..structure import Structure
from ..geometry import Box, Sphere, Cylinder, GeometryGroup

from ...exceptions import SetupError
from ...constants import inf, VOLUMETRIC_HEAT_RATE

HEAT_BACK_STRUCTURE_STR = "<<<HEAT_BACKGROUND_STRUCTURE>>>"


class HeatSimulation(Tidy3dBaseModel):
    """Contains all information about heat simulation.

    Example
    -------
    >>> FIXME
    """

    scene: Scene = pd.Field(
        title="Simulation Scene",
        description="Simulation scene describing problem geometry.",
    )

    boundary_specs: Tuple[HeatBoundarySpec, ...] = pd.Field(
        (),
        title="Boundary Condition Specifications",
        description="List of boundary condition specifications.",
    )

    heat_sources: Tuple[HeatSourceType, ...] = pd.Field(
        (),
        title="Heat Sources",
        description="List of heat sources.",
    )

    grid_spec: HeatGridType = pd.Field(
        title="Grid Specification",
        description="Grid specification for heat simulation.",
        discriminator=TYPE_TAG_STR,
    )

#    output_grid_spec: HeatGridType = pd.Field(
#        title="Grid Specification",
#        description="Grid specification for heat simulation.",
##        discriminator=TYPE_TAG_STR,
#    )

    heat_domain: Box = pd.Field(
        None,
        title="Heat Simulation Domain",
        description="Domain in which heat simulation is solved. If ``None`` heat simulation is "
        "solved in the entire domain of the scene."
    )

    @pd.validator("scene", always=True)
    def check_unsupported_geometries(cls, val, values):
        """Error if structures contain unsupported yet geometries."""
        for structure in val.structures:
            if isinstance(structure.geometry, GeometryGroup):
                geometries = structure.geometry.geometries
            else:
                geometries = [structure.geometry]
            for geom in geometries:
                if not isinstance(geom, (Box, Cylinder, Sphere)):
                    raise SetupError(
                        f"HeatSimulation does not currently support geometries of type {type(geom)}"
                        f". Allowed geometries are 'Box', 'Cylinder', and 'Sphere'."
                    )
        return val

    @pd.validator("heat_domain", always=True)
    def check_zero_dim_domain(cls, val, values):
        """Error if heat domain have zero dimensions."""

        if val is not None:
            size = val.size
        else:
            size = values.get("scene").size

        if any(length == 0 for length in size):
            raise SetupError(
                "HeatSimulation does not currently support domain with zero dimensions."
            )

        return val

    @pd.validator("boundary_specs", always=True)
    def names_exist_bcs(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media."""
        scene = values.get("scene")
        structures = scene.structures
        mediums = scene.mediums
        structures_names = {s.name for s in structures}
        mediums_names = {m.name for m in mediums}

        for bc_ind, bc_spec in enumerate(val):
            bc_place = bc_spec.placement
            if isinstance(bc_place, (StructureBoundary, StructureSimulationBoundary)):
                if bc_place.structure not in structures_names:
                    raise SetupError(
                        f"Structure '{bc_place.structure}' provided in "
                        f"`boundary_specs[{bc_ind}].placement' (type '{bc_place.type}')"
                        "is not found among simulation structures."
                    )
            if isinstance(bc_place, (StructureStructureInterface)):
                for ind in range(2):
                    if bc_place.structures[ind] and bc_place.structures[ind] not in structures_names:
                        raise SetupError(
                            f"Structure '{bc_place.structures[ind]}' provided in "
                            f"`boundary_specs[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation structures."
                        )
            if isinstance(bc_place, (MediumMediumInterface)):
                for ind in range(2):
                    if bc_place.mediums[ind] not in mediums_names:
                        raise SetupError(
                            f"Material '{bc_place.mediums[ind]}' provided in "
                            f"`boundary_specs[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation mediums."
                        )
        return val

    @pd.validator("heat_sources", always=True)
    def names_exist_sources(cls, val, values):
        """Error if heat point to non-existing structures."""
        scene = values.get("scene")
        structures = scene.structures
        structures_names = {s.name for s in structures}

        for source in val:
            for name in source.structures:
                if name not in structures_names:
                    raise SetupError(
                        f"Structure '{name}' provided in a '{source.type}' "
                        "is not found among simulation structures."
                    )
        return val

#    def to_perturbed_mediums_scene(self, temperature: SpatialDataArray) -> Simulation:
#        """Returns underlying :class:`.Simulation` object."""

#        return self.scene.perturbed_mediums_copy(temperature=temperature)

    @cached_property
    def heat_domain_structure(self) -> Structure:
        """Returns structure representing the domain of the :class:`.HeatSimulation`."""

        # Unlike the FDTD Simulation.background_structure, the current one is also used to provide/
        # information about domain in which heat simulation is solved. Thus, we set its boundaries
        # either to self.heat_domain or, if None, to bounding box of self.scene
        if self.heat_domain:
            heat_domain_actual = self.heat_domain
        else:
            heat_domain_actual = self.scene.bounding_box

#        fdtd_background = self.background_structure
#        return fdtd_background.updated_copy(geometry=heat_domain_actual, name=HEAT_BACK_STRUCTURE_STR)
        return Structure(geometry=heat_domain_actual, medium=self.scene.medium, name=HEAT_BACK_STRUCTURE_STR)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        colorbar: str = "conductivity",
        **patch_kwargs,
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
        colorbar: str = "conductivity"
            Display colorbar for thermal conductivity ("conductivity") or heat source rate
            ("source").
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.scene.plot_heat_conductivity(ax=ax, x=x, y=y, z=z, cbar=(colorbar == "conductivity"))
        ax = self.plot_heat_sources(ax=ax, x=x, y=y, z=z, cbar=(colorbar == "source"))
        ax = self.plot_heat_boundaries(ax=ax, x=x, y=y, z=z)
        ax = self.scene._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_heat_boundaries(  # pylint: disable=too-many-arguments,too-many-locals
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
        structures = [self.heat_domain_structure]
        structures += list(self.scene.structures)

        # construct slicing plane
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        # get boundary conditions in the plane
        boundaries = self._construct_heat_boundaries(
            structures=structures,
            plane=plane,
            boundary_specs=self.boundary_specs,
        )

        # plot boundary conditions
        for (bc_spec, shape) in boundaries:
            ax = self._plot_boundary_condition(shape=shape, boundary_spec=bc_spec, ax=ax)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.scene.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = self.scene._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        return ax

    def _get_bc_plot_params(self, boundary_spec: HeatBoundarySpec) -> PlotParams:
        """Constructs the plot parameters for given boundary conditions."""

        plot_params = plot_params_heat_bc
        bc = boundary_spec.condition

        if isinstance(bc, TemperatureBC):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(bc, HeatFluxBC):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(bc, ConvectionBC):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_CONVECTION)

        return plot_params

    def _plot_boundary_condition(self, shape: Shapely, boundary_spec: HeatBoundarySpec, ax: Ax) -> Ax:
        """Plot a structure's cross section shape for a given boundary condition."""
        plot_params_bc = self._get_bc_plot_params(boundary_spec=boundary_spec)
        ax = self.plot_line(line=shape, plot_params=plot_params_bc, ax=ax)
        return ax

    # FIXME: probably needs revision
    def plot_line(self, line: Shapely, plot_params: PlotParams, ax: Ax) -> Ax:
        """Defines how a line is plotted on a matplotlib axes."""

        if isinstance(line, (MultiLineString, GeometryCollection)):
            lines = line.geoms
        elif isinstance(line, LineString):
            lines = [line]

        for l in lines:
            plot_line(l, ax=ax, add_points=False, color=plot_params.edgecolor, linewidth=plot_params.linewidth)
            # ax.add_artist(patch)
        return ax

    # pylint:disable=too-many-locals
    @staticmethod
    def _construct_heat_boundaries(
        structures: List[Structure],
        plane: Box,
        boundary_specs: List[HeatBoundarySpec],
    ) -> List[Tuple[HeatBoundarySpec, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_specs : List[HeatBoundarySpec]
            list of boundary conditions associated with structures.

        Returns
        -------
        List[Tuple[:class:`.HeatBoundarySpec`, shapely.geometry.base.BaseGeometry]]
            List of boundary lines and boundary conditions on the plane after merging.
        """

        # get structures in the plane and present named structures and media
        shapes = []  # structure name, structure medium, shape, bounds
        named_structures_present = set()
        named_mediums_present = set()
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections_2dbox(plane)

            # append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, structure.medium, shape, shape.bounds))

            # also keep track of present named structures and media
            if structure.name:
                named_structures_present.add(structure.name)

            if structure.medium.name:
                named_mediums_present.add(structure.medium.name)

        background_structure_shape = shapes[0][2]

        # construct an inverse mapping structure -> bc for present structures
        struct_to_bc_spec = {}
        for bc_spec in boundary_specs:
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

        # construct an inverse mapping medium -> bc for present mediums
        med_to_bc_spec = {}
        for bc_spec in boundary_specs:
            bc_place = bc_spec.placement
            if isinstance(bc_place, MediumMediumInterface):
                for med in bc_place.mediums:
                    if med in named_mediums_present:
                        if med in med_to_bc_spec:
                            med_to_bc_spec[med] += [bc_spec]
                        else:
                            med_to_bc_spec[med] = [bc_spec]

        # construct boundaries in 2 passes:

        # 1. forward foop to take care of Simulation, StructureSimulation, Structure, and MediumMediums
        boundaries = [] # bc_spec, structure name, shape, bounds
        background_shapes = []
        for name, medium, shape, bounds in shapes:

            minx, miny, maxx, maxy = bounds

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

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _bdry.is_empty or not shape.intersects(_bdry):
                    continue

                diff_shape = _bdry - shape

                boundaries[index] = (_bc_spec, _name, diff_shape, diff_shape.bounds)

            # create new srtucture based boundary

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
            # this is similar to _filter_structures_plane but only mediums participating in BCs are tracked
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _shape.is_empty or not shape.intersects(_shape):
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

        # 2. backward foop to take care of StructureStructure
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
            for index, (_bc_spec, _name, _bdry, _bounds, _completed) in enumerate(boundaries_reverse):

                if not _completed:

                    _minx, _miny, _maxx, _maxy = _bounds

                    # do a bounding box check to see if any intersection to do anything about
                    if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                        continue

                    # look more closely to see if intersected.
                    if _bdry.is_empty or not shape.intersects(_bdry):
                        continue

                    # event (3) from above
                    if name in _bc_spec.structure.structures:
                        new_bdry = _bdry.intersection(shape)
                        boundaries_reverse[index] = (_bc_spec, _name, new_bdry, new_bdry.bounds, True)

                    # event (2) from above
                    else:
                        new_bdry = _bdry - shape
                        boundaries_reverse[index] = (_bc_spec, _name, new_bdry, new_bdry.bounds, _completed)

            # create new boundary (event (1) from above)
            if name in struct_to_bc_spec:
                for bc_spec in struct_to_bc_spec[name]:
                    if isinstance(bc_spec.placement, StructureStructureInterface):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries_reverse.append((bc_spec, name, bdry, bdry.bounds, False))

        # filter and append completed boundaries to main list
        for bc_spec, _, bdry, _, is_completed in boundaries_reverse:
            if bdry and is_completed:
                boundaries.append((bc_spec, bdry))

        return boundaries

    @equal_aspect
    @add_ax_if_none
    def plot_heat_sources(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        alpha: float = None,
        cbar: bool = True,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = 0.5
            Opacity of the sources being plotted.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # background can't have source, so no need to add background structure
        structures = self.scene.structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        # distribute source where there are assigned
        structure_source_map = {}
        for source in self.heat_sources:
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

        source_min, source_max = self.source_bounds()
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

        if cbar:
            self.scene._add_cbar(
                vmin=source_min,
                vmax=source_max,
                label=f"Volumetric heat rate ({VOLUMETRIC_HEAT_RATE})",
                cmap=HEAT_SOURCE_CMAP,
                ax=ax,
            )

        # clean up the axis display
        axis, position = self.scene.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.scene.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = self.scene._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    def source_bounds(self, freq: float = None) -> Tuple[float, float]:
        """Compute range of heat sources present in the simulation."""

        rate_list = [
            source.rate
            for source in self.heat_sources
            if isinstance(source, UniformHeatSource)
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
        ax = self.scene.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax
