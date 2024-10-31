"""Container holding about the geometry and medium properties common to all types of simulations."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Union

import autograd.numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import pydantic.v1 as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..constants import THERMAL_CONDUCTIVITY, inf
from ..exceptions import SetupError, Tidy3dError
from ..log import log
from .base import Tidy3dBaseModel, cached_property
from .data.dataset import (
    CustomSpatialDataType,
    TetrahedralGridDataset,
    TriangularGridDataset,
    UnstructuredGridDataset,
    _get_numpy_array,
)
from .geometry.base import Box, ClipOperation, GeometryGroup
from .geometry.utils import flatten_groups, traverse_geometries
from .grid.grid import Coords, Grid
from .heat_spec import SolidSpec
from .medium import (
    AbstractCustomMedium,
    AbstractPerturbationMedium,
    Medium,
    Medium2D,
    MediumType,
    MediumType3D,
)
from .structure import Structure
from .types import TYPE_TAG_STR, Ax, Bound, Coordinate, InterpMethod, Shapely, Size
from .validators import assert_unique_names
from .viz import (
    MEDIUM_CMAP,
    STRUCTURE_EPS_CMAP,
    STRUCTURE_HEAT_COND_CMAP,
    PlotParams,
    add_ax_if_none,
    equal_aspect,
    plot_params_fluid,
    plot_params_structure,
    polygon_path,
)

# maximum number of mediums supported
MAX_NUM_MEDIUMS = 65530

# maximum geometry count in a single structure
MAX_GEOMETRY_COUNT = 100


class Scene(Tidy3dBaseModel):
    """Contains generic information about the geometry and medium properties common to all types of
    simulations.

    Example
    -------
    >>> sim = Scene(
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     medium=Medium(permittivity=3.0),
    ... )
    """

    medium: MediumType3D = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium of scene, defaults to vacuum if not specified.",
        discriminator=TYPE_TAG_STR,
    )

    structures: Tuple[Structure, ...] = pd.Field(
        (),
        title="Structures",
        description="Tuple of structures present in scene. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )

    """ Validating setup """

    # make sure all names are unique
    _unique_structure_names = assert_unique_names("structures")

    @pd.validator("structures", always=True)
    def _validate_num_mediums(cls, val):
        """Error if too many mediums present."""

        if val is None:
            return val

        mediums = {structure.medium for structure in val}
        if len(mediums) > MAX_NUM_MEDIUMS:
            raise SetupError(
                f"Tidy3D only supports {MAX_NUM_MEDIUMS} distinct mediums."
                f"{len(mediums)} were supplied."
            )

        return val

    @pd.validator("structures", always=True)
    def _validate_num_geometries(cls, val):
        """Error if too many geometries in a single structure."""

        if val is None:
            return val

        for i, structure in enumerate(val):
            for geometry in flatten_groups(structure.geometry, flatten_transformed=True):
                count = sum(
                    1
                    for g in traverse_geometries(geometry)
                    if not isinstance(g, (GeometryGroup, ClipOperation))
                )
                if count > MAX_GEOMETRY_COUNT:
                    raise SetupError(
                        f"Structure at 'structures[{i}]' has {count} geometries that cannot be "
                        f"flattened. A maximum of {MAX_GEOMETRY_COUNT} is supported due to "
                        f"preprocessing performance."
                    )

        return val

    """ Accounting """

    @cached_property
    def bounds(self) -> Bound:
        """Automatically defined scene's bounds based on present structures. Infinite dimensions
        are ignored. If the scene contains no structures, the bounds are set to
        (-1, -1, -1), (1, 1, 1). Similarly, if along a given axis all structures extend infinitely,
        the bounds along that axis are set from -1 to 1.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float, float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        bounds = tuple(structure.geometry.bounds for structure in self.structures)
        return (
            tuple(min((b[i] for b, _ in bounds if b[i] != -inf), default=-1) for i in range(3)),
            tuple(max((b[i] for _, b in bounds if b[i] != inf), default=1) for i in range(3)),
        )

    @cached_property
    def size(self) -> Size:
        """Automatically defined scene's size.

        Returns
        -------
        Tuple[float, float, float]
            Scene's size.
        """

        return tuple(bmax - bmin for bmin, bmax in zip(self.bounds[0], self.bounds[1]))

    @cached_property
    def center(self) -> Coordinate:
        """Automatically defined scene's center.

        Returns
        -------
        Tuple[float, float, float]
            Scene's center.
        """

        return tuple(0.5 * (bmin + bmax) for bmin, bmax in zip(self.bounds[0], self.bounds[1]))

    @cached_property
    def box(self) -> Box:
        """Automatically defined scene's :class:`.Box`.

        Returns
        -------
        Box
            Scene's box.
        """

        return Box(center=self.center, size=self.size)

    @cached_property
    def mediums(self) -> Set[MediumType]:
        """Returns set of distinct :class:`.AbstractMedium` in scene.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums in the scene.
        """
        medium_dict = {self.medium: None}
        medium_dict.update({structure.medium: None for structure in self.structures})
        return list(medium_dict.keys())

    @cached_property
    def medium_map(self) -> Dict[MediumType, pd.NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`.AbstractMedium` in scene.

        Returns
        -------
        Dict[:class:`.AbstractMedium`, int]
            Mapping between distinct mediums to index in scene.
        """

        return {medium: index for index, medium in enumerate(self.mediums)}

    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.Scene`."""
        geometry = Box(size=(inf, inf, inf))
        return Structure(geometry=geometry, medium=self.medium)

    @staticmethod
    def intersecting_media(
        test_object: Box, structures: Tuple[Structure, ...]
    ) -> Tuple[MediumType, ...]:
        """From a given list of structures, returns a list of :class:`.AbstractMedium` associated
        with those structures that intersect with the ``test_object``, if it is a surface, or its
        surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : List[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums that intersect with the given planar object.
        """
        structures = [s.to_static() for s in structures]
        if test_object.size.count(0.0) == 1:
            # get all merged structures on the test_object, which is already planar
            structures_merged = Scene._filter_structures_plane_medium(structures, test_object)
            mediums = {medium for medium, _ in structures_merged}
            return mediums

        # if the test object is a volume, test each surface recursively
        surfaces = test_object.surfaces_with_exclusion(**test_object.dict())
        mediums = set()
        for surface in surfaces:
            _mediums = Scene.intersecting_media(surface, structures)
            mediums.update(_mediums)
        return mediums

    @staticmethod
    def intersecting_structures(
        test_object: Box, structures: Tuple[Structure, ...]
    ) -> Tuple[Structure, ...]:
        """From a given list of structures, returns a list of :class:`.Structure` that intersect
        with the ``test_object``, if it is a surface, or its surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : List[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        List[:class:`.Structure`]
            Set of distinct structures that intersect with the given surface, or with the surfaces
            of the given volume.
        """
        if test_object.size.count(0.0) == 1:
            # get all merged structures on the test_object, which is already planar
            normal_axis_index = test_object.size.index(0.0)
            dim = "xyz"[normal_axis_index]
            pos = test_object.center[normal_axis_index]
            xyz_kwargs = {dim: pos}

            structures_merged = []
            for structure in structures:
                intersections = structure.geometry.intersections_plane(**xyz_kwargs)
                if len(intersections) > 0:
                    structures_merged.append(structure)
            return structures_merged

        # if the test object is a volume, test each surface recursively
        surfaces = test_object.surfaces_with_exclusion(**test_object.dict())
        structures_merged = []
        for surface in surfaces:
            structures_merged += Scene.intersecting_structures(surface, structures)
        return structures_merged

    """ Plotting General """

    @staticmethod
    def _get_plot_lims(
        bounds: Bound,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # if no hlim and/or vlim given, the bounds will then be the usual pml bounds
        axis, _ = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (hmin, vmin) = Box.pop_axis(bounds[0], axis=axis)
        _, (hmax, vmax) = Box.pop_axis(bounds[1], axis=axis)

        # account for unordered limits
        if hlim is None:
            hlim = (hmin, hmax)
        if vlim is None:
            vlim = (vmin, vmax)

        if hlim[0] > hlim[1]:
            raise Tidy3dError("Error: 'hmin' > 'hmax'")
        if vlim[0] > vlim[1]:
            raise Tidy3dError("Error: 'vmin' > 'vmax'")

        return hlim, vlim

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot each of scene's components on a plane defined by one nonzero x,y,z coordinate.

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
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(bounds=self.bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of scene's structures on a plane defined by one nonzero x,y,z coordinate.

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
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        medium_shapes = self._get_structures_2dbox(
            structures=self.structures, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        medium_map = self.medium_map
        for medium, shape in medium_shapes:
            mat_index = medium_map[medium]
            ax = self._plot_shape_structure(medium=medium, mat_index=mat_index, shape=shape, ax=ax)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        return ax

    def _plot_shape_structure(self, medium: Medium, mat_index: int, shape: Shapely, ax: Ax) -> Ax:
        """Plot a structure's cross section shape for a given medium."""
        plot_params_struct = self._get_structure_plot_params(medium=medium, mat_index=mat_index)
        ax = self.box.plot_shape(shape=shape, plot_params=plot_params_struct, ax=ax)
        return ax

    def _get_structure_plot_params(self, mat_index: int, medium: Medium) -> PlotParams:
        """Constructs the plot parameters for a given medium in scene.plot()."""

        plot_params = plot_params_structure.copy(update={"linewidth": 0})

        if mat_index == 0 or medium == self.medium:
            # background medium
            plot_params = plot_params.copy(update={"facecolor": "white", "edgecolor": "white"})
        elif medium.is_pec:
            # perfect electrical conductor
            plot_params = plot_params.copy(
                update={"facecolor": "gold", "edgecolor": "k", "linewidth": 1}
            )
        elif medium.is_time_modulated:
            # time modulated medium
            plot_params = plot_params.copy(
                update={"facecolor": "red", "linewidth": 0, "hatch": "x*"}
            )
        elif isinstance(medium, Medium2D):
            # 2d material
            plot_params = plot_params.copy(update={"edgecolor": "k", "linewidth": 1})
        else:
            # regular medium
            facecolor = MEDIUM_CMAP[(mat_index - 1) % len(MEDIUM_CMAP)]
            plot_params = plot_params.copy(update={"facecolor": facecolor})

        return plot_params

    @staticmethod
    def _add_cbar(vmin: float, vmax: float, label: str, cmap: str, ax: Ax = None) -> None:
        """Add a colorbar to plot."""
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mappable, cax=cax, label=label)

    @staticmethod
    def _set_plot_bounds(
        bounds: Bound,
        ax: Ax,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Sets the xy limits of the scene at a plane, useful after plotting.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to set bounds on.
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
        Returns
        -------
        matplotlib.axes._subplots.Axes
            The axes after setting the boundaries.
        """

        hlim, vlim = Scene._get_plot_lims(bounds=bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax.set_xlim(hlim)
        ax.set_ylim(vlim)
        return ax

    def _get_structures_2dbox(
        self,
        structures: List[Structure],
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on 2d box specified by (x_min, x_max), (y_min, y_max).

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
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

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane.
        """
        # if no hlim and/or vlim given, the bounds will then be the usual pml bounds
        axis, _ = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (hmin, vmin) = Box.pop_axis(self.bounds[0], axis=axis)
        _, (hmax, vmax) = Box.pop_axis(self.bounds[1], axis=axis)

        if hlim is not None:
            (hmin, hmax) = hlim
        if vlim is not None:
            (vmin, vmax) = vlim

        # get center and size with h, v
        h_center = (hmin + hmax) / 2.0
        v_center = (vmin + vmax) / 2.0
        h_size = (hmax - hmin) or inf
        v_size = (vmax - vmin) or inf

        axis, center_normal = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(center_normal, (h_center, v_center), axis=axis)
        size = Box.unpop_axis(0.0, (h_size, v_size), axis=axis)
        plane = Box(center=center, size=size)

        medium_shapes = []
        for structure in structures:
            intersections = plane.intersections_with(structure.geometry)
            for shape in intersections:
                if not shape.is_empty:
                    shape = Box.evaluate_inf_shape(shape)
                    medium_shapes.append((structure.medium, shape))
        return medium_shapes

    @staticmethod
    def _filter_structures_plane_medium(
        structures: List[Structure], plane: Box
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane. Overlaps are removed or merged depending on
        medium.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            List of structures to filter on the plane.
        plane : Box
            Plane specification.

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane after merging.
        """

        medium_list = [structure.medium for structure in structures]
        return Scene._filter_structures_plane(
            structures=structures, plane=plane, property_list=medium_list
        )

    @staticmethod
    def _filter_structures_plane(
        structures: List[Structure],
        plane: Box,
        property_list: List,
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane. Overlaps are removed or merged depending on
        provided property_list.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            List of structures to filter on the plane.
        plane : Box
            Plane specification.
        property_list : List = None
            Property value for each structure.

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and their property value on the plane after merging.
        """

        if len(structures) != len(property_list):
            raise SetupError(
                "Number of provided property values is not equal to the number of structures."
            )

        shapes = []
        for structure, prop in zip(structures, property_list):
            # get list of Shapely shapes that intersect at the plane
            shapes_plane = plane.intersections_with(structure.geometry)

            # Append each of them and their property information to the list of shapes
            for shape in shapes_plane:
                shapes.append((prop, shape, shape.bounds))

        background_shapes = []
        for prop, shape, bounds in shapes:
            minx, miny, maxx, maxy = bounds

            # loop through background_shapes (note: all background are non-intersecting or merged)
            for index, (_prop, _shape, _bounds) in enumerate(background_shapes):
                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _shape.is_empty or not shape.intersects(_shape):
                    continue

                diff_shape = _shape - shape

                # different prop, remove intersection from background shape
                if prop != _prop and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_prop, diff_shape, diff_shape.bounds)

                # same prop, add diff shape to this shape and mark background shape for removal
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            background_shapes.append((prop, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out any remaining None or empty shapes (shapes with area completely removed)
        return [(prop, shape) for (prop, shape, _) in background_shapes if shape]

    """ Plotting Optical """

    @equal_aspect
    @add_ax_if_none
    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of scene's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(bounds=self.bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        ax = self.plot_structures_eps(
            freq=freq, cbar=True, alpha=alpha, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        eps_lim: Tuple[Union[float, None], Union[float, None]] = (None, None),
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        grid: Grid = None,
    ) -> Ax:
        """Plot each of scene's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        eps_lim : Tuple[float, float] = None
            Custom limits for eps coloring.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        structures = self.structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        if alpha < 1 and not isinstance(self.medium, AbstractCustomMedium):
            axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
            center = Box.unpop_axis(position, (0, 0), axis=axis)
            size = Box.unpop_axis(0, (inf, inf), axis=axis)
            plane = Box(center=center, size=size)
            medium_shapes = self._filter_structures_plane_medium(structures=structures, plane=plane)
        else:
            structures = [self.background_structure] + list(structures)
            medium_shapes = self._get_structures_2dbox(
                structures=structures, x=x, y=y, z=z, hlim=hlim, vlim=vlim
            )

        eps_min, eps_max = eps_lim

        if eps_min is None or eps_max is None:
            eps_min_sim, eps_max_sim = self.eps_bounds(freq=freq)

            if eps_min is None:
                eps_min = eps_min_sim

            if eps_max is None:
                eps_max = eps_max_sim

        for medium, shape in medium_shapes:
            # if the background medium is custom medium, it needs to be rendered separately
            if medium == self.medium and alpha < 1 and not isinstance(medium, AbstractCustomMedium):
                continue
            # no need to add patches for custom medium
            if not isinstance(medium, AbstractCustomMedium):
                ax = self._plot_shape_structure_eps(
                    freq=freq,
                    alpha=alpha,
                    medium=medium,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    reverse=reverse,
                    shape=shape,
                    ax=ax,
                )
            else:
                # For custom medium, apply pcolormesh clipped by the shape.
                self._pcolormesh_shape_custom_medium_structure_eps(
                    x, y, z, freq, alpha, medium, eps_min, eps_max, reverse, shape, ax, grid
                )

        if cbar:
            self._add_cbar_eps(eps_min=eps_min, eps_max=eps_max, ax=ax)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        return ax

    @staticmethod
    def _add_cbar_eps(eps_min: float, eps_max: float, ax: Ax = None) -> None:
        """Add a permittivity colorbar to plot."""
        Scene._add_cbar(
            vmin=eps_min, vmax=eps_max, label=r"$\epsilon_r$", cmap=STRUCTURE_EPS_CMAP, ax=ax
        )

    @staticmethod
    def _eps_bounds(medium_list: list[Medium], freq: float = None) -> Tuple[float, float]:
        """Compute range of (real) permittivity present in the mediums at frequency "freq"."""
        medium_list = [medium for medium in medium_list if not medium.is_pec]
        # regular medium
        eps_list = [
            np.real(medium.eps_model(freq))
            for medium in medium_list
            if not isinstance(medium, AbstractCustomMedium) and not isinstance(medium, Medium2D)
        ]
        eps_min = min(eps_list, default=1)
        eps_max = max(eps_list, default=1)
        # custom medium, the min and max in the supplied dataset over all components and
        # spatial locations.
        for mat in [medium for medium in medium_list if isinstance(medium, AbstractCustomMedium)]:
            eps_dataarray = mat.eps_dataarray_freq(freq)
            eps_min = min(
                eps_min,
                min(
                    np.min(_get_numpy_array(np.real(eps_comp)).ravel())
                    for eps_comp in eps_dataarray
                ),
            )
            eps_max = max(
                eps_max,
                max(
                    np.max(_get_numpy_array(np.real(eps_comp)).ravel())
                    for eps_comp in eps_dataarray
                ),
            )
        return eps_min, eps_max

    def eps_bounds(self, freq: float = None) -> Tuple[float, float]:
        """Compute range of (real) permittivity present in the scene at frequency "freq".

        Parameters
        ----------
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        Tuple[float, float]
            Minimal and maximal values of relative permittivity in scene.
        """

        medium_list = [self.medium] + list(self.mediums)
        return self._eps_bounds(medium_list=medium_list, freq=freq)

    def _pcolormesh_shape_custom_medium_structure_eps(
        self,
        x: float,
        y: float,
        z: float,
        freq: float,
        alpha: float,
        medium: Medium,
        eps_min: float,
        eps_max: float,
        reverse: bool,
        shape: Shapely,
        ax: Ax,
        grid: Grid,
    ):
        """
        Plot shape made of custom medium with ``pcolormesh``.
        """
        coords = "xyz"
        normal_axis_ind, normal_position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        normal_axis, plane_axes = Box.pop_axis(coords, normal_axis_ind)

        # make grid for eps interpolation
        # we will do this by combining shape bounds and points where custom eps is provided
        shape_bounds = shape.bounds
        rmin, rmax = [*shape_bounds[:2]], [*shape_bounds[2:]]
        rmin.insert(normal_axis_ind, normal_position)
        rmax.insert(normal_axis_ind, normal_position)

        if grid is None:
            plane_axes_inds = [0, 1, 2]
            plane_axes_inds.pop(normal_axis_ind)

            eps_diag = medium.eps_dataarray_freq(frequency=freq)

            # handle unstructured data case
            if isinstance(eps_diag[0], UnstructuredGridDataset):
                if (
                    isinstance(eps_diag[0], TriangularGridDataset)
                    and eps_diag[0].normal_axis != normal_axis_ind
                ):
                    # if we trying to visualize 2d unstructured data not along its normal direction
                    # we need to extract line slice that lies in the visualization plane
                    # note that after this eps_diag[] will be SpatialDataArray's
                    eps_diag = list(eps_diag)
                    for dim in range(3):
                        eps_diag[dim] = eps_diag[dim].plane_slice(
                            axis=normal_axis_ind, pos=normal_position
                        )
                else:
                    eps_mean = (eps_diag[0] + eps_diag[1] + eps_diag[2]) / 3

                    if isinstance(eps_mean, TetrahedralGridDataset):
                        # extract slice if volumetric unstructured data
                        eps_mean = eps_mean.plane_slice(axis=normal_axis_ind, pos=normal_position)

                    if reverse:
                        eps_mean = eps_min + eps_max - eps_mean

                    # at this point eps_mean is TriangularGridDataset and we just plot it directly
                    # with applying shape mask
                    eps_mean.plot(
                        grid=False,
                        ax=ax,
                        cbar=False,
                        cmap=STRUCTURE_EPS_CMAP,
                        vmin=eps_min,
                        vmax=eps_max,
                        pcolor_kwargs=dict(
                            clip_path=(polygon_path(shape), ax.transData),
                            clip_box=ax.bbox,
                            alpha=alpha,
                        ),
                    )
                    return

            # in case when different components of custom medium are defined on different grids
            # we will combine all points along each dimension
            if (
                eps_diag[0].coords == eps_diag[1].coords
                and eps_diag[0].coords == eps_diag[2].coords
            ):
                coords_to_insert = [eps_diag[0].coords]
            else:
                coords_to_insert = [eps_diag[0].coords, eps_diag[1].coords, eps_diag[2].coords]

            # actual combining of points along each of plane dimensions
            plane_coord = []
            for ind, comp in zip(plane_axes_inds, plane_axes):
                # first start with an array made of shapes bounds
                axis_coords = np.array([rmin[ind], rmax[ind]])
                # now add points in between them
                for coords in coords_to_insert:
                    comp_axis_coords = coords[comp]
                    inds_inside_shape = np.where(
                        np.logical_and(comp_axis_coords > rmin[ind], comp_axis_coords < rmax[ind])
                    )[0]
                    if len(inds_inside_shape) > 0:
                        axis_coords = np.concatenate(
                            (axis_coords, comp_axis_coords[inds_inside_shape])
                        )
                # remove duplicates
                axis_coords = np.unique(axis_coords)

                plane_coord.append(axis_coords)
        else:
            span_inds = grid.discretize_inds(Box.from_bounds(rmin=rmin, rmax=rmax), extend=True)
            # filter negative or too large inds
            n_grid = [len(grid_comp) for grid_comp in grid.boundaries.to_list]
            span_inds = [
                (max(fmin, 0), min(fmax, n_grid[f_ind]))
                for f_ind, (fmin, fmax) in enumerate(span_inds)
            ]

            # assemble the coordinate in the 2d plane
            plane_coord = []
            for plane_axis in range(2):
                ind_axis = "xyz".index(plane_axes[plane_axis])
                plane_coord.append(grid.boundaries.to_list[ind_axis][slice(*span_inds[ind_axis])])

        # prepare `Coords` for interpolation
        coord_dict = {
            plane_axes[0]: plane_coord[0],
            plane_axes[1]: plane_coord[1],
            normal_axis: [normal_position],
        }
        coord_shape = Coords(**coord_dict)
        # interpolate permittivity and take the average over components
        eps_shape = np.mean(medium.eps_diagonal_on_grid(frequency=freq, coords=coord_shape), axis=0)
        # remove the normal_axis and take real part
        eps_shape = eps_shape.real.mean(axis=normal_axis_ind)
        # reverse
        if reverse:
            eps_shape = eps_min + eps_max - eps_shape

        # pcolormesh
        plane_xp, plane_yp = np.meshgrid(plane_coord[0], plane_coord[1], indexing="ij")
        ax.pcolormesh(
            plane_xp,
            plane_yp,
            eps_shape,
            clip_path=(polygon_path(shape), ax.transData),
            cmap=STRUCTURE_EPS_CMAP,
            vmin=eps_min,
            vmax=eps_max,
            alpha=alpha,
            clip_box=ax.bbox,
        )

    @staticmethod
    def _get_structure_eps_plot_params(
        medium: Medium,
        freq: float,
        eps_min: float,
        eps_max: float,
        reverse: bool = False,
        alpha: float = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in scene.plot_eps()."""

        plot_params = plot_params_structure.copy(update={"linewidth": 0})
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if medium.is_pec:
            # perfect electrical conductor
            plot_params = plot_params.copy(
                update={"facecolor": "gold", "edgecolor": "k", "linewidth": 1}
            )
        elif isinstance(medium, Medium2D):
            # 2d material
            plot_params = plot_params.copy(update={"edgecolor": "k", "linewidth": 1})
        else:
            # regular medium
            eps_medium = medium.eps_model(frequency=freq).real
            delta_eps = eps_medium - eps_min
            delta_eps_max = eps_max - eps_min + 1e-5
            eps_fraction = delta_eps / delta_eps_max
            color = eps_fraction if reverse else 1 - eps_fraction
            color = min(1, max(color, 0))  # clip in case of custom eps limits
            plot_params = plot_params.copy(update={"facecolor": str(color)})

        return plot_params

    def _plot_shape_structure_eps(
        self,
        freq: float,
        medium: Medium,
        shape: Shapely,
        eps_min: float,
        eps_max: float,
        ax: Ax,
        reverse: bool = False,
        alpha: float = None,
    ) -> Ax:
        """Plot a structure's cross section shape for a given medium, grayscale for permittivity."""
        plot_params = self._get_structure_eps_plot_params(
            medium=medium, freq=freq, eps_min=eps_min, eps_max=eps_max, alpha=alpha, reverse=reverse
        )
        ax = self.box.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax

    """ Plotting Heat """

    @equal_aspect
    @add_ax_if_none
    def plot_heat_conductivity(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        alpha: float = None,
        cbar: bool = True,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of scebe's components on a plane defined by one nonzero x,y,z coordinate.
        The thermal conductivity is plotted in grayscale based on its value.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        cbar : bool = True
            Whether to plot a colorbar for the thermal conductivity.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(bounds=self.bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        ax = self.plot_structures_heat_conductivity(
            cbar=cbar, alpha=alpha, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures_heat_conductivity(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of scene's structures on a plane defined by one nonzero x,y,z coordinate.
        The thermal conductivity is plotted in grayscale based on its value.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        structures = self.structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        if alpha < 1:
            axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
            center = Box.unpop_axis(position, (0, 0), axis=axis)
            size = Box.unpop_axis(0, (inf, inf), axis=axis)
            plane = Box(center=center, size=size)
            medium_shapes = self._filter_structures_plane_medium(structures=structures, plane=plane)
        else:
            structures = [self.background_structure] + list(structures)
            medium_shapes = self._get_structures_2dbox(
                structures=structures, x=x, y=y, z=z, hlim=hlim, vlim=vlim
            )

        heat_cond_min, heat_cond_max = self.heat_conductivity_bounds()
        for medium, shape in medium_shapes:
            ax = self._plot_shape_structure_heat_cond(
                alpha=alpha,
                medium=medium,
                heat_cond_min=heat_cond_min,
                heat_cond_max=heat_cond_max,
                reverse=reverse,
                shape=shape,
                ax=ax,
            )

        if cbar:
            self._add_cbar(
                vmin=heat_cond_min,
                vmax=heat_cond_max,
                label=f"Thermal conductivity ({THERMAL_CONDUCTIVITY})",
                cmap=STRUCTURE_HEAT_COND_CMAP,
                ax=ax,
            )
        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    def heat_conductivity_bounds(self) -> Tuple[float, float]:
        """Compute range of thermal conductivities present in the scene.

        Returns
        -------
        Tuple[float, float]
            Minimal and maximal values of thermal conductivity in scene.
        """

        medium_list = [self.medium] + list(self.mediums)
        medium_list = [medium for medium in medium_list if isinstance(medium.heat_spec, SolidSpec)]
        cond_list = [medium.heat_spec.conductivity for medium in medium_list]
        cond_min = min(cond_list)
        cond_max = max(cond_list)
        return cond_min, cond_max

    def _get_structure_heat_cond_plot_params(
        self,
        medium: Medium,
        heat_cond_min: float,
        heat_cond_max: float,
        reverse: bool = False,
        alpha: float = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in
        scene.plot_heat_conductivity().
        """

        plot_params = plot_params_structure.copy(update={"linewidth": 0})
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if isinstance(medium.heat_spec, SolidSpec):
            # regular medium
            cond_medium = medium.heat_spec.conductivity
            delta_cond = cond_medium - heat_cond_min
            delta_cond_max = heat_cond_max - heat_cond_min + 1e-5 * heat_cond_min
            cond_fraction = delta_cond / delta_cond_max
            color = cond_fraction if reverse else 1 - cond_fraction
            plot_params = plot_params.copy(update={"facecolor": str(color)})
        else:
            plot_params = plot_params_fluid
            if alpha is not None:
                plot_params = plot_params.copy(update={"alpha": alpha})

        return plot_params

    def _plot_shape_structure_heat_cond(
        self,
        medium: Medium,
        shape: Shapely,
        heat_cond_min: float,
        heat_cond_max: float,
        ax: Ax,
        reverse: bool = False,
        alpha: float = None,
    ) -> Ax:
        """Plot a structure's cross section shape for a given medium, grayscale for thermal
        conductivity.
        """
        plot_params = self._get_structure_heat_cond_plot_params(
            medium=medium,
            heat_cond_min=heat_cond_min,
            heat_cond_max=heat_cond_max,
            alpha=alpha,
            reverse=reverse,
        )
        ax = self.box.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax

    """ Misc """

    def perturbed_mediums_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Scene:
        """Return a copy of the scene with heat and/or charge data applied to all mediums
        that have perturbation models specified. That is, such mediums will be replaced with
        spatially dependent custom mediums that reflect perturbation effects. Any of temperature,
        electron_density, and hole_density can be ``None``. All provided fields must have identical
        coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        :class:`.Scene`
            Simulation after application of heat and/or charge data.
        """

        scene_dict = self.dict()
        structures = self.structures
        array_dict = {
            "temperature": temperature,
            "electron_density": electron_density,
            "hole_density": hole_density,
        }

        # For each structure made of mediums with perturbation models, convert those mediums into
        # spatially dependent mediums by selecting minimal amount of heat and charge data points
        # covering the structure, and create a new structure containing the resulting custom medium
        new_structures = []
        for s_ind, structure in enumerate(structures):
            med = structure.medium
            if isinstance(med, AbstractPerturbationMedium):
                # get structure's bounding box
                bounds = structure.geometry.bounds

                # for each structure select a minimal subset of data that covers it
                restricted_arrays = {}

                for name, array in array_dict.items():
                    if array is not None:
                        restricted_arrays[name] = array.sel_inside(bounds)

                        # check provided data fully cover structure
                        if not array.does_cover(bounds):
                            log.warning(
                                f"Provided '{name}' does not fully cover structures[{s_ind}]."
                            )

                new_medium = med.perturbed_copy(**restricted_arrays, interp_method=interp_method)
                new_structure = structure.updated_copy(medium=new_medium)
                new_structures.append(new_structure)
            else:
                new_structures.append(structure)

        scene_dict["structures"] = new_structures

        # do the same for background medium if it a medium with perturbation models.
        med = self.medium
        if isinstance(med, AbstractPerturbationMedium):
            scene_dict["medium"] = med.perturbed_copy(**array_dict, interp_method=interp_method)

        return Scene.parse_obj(scene_dict)
