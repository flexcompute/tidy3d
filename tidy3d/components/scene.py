""" Container holding about the geometry and medium properties common to all types of simulations.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Set, Union

import pydantic.v1 as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import cached_property, Tidy3dBaseModel
from .validators import assert_unique_names
from .geometry.base import Box
from .geometry.mesh import TriangleMesh
from .types import Ax, Shapely, TYPE_TAG_STR, Bound, Size, Coordinate, InterpMethod
from .medium import Medium, MediumType, PECMedium
from .medium import AbstractCustomMedium, Medium2D, MediumType3D
from .medium import AnisotropicMedium, AbstractPerturbationMedium
from .structure import Structure
from .data.dataset import Dataset
from .data.data_array import SpatialDataArray
from .viz import add_ax_if_none, equal_aspect
from .grid.grid import Coords
from .heat_spec import SolidSpec

from .viz import MEDIUM_CMAP, STRUCTURE_EPS_CMAP, PlotParams, polygon_path, STRUCTURE_HEAT_COND_CMAP
from .viz import plot_params_structure, plot_params_fluid

from ..constants import inf, THERMAL_CONDUCTIVITY
from ..exceptions import SetupError, Tidy3dError
from ..log import log

# maximum number of mediums supported
MAX_NUM_MEDIUMS = 65530


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

    """ Accounting """

    @cached_property
    def bounds(self) -> Bound:
        """Automatically defined scene's bounds based on present structures. Infinite dimensions
        are ignored. If the scene contains no strucutres, the bounds are set to
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

        medium_shapes = self._get_structures_plane(structures=self.structures, x=x, y=y, z=z)
        medium_map = self.medium_map

        for (medium, shape) in medium_shapes:
            mat_index = medium_map[medium]
            ax = self._plot_shape_structure(medium=medium, mat_index=mat_index, shape=shape, ax=ax)

        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

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
        elif isinstance(medium, PECMedium):
            # perfect electrical conductor
            plot_params = plot_params.copy(
                update={"facecolor": "gold", "edgecolor": "k", "linewidth": 1}
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

    @staticmethod
    def _get_structures_plane(
        structures: List[Structure], x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.

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

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane.
        """
        medium_shapes = []
        for structure in structures:
            intersections = structure.geometry.intersections_plane(x=x, y=y, z=z)
            if len(intersections) > 0:
                for shape in intersections:
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
        structures = [self.background_structure] + list(structures)

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
            medium_shapes = self._get_structures_plane(structures=structures, x=x, y=y, z=z)

        eps_min, eps_max = eps_lim

        if eps_min is None or eps_max is None:

            eps_min_sim, eps_max_sim = self.eps_bounds(freq=freq)

            if eps_min is None:
                eps_min = eps_min_sim

            if eps_max is None:
                eps_max = eps_max_sim

        for (medium, shape) in medium_shapes:
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
                    x, y, z, freq, alpha, medium, eps_min, eps_max, reverse, shape, ax
                )

        if cbar:
            self._add_cbar_eps(eps_min=eps_min, eps_max=eps_max, ax=ax)
        ax = self._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    @staticmethod
    def _add_cbar_eps(eps_min: float, eps_max: float, ax: Ax = None) -> None:
        """Add a permittivity colorbar to plot."""
        Scene._add_cbar(
            vmin=eps_min, vmax=eps_max, label=r"$\epsilon_r$", cmap=STRUCTURE_EPS_CMAP, ax=ax
        )

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
        medium_list = [medium for medium in medium_list if not isinstance(medium, PECMedium)]
        # regular medium
        eps_list = [
            medium.eps_model(freq).real
            for medium in medium_list
            if not isinstance(medium, AbstractCustomMedium)
        ]
        eps_list.append(1)
        eps_min = min(eps_list)
        eps_max = max(eps_list)
        # custom medium, the min and max in the supplied dataset over all components and
        # spatial locations.
        for mat in [medium for medium in medium_list if isinstance(medium, AbstractCustomMedium)]:
            eps_dataarray = mat.eps_dataarray_freq(freq)
            eps_min = min(
                eps_min,
                min(np.min(eps_comp.real.values.ravel()) for eps_comp in eps_dataarray),
            )
            eps_max = max(
                eps_max,
                max(np.max(eps_comp.real.values.ravel()) for eps_comp in eps_dataarray),
            )
        return eps_min, eps_max

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
    ):
        """
        Plot shape made of custom medium with ``pcolormesh``.
        """
        coords = "xyz"
        normal_axis_ind, normal_position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        normal_axis, plane_axes = Box.pop_axis(coords, normal_axis_ind)
        plane_axes_inds = [0, 1, 2]
        plane_axes_inds.pop(normal_axis_ind)

        # make grid for eps interpolation
        # we will do this by combining shape bounds and points where custom eps is provided
        shape_bounds = shape.bounds
        rmin, rmax = [*shape_bounds[:2]], [*shape_bounds[2:]]
        rmin.insert(normal_axis_ind, normal_position)
        rmax.insert(normal_axis_ind, normal_position)

        # in case when different components of custom medium are defined on different grids
        # we will combine all points along each dimension
        eps_diag = medium.eps_dataarray_freq(frequency=freq)
        if eps_diag[0].coords == eps_diag[1].coords and eps_diag[0].coords == eps_diag[2].coords:
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
                    axis_coords = np.concatenate((axis_coords, comp_axis_coords[inds_inside_shape]))
            # remove duplicates
            axis_coords = np.unique(axis_coords)

            plane_coord.append(axis_coords)

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

    def _get_structure_eps_plot_params(
        self,
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

        if isinstance(medium, PECMedium):
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
        structures = [self.background_structure] + list(structures)

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
            medium_shapes = self._get_structures_plane(structures=structures, x=x, y=y, z=z)

        heat_cond_min, heat_cond_max = self.heat_conductivity_bounds()
        for (medium, shape) in medium_shapes:
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

    @property
    def custom_datasets(self) -> List[Dataset]:
        """List of custom datasets for verification purposes. If the list is not empty, then
        the scene needs to be exported to hdf5 to store the data.
        """
        datasets_medium = [mat for mat in self.mediums if isinstance(mat, AbstractCustomMedium)]
        datasets_geometry = [
            struct.geometry.mesh_dataset
            for struct in self.structures
            if isinstance(struct.geometry, TriangleMesh)
        ]
        return datasets_medium + datasets_geometry

    @cached_property
    def allow_gain(self) -> bool:
        """``True`` if any of the mediums in the scene allows gain."""

        for medium in self.mediums:
            if isinstance(medium, AnisotropicMedium):
                if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                    return True
            elif medium.allow_gain:
                return True
        return False

    def perturbed_mediums_copy(
        self,
        temperature: SpatialDataArray = None,
        electron_density: SpatialDataArray = None,
        hole_density: SpatialDataArray = None,
        interp_method: InterpMethod = "linear",
    ) -> Scene:
        """Return a copy of the scene with heat and/or charge data applied to all mediums
        that have perturbation models specified. That is, such mediums will be replaced with
        spatially dependent custom mediums that reflect perturbation effects. Any of temperature,
        electron_density, and hole_density can be ``None``. All provided fields must have identical
        coords.

        Parameters
        ----------
        temperature : SpatialDataArray = None
            Temperature field data.
        electron_density : SpatialDataArray = None
            Electron density field data.
        hole_density : SpatialDataArray = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Scene
            Simulation after application of heat and/or charge data.
        """

        scene_dict = self.dict()
        structures = self.structures
        scene_bounds = self.bounds
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
                s_bounds = structure.geometry.bounds

                bounds = [
                    np.max([scene_bounds[0], s_bounds[0]], axis=0),
                    np.min([scene_bounds[1], s_bounds[1]], axis=0),
                ]

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

            # get scene's bounding box
            bounds = scene_bounds

            # for each structure select a minimal subset of data that covers it
            restricted_arrays = {}

            for name, array in array_dict.items():
                if array is not None:
                    restricted_arrays[name] = array.sel_inside(bounds)

                    # check provided data fully cover scene
                    if not array.does_cover(bounds):
                        log.warning(f"Provided '{name}' does not fully cover scene domain.")

            scene_dict["medium"] = med.perturbed_copy(
                **restricted_arrays, interp_method=interp_method
            )

        return Scene.parse_obj(scene_dict)
