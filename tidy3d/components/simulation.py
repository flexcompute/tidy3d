""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List

import pydantic
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from descartes import PolygonPatch

from .types import GridSize, Symmetry, Ax, Numpy, Shapely
from .geometry import Box
from .medium import Medium, MediumType
from .structure import Structure
from .source import SourceType
from .monitor import MonitorType
from .pml import PMLLayer
from .viz import StructMediumParams, StructEpsParams, PMLParams, SymParams, add_ax_if_none
from ..constants import inf
from ..log import SetupError

# technically this is creating a circular import issue because it calls tidy3d/__init__.py
# from .. import __version__ as version_number


class Simulation(Box):
    """Contains all information about simulation.

    Parameters
    ----------
    center : Tuple[float, float, float], optional
        Center of simulation domain in x,y,z, defualts to (0.0, 0.0, 0.0)
    size : Tuple[float, float, float]
        Size of simulation domain in x,y,z.
    grid_size : Tuple[float, float, float]
        Grid size in x,y,z direction.
    run_time: float, optional
        Maximum run time of simulation in seconds, defaults to 0.0.
    medium : ``tidy3d.Medium``, optional
        Background medium of simulation, defaults to air.
    structures : List[``tidy3d.Structure``], optional
        Structures in simulation, in case of overlap, prefernce goes to those later in list.
    sources : Dict[str: ``Source``], optional
        Names and sources in the simulation.
    monitors :
        Names and monitors in the simulation.
    pml_layers: Tuple[``tidy3d.PMLLayer``, ``PMLLayer``, ``PMLLayer``], optional.
        Specifies PML layers (aborbers) in x,y,z location, defaults to no PML
    symmetry : Tuple[int], optional
        Specifies symmetry in x,y,z with values 0, 1, -1 specifying no symmetry, even symmetry, and
        odd symmetry, respectively.
    shutoff : float, optional
        Simulation ends when field intensity gets below this value, defaults to 1e-5
    courant : float, optional
        Courant stability factor, controls time step to spatial step ratio, defaults to 0.9.
    subpixel : bool, optional
        Uses subpixel averaging of permittivity if True for much higher accuracy, defaults to True.
    """

    """TODO: Some parameters (e.g. pml_layers, courant, shutoff) contain more information in the
    current doc version. There should probably be a more extended discussion in the documents
    proper, but we may want to keep the necessary informatino to understand the parameter here as
    well (or a link to the document section where it's discussed in more detail).
    """

    grid_size: Tuple[GridSize, GridSize, GridSize]
    medium: MediumType = Medium()
    run_time: pydantic.NonNegativeFloat = 0.0
    structures: List[Structure] = []
    sources: Dict[str, SourceType] = {}
    monitors: Dict[str, MonitorType] = {}
    pml_layers: Tuple[PMLLayer, PMLLayer, PMLLayer] = (
        PMLLayer(),
        PMLLayer(),
        PMLLayer(),
    )
    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = (0, 0, 0)
    shutoff: pydantic.NonNegativeFloat = 1e-5
    # TODO: We should see if we can safely increase courant to 0.99
    courant: pydantic.confloat(ge=0.0, le=1.0) = 0.9
    subpixel: bool = True
    # version: str = str(version_number)

    def __init__(self, **kwargs):
        """initialize sim and then do validations"""
        super().__init__(**kwargs)
        self._check_geo_objs_in_bounds()
        # to do:
        # - check sources in medium freq range
        # - check PW in homogeneous medium
        # - check nonuniform grid covers the whole simulation domain

    """ Post-Init Validation """

    def _check_geo_objs_in_bounds(self):
        """For each geometry-containing object in simulation, make sure it intersects simulation
        bounding box.
        """

        for position_index, structure in enumerate(self.structures):
            if not self.intersects(structure.geometry):
                raise SetupError(
                    f"Structure '{structure}' "
                    f"(at `structures[{position_index}]`) is outside simulation"
                )

        for geo_obj_dict in (self.sources, self.monitors):
            for name, geo_obj in geo_obj_dict.items():
                if not self.intersects(geo_obj):
                    raise SetupError(f"object '{name}' is completely outside simulation")

    """ Accounting """

    @property
    def medium_map(self) -> Dict[Medium, pydantic.NonNegativeInt]:
        """medium_map[medium] returns unique global index of medium in siulation."""
        mediums = {structure.medium for structure in self.structures}
        return {medium: index for index, medium in enumerate(mediums)}

    """ Plotting """

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's components on a plan defined by one nonzero x,y,z
        coordinate.
        """

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_eps(  # pylint: disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot each of simulation's components on a plane where structures permittivities are
        plotted in grayscale.
        """

        ax = self.plot_structures_eps(freq=freq, cbar=True, ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_structures(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plot all of simulation's structures as distinct materials."""
        medium_map = self.medium_map
        medium_shapes = self._filter_plot_structures(x=x, y=y, z=z)
        for (medium, shape) in medium_shapes:
            params_updater = StructMediumParams(medium=medium, medium_map=medium_map)
            kwargs_struct = params_updater.update_params(**kwargs)
            if medium == self.medium:
                kwargs_struct["edgecolor"] = "white"
                kwargs_struct["facecolor"] = "white"
            patch = PolygonPatch(shape, **kwargs_struct)
            ax.add_artist(patch)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @staticmethod
    def _add_cbar(eps_min: float, eps_max: float, ax: Ax = None) -> None:
        """add colorbar to eps plot"""
        norm = mpl.colors.Normalize(vmin=eps_min, vmax=eps_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap="gist_yarg")
        plt.colorbar(mappable, cax=cax, label=r"$\epsilon_r$")

    @add_ax_if_none
    def plot_structures_eps(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        cbar: bool = True,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plots all of simulation's structures as permittivity grayscale."""
        if freq is None:
            freq = inf
        eps_list = [s.medium.eps_model(freq).real for s in self.structures]
        eps_max = max(eps_list)
        eps_min = min(eps_list)
        medium_shapes = self._filter_plot_structures(x=x, y=y, z=z)
        for (medium, shape) in medium_shapes:
            eps = medium.eps_model(freq).real
            params_updater = StructEpsParams(eps=eps, eps_max=eps_max)
            kwargs_struct = params_updater.update_params(**kwargs)
            if medium == self.medium:
                kwargs_struct["edgecolor"] = "white"
            patch = PolygonPatch(shape, **kwargs_struct)
            ax.add_artist(patch)
        if cbar:
            self._add_cbar(eps_min=eps_min, eps_max=eps_max, ax=ax)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_sources(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plots each of simulation's sources on plane."""
        for _, source in self.sources.items():
            if source.intersects_plane(x=x, y=y, z=z):
                ax = source.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_monitors(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plots each of simulation's monitors on plane."""
        for _, monitor in self.monitors.items():
            if monitor.intersects_plane(x=x, y=y, z=z):
                ax = monitor.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_symmetries(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plots each of the non-zero symmetries"""
        for sym_axis, sym_value in enumerate(self.symmetry):
            if sym_value == 0:
                continue
            sym_size = [inf, inf, inf]
            sym_size[sym_axis] = inf / 2
            sym_center = list(self.center)
            sym_center[sym_axis] += sym_size[sym_axis] / 2
            sym_box = Box(center=sym_center, size=sym_size)
            if sym_box.intersects_plane(x=x, y=y, z=z):
                new_kwargs = SymParams(sym_value=sym_value).update_params(**kwargs)
                ax = sym_box.plot(ax=ax, x=x, y=y, z=z, **new_kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_pml(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plots each of simulation's PML regions"""
        kwargs = PMLParams().update_params(**kwargs)
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            if pml_layer.num_layers == 0:
                continue
            pml_height = self.grid_size[pml_axis] * pml_layer.num_layers
            pml_size = [inf, inf, inf]
            pml_size[pml_axis] = pml_height
            pml_offset_center = (self.size[pml_axis] + pml_height) / 2.0
            for sign in (-1, 1):
                pml_center = list(self.center)
                pml_center[pml_axis] += sign * pml_offset_center
                pml_box = Box(center=pml_center, size=pml_size)
                if pml_box.intersects_plane(x=x, y=y, z=z):
                    ax = pml_box.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    def set_plot_bounds(self, ax: Ax, x: float = None, y: float = None, z: float = None) -> Ax:
        """sets the xy limits of the simulation, useful after plotting"""

        axis, _ = self._parse_xyz_kwargs(x=x, y=y, z=z)
        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)

        pml_heightes = [dl * pml.num_layers for (dl, pml) in zip(self.grid_size, self.pml_layers)]

        _, (pml_thick_x, pml_thick_y) = self.pop_axis(pml_heightes, axis=axis)

        ax.set_xlim(xmin - pml_thick_x, xmax + pml_thick_x)
        ax.set_ylim(ymin - pml_thick_y, ymax + pml_thick_y)
        return ax

    def _filter_plot_structures(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of (medium, shapely) to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.
        """

        shapes = []
        for struct in self.structures:

            # dont bother with geometries that dont intersect plane
            if not struct.geometry.intersects_plane(x=x, y=y, z=z):
                continue

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = struct.geometry.intersections(x=x, y=y, z=z)

            # Append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((struct.medium, shape))

        # returned a merged list of mediums and shapes.
        return self._merge_shapes(shapes)

    @staticmethod
    def _merge_shapes(shapes: List[Tuple[Medium, Shapely]]) -> List[Tuple[Medium, Shapely]]:
        """Merge list of (Medium, Shapely) by intersection with same medium
        edit background shapes to remove volume by intersection.
        """
        background_shapes = []
        for medium, shape in shapes:

            # loop through background_shapes (note: all background are non-intersecting or merged)
            for index, (_medium, _shape) in enumerate(background_shapes):

                # if not intersetion, move onto next background shape
                if not shape & _shape:
                    continue

                # different medium, remove intersection from background shape
                if medium != _medium:
                    background_shapes[index] = (_medium, _shape - shape)

                # same medium, add background to this shape and mark background shape for removal
                else:
                    shape = shape | (_shape - shape)
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            background_shapes.append((medium, shape))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out any remaining None or empty shapes (shapes with area completely removed)
        return [(medium, shape) for (medium, shape) in background_shapes if shape]

    """ Discretization """

    def _discretize(self, box: Box) -> Numpy:  # pylint: disable=too-many-locals
        """get x,y,z positions of box using self.grid_size"""

        (xmin, ymin, zmin), (xmax, ymax, zmax) = box.bounds
        dlx, dly, dlz = self.grid_size
        x_centers = np.arange(xmin, xmax + dlx / 2, dlx)
        y_centers = np.arange(ymin, ymax + dly / 2, dly)
        z_centers = np.arange(zmin, zmax + dlz / 2, dlz)
        x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        x_x = (x + dlx / 2.0, y, z)
        y_y = (x, y + dly / 2.0, z)
        z_z = (x, y, z + dlz / 2.0)
        # magnetic field locations?
        # x_yz = (x, y + dly / 2.0, z + dlz / 2.0)
        # y_xz = (x + dlx / 2.0, y, z + dlz / 2.0)
        # z_xy = (x + dlx / 2.0, y + dly / 2.0, z)
        return np.stack((x_x, y_y, z_z), axis=0)

    def epsilon(self, box: Box, freq: float) -> Numpy:
        """get permittivity at volume specified by box and freq"""
        xyz_pts = self._discretize(box)
        eps_background = self.medium.eps_model(freq)
        eps_array = eps_background * np.ones(xyz_pts.shape[1:], dtype=complex)
        for structure in self.structures:
            geo = structure.geometry
            if not geo.intersects(box):
                continue
            eps_structure = structure.medium.eps_model(freq)
            for component_index, pts in enumerate(xyz_pts):
                x, y, z = pts
                structure_map = geo.inside(x, y, z)
                eps_array[component_index, structure_map] = eps_structure
        return eps_array
