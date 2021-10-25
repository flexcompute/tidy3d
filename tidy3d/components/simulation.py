""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List

import pydantic
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from descartes import PolygonPatch

from .types import Symmetry, Ax, Numpy, Shapely, Array, FreqBound
from .geometry import Box
from .grid import GridSpecType, Coords1D, Grid, Coords
from .medium import Medium, MediumType, eps_complex_to_nk
from .structure import Structure
from .source import SourceType
from .monitor import MonitorType
from .pml import PMLLayer
from .viz import StructMediumParams, StructEpsParams, PMLParams, SymParams, add_ax_if_none
from ..constants import inf, C_0
from ..log import log

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
    grid_specs : :class:`GridSpec` or ``float``.
        Grid specifications along x,y,z.  If ``float``, sets grid size explicitly.
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
    subpixel : bool, optional
        Uses subpixel averaging of permittivity if True for much higher accuracy, defaults to True.
    courant : float, optional
        Courant stability factor, controls time step to spatial step ratio, defaults to 0.9.
    """

    """TODO: Some parameters (e.g. pml_layers, courant, shutoff) contain more information in the
    current doc version. There should probably be a more extended discussion in the documents
    proper, but we may want to keep the necessary informatino to understand the parameter here as
    well (or a link to the document section where it's discussed in more detail).
    """

    grid_specs: Tuple[GridSpecType, GridSpecType, GridSpecType]
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
    subpixel: bool = True
    courant: pydantic.confloat(gt=0.0, le=1.0) = 0.9
    # version: str = str(version_number)

    def __init__(self, **kwargs):
        """initialize sim and then do validations"""
        super().__init__(**kwargs)
        self._check_objects_in_bounds()
        # to do:
        # - check sources in medium freq range
        # - check PW in homogeneous medium
        # - check nonuniform grid covers the whole simulation domain

    """ Post-Init Validation """

    def _check_objects_in_bounds(self):
        """For each geometry-containing object in simulation, make sure it intersects simulation
        bounding box.
        """

        for position_index, structure in enumerate(self.structures):
            if not self.intersects(structure.geometry):
                log.error(f"Structure #{position_index} '{structure}' is outside simulation.")

        for name, source in self.sources.items():
            if not self.intersects(source):
                log.error(f"Source '{name}' is completely outside simulation.")

        for name, monitor in self.monitors.items():
            if not self.intersects(monitor):
                log.error(f"Monitor '{name}' is completely outside simulation.")

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

    @property
    def pml_thicknesses(
        self,
    ) -> List[Tuple[float, float]]:
        """thicknesses (um) of PML in all three axis and directions (+, -)"""
        grid_sizes = self.grid.cell_sizes
        pml_thicknesses = []
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            sizes_axis = grid_sizes.dict()["xyz"[pml_axis]]
            num_layers = pml_layer.num_layers
            pml_thicknesses.append((num_layers * sizes_axis[0], num_layers * sizes_axis[-1]))
        return pml_thicknesses

    @add_ax_if_none
    def plot_pml(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plots each of simulation's PML regions"""
        kwargs = PMLParams().update_params(**kwargs)
        pml_thicks = self.pml_thicknesses
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            if pml_layer.num_layers == 0:
                continue
            for sign, pml_height in zip((-1, 1), pml_thicks[pml_axis]):
                pml_size = [inf, inf, inf]
                pml_size[pml_axis] = pml_height
                pml_center = list(self.center)
                pml_offset_center = (self.size[pml_axis] + pml_height) / 2.0
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
        _, (pml_thick_x, pml_thick_y) = self.pop_axis(self.pml_thicknesses, axis=axis)

        ax.set_xlim(xmin - pml_thick_x[0], xmax + pml_thick_x[1])
        ax.set_ylim(ymin - pml_thick_y[0], ymax + pml_thick_y[1])
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

    @property
    def frequency_range(self) -> FreqBound:
        """range of frequencies spanning all sources' frequency dependence"""
        freq_min = min(source.frequency_range[0] for source in self.sources)
        freq_max = max(source.frequency_range[1] for source in self.sources)
        return (freq_min, freq_max)

    """ Discretization """

    @property
    def dt(self) -> float:
        """compute time step (distance)"""
        dl_mins = [np.min(sizes) for sizes in self.grid.cell_sizes]
        dl_sum_inv_sq = [1 / dl ** 2 for dl in dl_mins]
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        return self.courant * dl_avg / C_0

    @property
    def tmesh(self) -> Coords1D:
        """compute time steps"""
        dt = self.dt
        return np.arange(0.0, self.run_time + dt, dt)

    @property
    def grid(self) -> Grid:
        """:class:`Grid` interface to the spatial locations in Simulation"""
        cell_boundary_dict = {}
        for key, grid_spec, center, size in zip("xyz", self.grid_specs, self.center, self.size):
            if isinstance(grid_spec, float):
                size_snapped = grid_spec * np.floor(size / grid_spec)
                if size_snapped != size:
                    log.warning(f"dl = {grid_spec} not commensurate with simulation size = {size}")
                bound_coords = center + np.linspace(-size_snapped / 2, size_snapped / 2, grid_spec)
            else:
                dl = self.wvl_mat_min / grid_spec.pts_per_wvl
                dl_commensurate = size / np.ceil(size / dl)
                bound_coords = center + np.linspace(-size / 2, size / 2, dl_commensurate)
            cell_boundary_dict[key] = bound_coords
        cell_boundaries = Coords(**cell_boundary_dict)
        return Grid(cell_boundaries=cell_boundaries)

    @property
    def wvl_mat_min(self) -> float:
        """minimum wavelength in the material"""
        freq_max = max(source.source_time.freq0 for source in self.sources)
        wvl_min = C_0 / min(freq_max)
        eps_max = max(abs(structure.medium.get_eps(freq_max)) for structure in self.structures)
        n_max, _ = eps_complex_to_nk(eps_max)
        return wvl_min / n_max

    def discretize(self, box: Box) -> Grid:
        """returns subgrid containing only cells that intersect with Box"""
        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize")
        pts_min, pts_max = box.bounds
        cell_boundaries = self.grid.cell_boundaries
        sub_cell_boundary_dict = {}
        for label, pt_min, pt_max, bound_coords in zip("xyz", pts_min, pts_max, cell_boundaries):
            outer_coords_min = np.where(bound_coords <= pt_min)
            outer_coords_max = np.where(bound_coords >= pt_max)
            ind_min = outer_coords_min[0][0] if outer_coords_min else 0
            ind_max = outer_coords_max[0][-1] if outer_coords_max else len(bound_coords) - 1
            sub_cell_boundary_dict[label] = cell_boundaries[ind_min : ind_max + 1]
        sub_cell_boundaries = Coords(**sub_cell_boundary_dict)
        return Grid(cell_boundaries=sub_cell_boundaries)

    def epsilon(self, box: Box, freq: float = None) -> xr.Dataset:
        """get data of permittivity at volume specified by box and freq"""

        sub_grid = self.discretize(box)
        eps_background = self.medium.eps_model(freq)

        def make_eps_data(coords: Coords):
            """returns epsilon data on grid of points defined by coords"""
            xs, ys, zs = coords.x, coords.y, coords.z
            x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
            eps_array = eps_background * np.ones(x.shape, dtype=complex)
            for structure in self.structures:
                eps_structure = structure.medium.eps_model(freq)
                is_inside = structure.geometry.inside(x, y, z)
                eps_array[np.where(is_inside)] = eps_structure
            return xr.DataArray(eps_array, coords={"x": xs, "y": ys, "z": zs})

        # combine all data into dataset
        data_arrays = {
            "centers": make_eps_data(sub_grid.cell_centers),
            "boundaries": make_eps_data(sub_grid.cell_boundaries),
            "Ex": make_eps_data(sub_grid.yee.E.x),
            "Ey": make_eps_data(sub_grid.yee.E.y),
            "Ez": make_eps_data(sub_grid.yee.E.z),
            "Hx": make_eps_data(sub_grid.yee.H.x),
            "Hy": make_eps_data(sub_grid.yee.H.y),
            "Hz": make_eps_data(sub_grid.yee.H.z),
        }

        return xr.Dataset(data_arrays)
