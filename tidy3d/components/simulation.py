""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List

import pydantic
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

from .types import GridSize, Symmetry, Axis, Ax, Numpy
from .geometry import Box
from .medium import Medium, MediumType
from .structure import Structure
from .source import SourceType
from .monitor import MonitorType
from .pml import PMLLayer
from .viz import StructMediumParams, StructEpsParams, PMLParams, SymParams, add_ax_if_none
from ..constants import inf

# technically this is creating a circular import issue because it calls tidy3d/__init__.py
# from .. import __version__ as version_number


class Simulation(Box):
    """Contains all information about simulation"""

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
        """for each geometry-containing object in simulation, make sure it intersects simulation"""

        for position_index, structure in enumerate(self.structures):
            assert self.intersects(
                structure.geometry
            ), f"Structure '{structure}' (structures[{position_index}]) is outside simulation"

        for geo_obj_dict in (self.sources, self.monitors):
            for name, geo_obj in geo_obj_dict.items():
                assert self.intersects(geo_obj), f"object '{name}' is completely outside simulation"

    """ Accounting """

    @property
    def medium_map(self):
        """medium_map[medium] returns index of unique medium"""
        mediums = {s.medium for s in self.structures}
        return {m: i for i, m in enumerate(mediums)}

    """ Visualization """

    @add_ax_if_none
    def plot(  # pylint: disable=arguments-differ
        self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict
    ) -> Ax:
        """plot each of simulation's components on a plane"""

        ax = self.plot_structures(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_sources(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_monitors(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_symmetries(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_pml(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.set_plot_bounds(axis=axis, ax=ax, **plot_params)
        return ax

    @add_ax_if_none
    def plot_eps(
        self,
        position: float,
        axis: Axis,
        freq: float = None,
        ax: Ax = None,
        **plot_params: dict,
    ) -> Ax:
        """plot the permittivity of each of simulation's components on a plane"""

        ax = self.plot_structures_eps(position=position, axis=axis, freq=freq, ax=ax, **plot_params)
        ax = self.plot_sources(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_monitors(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_symmetries(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.plot_pml(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.set_plot_bounds(axis=axis, ax=ax, **plot_params)
        return ax

    @add_ax_if_none
    def plot_structures(
        self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict
    ) -> Ax:
        """plots all of simulation's structures as materials"""
        medium_map = self.medium_map
        for structure in self.structures:
            if structure.geometry.intersects_plane(position=position, axis=axis):
                params_updater = StructMediumParams(medium=structure.medium, medium_map=medium_map)
                plot_params_new = params_updater.update_params(**plot_params)
                ax = structure.geometry.plot(position=position, axis=axis, ax=ax, **plot_params_new)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    @add_ax_if_none
    def plot_structures_eps(  # pylint: disable=too-many-arguments
        self,
        position: float,
        axis: Axis,
        freq: float = None,
        ax: Ax = None,
        cbar: bool = True,
        **plot_params: dict,
    ) -> Ax:
        """plots all of simulation's structures as permittivity"""
        if freq is None:
            freq = inf
        eps_list = [s.medium.eps_model(freq).real for s in self.structures]
        eps_max = max(eps_list)
        eps_min = min(eps_list)
        for structure in self.structures:
            if structure.geometry.intersects_plane(position=position, axis=axis):
                eps = structure.medium.eps_model(freq).real
                params_updater = StructEpsParams(eps=eps, eps_max=eps_max)
                plot_params_new = params_updater.update_params(**plot_params)
                ax = structure.geometry.plot(position=position, axis=axis, ax=ax, **plot_params_new)
        if cbar:
            norm = mpl.colors.Normalize(vmin=eps_min, vmax=eps_max)
            plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap="gist_yarg"), ax=ax, label=r"$\epsilon_r$"
            )
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    @add_ax_if_none
    def plot_sources(self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict) -> Ax:
        """plots each of simulation's sources on plane"""
        for _, source in self.sources.items():
            if source.intersects_plane(position=position, axis=axis):
                ax = source.plot(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    @add_ax_if_none
    def plot_monitors(self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict) -> Ax:
        """plots each of simulation's monitors on plane"""
        for _, monitor in self.monitors.items():
            if monitor.intersects_plane(position=position, axis=axis):
                ax = monitor.plot(position=position, axis=axis, ax=ax, **plot_params)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    @add_ax_if_none
    def plot_symmetries(
        self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict
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
            if sym_box.intersects_plane(position=position, axis=axis):
                plot_params_new = SymParams(sym_value=sym_value).update_params(**plot_params)
                ax = sym_box.plot(position=position, axis=axis, ax=ax, **plot_params_new)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    @add_ax_if_none
    def plot_pml(self, position: float, axis: Axis, ax: Ax = None, **plot_params: dict) -> Ax:
        """plots each of simulation's PML regions"""
        plot_params_new = PMLParams().update_params(**plot_params)
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            if pml_layer.num_layers == 0:
                continue
            pml_thickness = self.grid_size[pml_axis] * pml_layer.num_layers
            pml_size = [inf, inf, inf]
            pml_size[pml_axis] = pml_thickness
            pml_offset_center = (self.size[pml_axis] + pml_thickness) / 2.0
            for sign in (-1, 1):
                pml_center = list(self.center)
                pml_center[pml_axis] += sign * pml_offset_center
                pml_box = Box(center=pml_center, size=pml_size)
                if pml_box.intersects_plane(position=position, axis=axis):
                    ax = pml_box.plot(position=position, axis=axis, ax=ax, **plot_params_new)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    def set_plot_bounds(self, axis: Axis, ax: Ax) -> Ax:
        """sets the xy limits of the simulation, useful after plotting"""

        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)

        pml_thicknesses = [
            dl * pml.num_layers for (dl, pml) in zip(self.grid_size, self.pml_layers)
        ]

        _, (pml_thick_x, pml_thick_y) = self.pop_axis(pml_thicknesses, axis=axis)

        ax.set_xlim(xmin - pml_thick_x, xmax + pml_thick_x)
        ax.set_ylim(ymin - pml_thick_y, ymax + pml_thick_y)
        return ax

    """ Discretization """

    def _discretize(self, box: Box) -> Numpy:
        """get x,y,z positions of box using self.grid_size"""

        (xmin, ymin, zmin), (xmax, ymax, zmax) = box.get_bounds()
        dlx, dly, dlz = self.grid_size
        x_range = np.arange(xmin, xmax + dlx / 2, dlx)
        y_range = np.arange(ymin, ymax + dly / 2, dly)
        z_range = np.arange(zmin, zmax + dlz / 2, dlz)
        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        xx = (x + dlx / 2.0, y, z)
        yy = (x, y + dly / 2.0, z)
        zz = (x, y, z + dlz / 2.0)
        return np.stack((xx, yy, zz), axis=0)

    def epsilon(self, box: Box, freq: float) -> Numpy:
        """get permittivity at volume specified by box and freq"""
        xyz_pts = self._discretize(box)
        eps_background = self.medium.eps_model(freq)
        eps_array = eps_background * np.ones(xyz_pts.shape[1:], dtype=complex)
        # print(eps_array.shape)
        for structure in self.structures:
            geo = structure.geometry
            if not geo.intersects(box):
                continue
            eps_structure = structure.medium.eps_model(freq)
            for i, pts in enumerate(xyz_pts):
                x, y, z = pts
                structure_map = geo.is_inside(x, y, z)
                eps_array[i, structure_map] = eps_structure
        return eps_array
