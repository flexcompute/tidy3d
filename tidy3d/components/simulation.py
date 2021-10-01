# pylint: disable=invalid-name
""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List

import pydantic
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.cm import Set2 as mat_cmap  # pylint: disable=no-name-in-module

from .types import GridSize, Symmetry, Axis, AxesSubplot
from .geometry import Box
from .medium import Medium, MediumType
from .structure import Structure
from .source import SourceType
from .monitor import MonitorType
from .pml import PMLLayer
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
        # - check sources in medium frequency range
        # - check PW in homogeneous medium
        # - check nonuniform grid covers the whole simulation domain

    """ Post-Init Validation """

    def _check_geo_objs_in_bounds(self):
        """for each geometry-containing object in simulation, make sure it intersects simulation"""

        for i, structure in enumerate(self.structures):
            assert self.intersects(structure.geometry), (
                f"Structure '{structure}' (at position {i}) " "is completely outside simulation"
            )

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

    def plot(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plot each of simulation's components on a plane"""

        ax = self.plot_structures(position=position, axis=axis, ax=ax)
        ax = self.plot_sources(position=position, axis=axis, ax=ax)
        ax = self.plot_monitors(position=position, axis=axis, ax=ax)
        ax = self.plot_symmetries(position=position, axis=axis, ax=ax)
        ax = self.plot_pml(position=position, axis=axis, ax=ax)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    def plot_eps(
        self, position: float, axis: Axis, frequency: float = None, ax: AxesSubplot = None
    ) -> AxesSubplot:
        """plot the permittivity of each of simulation's components on a plane"""

        ax = self.plot_structures_eps(position=position, axis=axis, frequency=frequency, ax=ax)
        ax = self.plot_sources(position=position, axis=axis, ax=ax)
        ax = self.plot_monitors(position=position, axis=axis, ax=ax)
        ax = self.plot_symmetries(position=position, axis=axis, ax=ax)
        ax = self.plot_pml(position=position, axis=axis, ax=ax)
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    def plot_structures(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots all of simulation's structures as materials"""
        medium_map = self.medium_map
        for structure in self.structures:
            if not structure.geometry.intersects_plane(position=position, axis=axis):
                continue
            mat_index = medium_map[structure.medium]
            facecolor = mat_cmap(mat_index % len(mat_cmap.colors))
            ax = structure.geometry.plot(position=position, axis=axis, facecolor=facecolor, ax=ax)
        return ax

    def plot_structures_eps(
        self, position: float, axis: Axis, frequency: float = None, ax: AxesSubplot = None
    ) -> AxesSubplot:
        """plots all of simulation's structures as permittivity"""
        max_eps = max([s.medium.eps_model(frequency).real for s in self.structures])
        max_chi = max_eps - 1.0
        for structure in self.structures:
            if structure.geometry.intersects_plane(position=position, axis=axis):
                eps = structure.medium.eps_model(frequency).real
                chi = eps - 1.0
                facecolor = str(1 - chi / max_chi)
                ax = structure.geometry.plot(
                    position=position, axis=axis, facecolor=facecolor, ax=ax
                )
        norm = mpl.colors.Normalize(vmin=1, vmax=1 + max_chi)
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="gist_yarg"), ax=ax, label=r"$\epsilon_r$"
        )
        return ax

    def plot_sources(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots each of simulation's sources on plane"""
        for _, source in self.sources.items():
            if source.intersects_plane(position=position, axis=axis):
                ax = source.geometry.plot(
                    position=position,
                    axis=axis,
                    alpha=0.7,
                    facecolor="blueviolet",
                    edgecolor="blueviolet",
                    ax=ax,
                )
        return ax

    def plot_monitors(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots each of simulation's monitors on plane"""
        for _, monitor in self.monitors.items():
            if monitor.intersects_plane(position=position, axis=axis):
                ax = monitor.geometry.plot(
                    position=position,
                    axis=axis,
                    alpha=0.5,
                    facecolor="crimson",
                    edgecolor="crimson",
                    ax=ax,
                )
        return ax

    def plot_symmetries(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots each of the non-zero symmetries"""
        for sym_axis, sym_value in enumerate(self.symmetry):
            if sym_value == 0:
                continue
            color = "lightgreen" if sym_value == 1 else "lightsteelblue"
            sym_size = [inf, inf, inf]
            sym_size[sym_axis] = inf / 2
            sym_center = list(self.center)
            sym_center[sym_axis] += sym_size[sym_axis] / 2
            sym_box = Box(center=sym_center, size=sym_size)
            if sym_box.intersects_plane(position=position, axis=axis):
                ax = sym_box.plot(
                    position=position,
                    axis=axis,
                    ax=ax,
                    facecolor=color,
                    alpha=0.5,
                    edgecolor=color,
                )
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    def plot_pml(self, position: float, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots each of simulation's PML regions"""
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
                    ax = pml_box.plot(
                        position=position,
                        axis=axis,
                        ax=ax,
                        facecolor="sandybrown",
                        alpha=0.7,
                        edgecolor="sandybrown",
                    )
        ax = self.set_plot_bounds(axis=axis, ax=ax)
        return ax

    def set_plot_bounds(self, axis: Axis, ax: AxesSubplot = None) -> AxesSubplot:
        """plots the boundaries of the simulation and sets xy lims"""

        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)

        pml_thicknesses = [
            dl * pml.num_layers for (dl, pml) in zip(self.grid_size, self.pml_layers)
        ]

        _, (pml_thick_x, pml_thick_y) = self._pop_axis(pml_thicknesses, axis=axis)

        ax.set_xlim(xmin - pml_thick_x, xmax + pml_thick_x)
        ax.set_ylim(ymin - pml_thick_y, ymax + pml_thick_y)
        return ax

    """ Discretization """

    def _discretize(self, box: Box):  # pylint: disable=too-many-locals
        """get x,y,z positions of box using self.grid_size"""

        (xmin, ymin, zmin), (xmax, ymax, zmax) = box.get_bounds()
        dlx, dly, dlz = self.grid_size
        x_range = np.arange(xmin, xmax + dlx / 2, dlx)
        y_range = np.arange(ymin, ymax + dly / 2, dly)
        z_range = np.arange(zmin, zmax + dlz / 2, dlz)
        x_pts, y_pts, z_pts = np.meshgrid(x_range, y_range, z_range, indexing="ij")

        return x_pts, y_pts, z_pts

    def epsilon(self, box: Box, freq: float):
        """get permittivity at volume specified by box and frequency"""
        x_pts, y_pts, z_pts = self._discretize(box)
        eps_background = self.medium.eps_model(freq)
        eps_array = eps_background * np.ones(x_pts.shape, dtype=complex)
        for structure in self.structures:
            geo = structure.geometry
            if not geo.intersects(box):
                continue
            eps_structure = structure.medium.eps_model(freq)
            # structure_box = geo._get_bounding_box()
            # _x, _y, _z = self._discretize(structure_box)
            structure_map = geo.is_inside(x_pts, y_pts, z_pts)
            eps_array[structure_map] = eps_structure
        return eps_array
