# pylint: disable=unused-import
""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List

import pydantic
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from descartes import PolygonPatch

from .types import Symmetry, Ax, Shapely, FreqBound
from .validators import assert_unique_names, assert_objects_in_sim_bounds, set_names
from .geometry import Box, PolySlab, Cylinder, Sphere
from .types import Symmetry, Ax, Shapely, FreqBound, Numpy
from .geometry import Box
from .grid import Coords1D, Grid, Coords
from .medium import Medium, MediumType, eps_complex_to_nk
from .structure import Structure
from .source import SourceType, VolumeSource, GaussianPulse
from .monitor import MonitorType, FieldMonitor, FluxMonitor
from .pml import PMLTypes, PML
from .viz import StructMediumParams, StructEpsParams, PMLParams, SymParams, add_ax_if_none
from ..constants import inf, C_0
from ..log import log, SetupError

# technically this is creating a circular import issue because it calls tidy3d/__init__.py
# from .. import __version__ as version_number


class Simulation(Box):  # pylint:disable=too-many-public-methods
    """Contains all information about Tidy3d simulation.

    Parameters
    ----------
    center : Tuple[float, float, float] = (0.0, 0.0, 0.0)
        (microns) Center of simulation domain in x, y, and z.
    size : Tuple[float, float, float]
        (microns) Size of simulation domain in x, y, and z.
        Each element must be non-negative.
    grid_size : Tuple[float, float, float]
        (microns) Grid size along x, y, and z.
        Each element must be non-negative.
    run_time : float = 0.0
        Total electromagnetic evolution time in seconds.
        Note: If ``shutoff`` specified, simulation will terminate early when shutoff condition met.
        Must be non-negative.
    medium : :class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye` = ``Medium(permittivity=1.0)``
        Background :class:`tidy3d.Medium` of simulation, defaults to air.
    structures : List[:class:`Structure`] = []
        List of structures in simulation.
        Note: Structures defined later in this list override the simulation material properties in
        regions of spatial overlap.
    sources : List[:class:`VolumeSource` or :class:`PlaneWave` or :class:`ModeSource`] = []
        List of electric current sources injecting fields into the simulation.
    monitors : List[:class:`FieldMonitor` or :class:`FieldTimeMonitor` or :class:`FluxMonitor` or :class:`FluxTimeMonitor` or :class:`ModeMonitor`] = []
        List of monitors in the simulation.
        Note: names stored in ``monitor.name`` are used to access data after simulation is run.
    pml_layers : Tuple[:class:`AbsorberSpec`, :class:`AbsorberSpec`, :class:`AbsorberSpec`]
        = ``(None, None, None)``
        Specifications for the absorbing layers on x, y, and z edges.
        Elements of ``None`` are assumed to have no absorber and use periodic boundary conditions.
    symmetry : Tuple[int, int, int] = (0, 0, 0)
        Tuple of integers defining reflection symmetry across a
        plane bisecting the simulation domain normal to the x-, y-, and
        z-axis, respectively. Each element can be ``0`` (no symmetry),
        ``1`` (even, i.e. 'PMC' symmetry) or ``-1`` (odd, i.e. 'PEC'
        symmetry).
        Note that the vectorial nature of the fields must be taken into account to correctly
        determine the symmetry value.
    shutoff : float = 1e-5
        Ratio of the instantaneous integrated E-field intensity to the maximum value
        at which the simulation will automatically shut down.
        Used to prevent extraneous run time of simulations with fully decayed fields.
        Set to ``0`` to disable this feature.
    subpixel : bool = True
        If ``True``, uses subpixel averaging of the permittivity based on structure definition,
        resulting in much higher accuracy for a given grid size.
    courant : float = 0.9
        Courant stability factor, controls time step to spatial step ratio.
        Lower values lead to more stable simulations for dispersive materials,
        but result in longer simulation times.
        Accepted values between 0 and 1, non-inclusive.

    Example
    -------
    >>> sim = Simulation(
    ...     size=(2.0, 2.0, 2.0),
    ...     grid_size=(0.1, 0.1, 0.1),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=1.0, conductivity=3.0),
    ...         ),
    ...         Structure(geometry=Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=Medium()),
    ...         Structure(
    ...             geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
    ...             medium=Medium(),
    ...         ),
    ...         Structure(
    ...             geometry=PolySlab(
    ...                 vertices=[(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5)], slab_bounds=[-1, 1]
    ...             ),
    ...             medium=Medium(permittivity=3.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         VolumeSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             source_time=GaussianPulse(
    ...                 freq0=2e14,
    ...                 fwidth=4e13,
    ...             ),
    ...         )
    ...     ],
    ...     monitors=[
    ...         FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1.5e14, 2e14], name='point'),
    ...         FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name='flux'),
    ...     ],
    ...     symmetry=(0, 0, 0),
    ...     pml_layers=(
    ...         PML(num_layers=20),
    ...         PML(num_layers=30),
    ...         None,
    ...     ),
    ...     shutoff=1e-6,
    ...     courant=0.8,
    ...     subpixel=False,
    ... )
    """

    grid_size: Tuple[pydantic.PositiveFloat, pydantic.PositiveFloat, pydantic.PositiveFloat]
    medium: MediumType = Medium()
    run_time: pydantic.NonNegativeFloat = 0.0
    structures: List[Structure] = []
    sources: List[SourceType] = []
    monitors: List[MonitorType] = []
    pml_layers: Tuple[PMLTypes, PMLTypes, PMLTypes] = (None, None, None)
    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = (0, 0, 0)
    shutoff: pydantic.NonNegativeFloat = 1e-5
    subpixel: bool = True
    courant: pydantic.confloat(gt=0.0, le=1.0) = 0.9

    # TODO: add version to json
    # version: str = str(version_number)

    """ Validating setup """

    @pydantic.validator("pml_layers", always=True)
    def set_none_to_zero_layers(cls, val):
        """if any PML layer is None, set it to an empty :class:`PML`."""
        return tuple(PML(num_layers=0) if pml is None else pml for pml in val)

    _structures_in_bounds = assert_objects_in_sim_bounds("structures")
    _sources_in_bounds = assert_objects_in_sim_bounds("sources")
    _monitors_in_bounds = assert_objects_in_sim_bounds("monitors")

    # assign names to unnamed structures, sources, and mediums
    _structure_names = set_names("structures")
    _source_names = set_names("sources")

    @pydantic.validator("structures", allow_reuse=True, always=True)
    def set_medium_names(cls, val, values):
        """check for intersection of each structure with simulation bounds."""
        background_medium = values.get("medium")
        all_mediums = [background_medium] + [structure.medium for structure in val]
        _, unique_indices = np.unique(all_mediums, return_inverse=True)
        for unique_index, medium in zip(unique_indices, all_mediums):
            if not medium.name:
                medium.name = f"mediums[{unique_index}]"
        return val

    # make sure all names are unique
    _unique_structure_names = assert_unique_names("structures")
    _unique_source_names = assert_unique_names("sources")
    _unique_monitor_names = assert_unique_names("monitors")
    # _unique_medium_names = assert_unique_names("structures", check_mediums=True)

    # TODO:
    # - check sources in medium freq range
    # - check PW in homogeneous medium
    # - check nonuniform grid covers the whole simulation domain

    """ Accounting """

    @property
    def mediums(self) -> List[MediumType]:
        """Returns set of distinct :class:`AbstractMedium` in simulation.

        Returns
        -------
        Set[:class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye`]
            Set of distinct mediums in the simulation.
        """
        return {structure.medium for structure in self.structures}

    @property
    def medium_map(self) -> Dict[MediumType, pydantic.NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`AbstractMedium` in simulation.

        Returns
        -------
        Dict[:class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye`, int]
            Mapping between distinct mediums to index in simulation.
        """

        return {medium: index for index, medium in enumerate(self.mediums)}

    """ Plotting """

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
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
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures_eps(freq=freq, cbar=True, ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_structures(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
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
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @staticmethod
    def _add_cbar(eps_min: float, eps_max: float, ax: Ax = None) -> None:
        """Add a colorbar to eps plot."""
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
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
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
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_sources(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        for source in self.sources:
            if source.intersects_plane(x=x, y=y, z=z):
                ax = source.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_monitors(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's monitors on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        for monitor in self.monitors:
            if monitor.intersects_plane(x=x, y=y, z=z):
                ax = monitor.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_symmetries(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's symmetries on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
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
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @property
    def num_pml_layers(self) -> List[Tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """
        num_layers = []
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            if self.symmetry[pml_axis] != 0:
                num_layers.append((0, pml_layer.num_layers))
            else:
                num_layers.append((pml_layer.num_layers, pml_layer.num_layers))
        return num_layers

    @property
    def pml_thicknesses(self) -> List[Tuple[float, float]]:
        """Thicknesses (um) of absorbers in all three axes and directions (-, +)

        Returns
        -------
        List[Tuple[float, float]]
            List containing the absorber thickness (micron) in - and + boundaries.
        """
        num_layers = self.num_pml_layers
        pml_thicknesses = []
        for boundaries in self.grid.boundaries.dict().values():
            thick_l = boundaries[num_layers[0]] - boundaries[0]
            thick_r = boundaries[-1] - boundaries[-1 - num_layers[1]]
            pml_thicknesses.append((thick_l, thick_r))
        return pml_thicknesses

    @add_ax_if_none
    def plot_pml(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot each of simulation's absorbing boundaries
        on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
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
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    def _set_plot_bounds(self, ax: Ax, x: float = None, y: float = None, z: float = None) -> Ax:
        """Sets the xy limits of the simulation at a plane, useful after plotting.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to set bounds on.
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The axes after setting the boundaries.
        """

        axis, _ = self._parse_xyz_kwargs(x=x, y=y, z=z)
        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)
        _, (pml_thick_x, pml_thick_y) = self.pop_axis(self.pml_thicknesses, axis=axis)

        ax.set_xlim(xmin - pml_thick_x[0], xmax + pml_thick_x[1])
        ax.set_ylim(ymin - pml_thick_y[0], ymax + pml_thick_y[1])
        return ax

    def _filter_plot_structures(
        self, x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.

        Parameters
        ----------
        x : float = None
            Position of point in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            Position of point in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            Position of point in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane after merging.
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
        """Merge list of shapes and mediums on plae by intersection with same medium.
        Edit background shapes to remove volume by intersection.

        Parameters
        ----------
        shapes : List[Tuple[:class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye`, shapely.geometry.base.BaseGeometry]]
            Ordered list of shapes and their mediums on a plane.

        Returns
        -------
        List[Tuple[:class:`Medium` or :class:`PoleResidue` or :class:`Lorentz` or :class:`Sellmeier` or :class:`Debye`, shapely.geometry.base.BaseGeometry]]
            Shapes and their mediums on a plane after merging and removing intersections with background.
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
        """Range of frequencies spanning all sources' frequency dependence.

        Returns
        -------
        Tuple[float, float]
            Minumum and maximum frequencies of the power spectrum of the sources
            at 5 standard deviations.
        """
        freq_min = min(source.frequency_range[0] for source in self.sources)
        freq_max = max(source.frequency_range[1] for source in self.sources)
        return (freq_min, freq_max)

    """ Discretization """

    @property
    def dt(self) -> float:
        """Simulation time step (distance).

        Returns
        -------
        float
            Time step (seconds).
        """
        dl_mins = [np.min(sizes) for sizes in self.grid.sizes.dict().values()]
        dl_sum_inv_sq = sum([1 / dl ** 2 for dl in dl_mins])
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        return self.courant * dl_avg / C_0

    @property
    def tmesh(self) -> Coords1D:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self.run_time + dt, dt)

    @property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`Grid`
            :class:`Grid` storing the spatial locations relevant to the simulation.
        """
        cell_boundary_dict = {}
        zipped_vals = zip("xyz", self.grid_size, self.center, self.size, self.num_pml_layers)
        for key, dl, center, size, num_layers in zipped_vals:
            num_cells = int(np.floor(size / dl))
            # Make sure there's at least one cell
            num_cells = max(num_cells, 1)
            size_snapped = dl * num_cells
            if size_snapped != size:
                log.warning(f"dl = {dl} not commensurate with simulation size = {size}")
            bound_coords = center + np.linspace(-size_snapped / 2, size_snapped / 2, num_cells + 1)
            bound_coords = self._add_pml_to_bounds(num_layers, bound_coords)
            cell_boundary_dict[key] = bound_coords
        boundaries = Coords(**cell_boundary_dict)
        return Grid(boundaries=boundaries)

    @staticmethod
    def _add_pml_to_bounds(num_layers: Tuple[int, int], bounds: Coords1D):
        """Append absorber layers to the beginning and end of the simulation bounds
        along one dimension.

        Parameters
        ----------
        num_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.
        bound_coords : np.ndarray
            coordinates specifying boundaries between cells along one dimension.

        Returns
        -------
        np.ndarray
            New bound coordinates along dimension taking abosrber into account.
        """
        if bounds.size < 2:
            return bounds

        first_step = bounds[1] - bounds[0]
        last_step = bounds[-1] - bounds[-2]
        add_left = bounds[0] - first_step * np.arange(num_layers[0], 0, -1)
        add_right = bounds[-1] + last_step * np.arange(1, num_layers[1] + 1)
        new_bounds = np.concatenate((add_left, bounds, add_right))

        return new_bounds

    @property
    def wvl_mat_min(self) -> float:
        """Minimum wavelength in the material.

        Returns
        -------
        float
            Minimum wavelength in the material (microns).
        """
        freq_max = max(source.source_time.freq0 for source in self.sources)
        wvl_min = C_0 / min(freq_max)
        eps_max = max(abs(structure.medium.get_eps(freq_max)) for structure in self.structures)
        n_max, _ = eps_complex_to_nk(eps_max)
        return wvl_min / n_max

    def discretize(self, box: Box) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """
        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize")

        pts_min, pts_max = box.bounds
        boundaries = self.grid.boundaries

        # stores coords for subgrid
        sub_cell_boundary_dict = {}

        # for each dimension
        for axis_label, pt_min, pt_max in zip("xyz", pts_min, pts_max):
            bound_coords = boundaries.dict()[axis_label]
            assert pt_min <= pt_max, "min point was greater than max point"

            # index of smallest coord greater than than pt_max
            inds_gt_pt_max = np.where(bound_coords > pt_max)[0]
            ind_max = len(bound_coords) - 1 if len(inds_gt_pt_max) == 0 else inds_gt_pt_max[0]

            # index of largest coord less than or equal to pt_min
            inds_leq_pt_min = np.where(bound_coords <= pt_min)[0]
            ind_min = 0 if len(inds_leq_pt_min) == 0 else inds_leq_pt_min[-1]

            # copy orginal bound coords into subgrid coords
            sub_cell_boundary_dict[axis_label] = bound_coords[ind_min : ind_max + 1]

        # construct sub grid
        sub_boundaries = Coords(**sub_cell_boundary_dict)
        return Grid(boundaries=sub_boundaries)

    def epsilon(self, box: Box, freq: float = None) -> Dict[str, xr.DataArray]:
        """Get array of permittivity at volume specified by box and freq

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry specifying where to measure the permittivity.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        Dict[str, xarray.DataArray]
            Mapping of coordinate type to xarray DataArray containing permittivity data.
            keys of dict are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'}``.
            ``'centers'`` contains the permittivity at the yee cell centers.
            `'boundaries'`` contains the permittivity at the corner intersections between yee cells.
            ``'Ex'`` and other field keys contain the permittivity
            at the corresponding field position in the yee lattice.
            For details on xarray datasets, refer to `xarray's Documentaton <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_.
        """

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

        # combine all data into dictionary
        data_arrays = {
            "centers": make_eps_data(sub_grid.centers),
            "boundaries": make_eps_data(sub_grid.boundaries),
            "Ex": make_eps_data(sub_grid.yee.E.x),
            "Ey": make_eps_data(sub_grid.yee.E.y),
            "Ez": make_eps_data(sub_grid.yee.E.z),
            "Hx": make_eps_data(sub_grid.yee.H.x),
            "Hy": make_eps_data(sub_grid.yee.H.y),
            "Hz": make_eps_data(sub_grid.yee.H.z),
        }

        return data_arrays
