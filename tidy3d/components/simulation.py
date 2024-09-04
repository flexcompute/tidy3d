"""Container holding all information about simulation and its components"""

from __future__ import annotations

import math
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

import autograd.numpy as np
import matplotlib as mpl
import pydantic.v1 as pydantic
import xarray as xr

from ..constants import C_0, SECOND, fp_eps, inf
from ..exceptions import SetupError, Tidy3dError, Tidy3dImportError, ValidationError
from ..log import log
from ..updater import Updater
from .base import cached_property, skip_if_fields_missing
from .base_sim.simulation import AbstractSimulation
from .boundary import (
    PML,
    Absorber,
    AbsorberSpec,
    BlochBoundary,
    Boundary,
    BoundarySpec,
    PECBoundary,
    Periodic,
    PMCBoundary,
    StablePML,
)
from .data.data_array import FreqDataArray
from .data.dataset import CustomSpatialDataType, Dataset
from .geometry.base import Box, Geometry
from .geometry.mesh import TriangleMesh
from .geometry.utils import flatten_groups, traverse_geometries
from .geometry.utils_2d import get_bounds, get_thickened_geom, snap_coordinate_to_grid, subdivide
from .grid.grid import Coords, Coords1D, Grid
from .grid.grid_spec import AutoGrid, GridSpec, UniformGrid
from .lumped_element import LumpedElementType
from .medium import (
    AbstractCustomMedium,
    AbstractMedium,
    AbstractPerturbationMedium,
    AnisotropicMedium,
    FullyAnisotropicMedium,
    Medium,
    Medium2D,
    MediumType,
    MediumType3D,
)
from .monitor import (
    AbstractFieldProjectionMonitor,
    AbstractModeMonitor,
    DiffractionMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldTimeMonitor,
    FreqMonitor,
    ModeMonitor,
    Monitor,
    MonitorType,
    PermittivityMonitor,
    SurfaceIntegrationMonitor,
    TimeMonitor,
)
from .run_time_spec import RunTimeSpec
from .scene import MAX_NUM_MEDIUMS, Scene
from .source import (
    TFSF,
    AstigmaticGaussianBeam,
    ContinuousWave,
    CustomCurrentSource,
    CustomFieldSource,
    CustomSourceTime,
    GaussianBeam,
    ModeSource,
    PlaneWave,
    Source,
    SourceType,
)
from .structure import MeshOverrideStructure, Structure
from .subpixel_spec import SubpixelSpec
from .types import TYPE_TAG_STR, Ax, Axis, FreqBound, InterpMethod, Literal, Symmetry, annotate_type
from .validators import assert_objects_in_sim_bounds, validate_mode_objects_symmetry
from .viz import (
    PlotParams,
    add_ax_if_none,
    equal_aspect,
    plot_params_bloch,
    plot_params_override_structures,
    plot_params_pec,
    plot_params_pmc,
    plot_params_pml,
    plot_sim_3d,
)

try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False

try:
    gdspy_available = True
    import gdspy
except ImportError:
    gdspy_available = False

# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of sources
MAX_NUM_SOURCES = 1000

# restrictions on simulation number of cells and number of time steps
MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16

# monitor warnings and restrictions
MAX_TIME_MONITOR_STEPS = 5000  # does not apply to 0D monitors
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5

# number of grid cells at which we warn about slow Simulation.epsilon()
NUM_CELLS_WARN_EPSILON = 100_000_000
# number of structures at which we warn about slow Simulation.epsilon()
NUM_STRUCTURES_WARN_EPSILON = 10_000

# height of the PML plotting boxes along any dimensions where sim.size[dim] == 0
PML_HEIGHT_FOR_0_DIMS = inf


class AbstractYeeGridSimulation(AbstractSimulation, ABC):
    """
    Abstract class for a simulation involving electromagnetic fields defined on a Yee grid.
    """

    lumped_elements: Tuple[LumpedElementType, ...] = pydantic.Field(
        (),
        title="Lumped Elements",
        description="Tuple of lumped elements in the simulation. "
        "Note: only :class:`tidy3d.LumpedResistor` is supported currently.",
    )
    """
    Tuple of lumped elements in the simulation.
    """

    grid_spec: GridSpec = pydantic.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )
    """
    Specifications for the simulation grid along each of the three directions.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             grid_spec=GridSpec(
                grid_x = AutoGrid(min_steps_per_wvl = 20),
                grid_y = AutoGrid(min_steps_per_wvl = 20),
                grid_z = AutoGrid(min_steps_per_wvl = 20)
            ),
            ...
         )

    See Also
    --------

    :class:`GridSpec`
        Collective grid specification for all three dimensions.

    :class:`UniformGrid`
        Uniform 1D grid.

    :class:`AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
    """

    subpixel: Union[bool, SubpixelSpec] = pydantic.Field(
        SubpixelSpec(),
        title="Subpixel Averaging",
        description="Apply subpixel averaging methods of the permittivity on structure interfaces "
        "to result in much higher accuracy for a given grid size. Supply a :class:`SubpixelSpec` "
        "to this field to select subpixel averaging methods separately on dielectric, metal, and "
        "PEC material interfaces. Alternatively, user may supply a boolean value: "
        "``True`` to apply the default subpixel averaging methods corresponding to ``SubpixelSpec()`` "
        ", or ``False`` to apply staircasing.",
    )

    simulation_type: Literal["autograd_fwd", "autograd_bwd", None] = pydantic.Field(
        None,
        title="Simulation Type",
        description="Tag used internally to distinguish types of simulations for "
        "``autograd`` gradient processing.",
    )

    post_norm: Union[float, FreqDataArray] = pydantic.Field(
        1.0,
        title="Post Normalization Values",
        description="Factor to multiply the fields by after running, "
        "given the adjoint source pipeline used. Note: this is used internally only.",
    )

    """
    Supply :class:`SubpixelSpec` to select subpixel averaging methods separately for dielectric, metal, and
    PEC material interfaces. Alternatively, supply ``True`` to use default subpixel averaging methods,
    or ``False`` to staircase all structure interfaces.

    **1D Illustration**

    For example, in the image below, two silicon slabs with thicknesses 150nm and 175nm centered in a grid with
    spatial discretization :math:`\\Delta z = 25\\text{nm}` compute the effective permittivity of each grid point as the
    average permittivity between the grid points. A simplified equation based on the ratio :math:`\\eta` between the
    permittivity of the two materials at the interface in this case:

    .. math::

        \\epsilon_{eff} = \\eta \\epsilon_{si} + (1 - \\eta) \\epsilon_{air}

    .. TODO check the actual implementation to be accurate here.

    .. image:: ../../_static/img/subpixel_permittivity_1d.png

    However, in this 1D case, this averaging is accurate because the dominant electric field is parallel to the
    dielectric grid points.

    You can learn more about the subpixel averaging derivation from Maxwell's equations in 1D in this lecture:
    `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel
    -averaging/>`_.

    **2D & 3D Usage Caveats**

    *   In 2D, the subpixel averaging implementation depends on the polarization (:math:`s` or :math:`p`)  of the
        incident electric field on the interface.

    *   In 3D, the subpixel averaging is implemented with tensorial averaging due to arbitrary surface and field
        spatial orientations.


    See Also
    --------

    **Lectures:**
        *  `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel-averaging/>`_
        *  `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_
    """

    @pydantic.validator("lumped_elements", always=True)
    @skip_if_fields_missing(["structures"])
    def _validate_num_lumped_elements(cls, val, values):
        """Error if too many lumped elements present."""

        if val is None:
            return val
        structures = values.get("structures")
        mediums = {structure.medium for structure in structures}
        total_num_mediums = len(val) + len(mediums)
        if total_num_mediums > MAX_NUM_MEDIUMS:
            raise ValidationError(
                f"Tidy3D only supports {MAX_NUM_MEDIUMS} distinct lumped elements and structures."
                f"{total_num_mediums} were supplied."
            )

        return val

    @pydantic.validator("lumped_elements")
    @skip_if_fields_missing(["size"])
    def _check_3d_simulation_with_lumped_elements(cls, val, values):
        """Error if Simulation contained lumped elements and is not a 3D simulation"""
        size = values.get("size")
        if val and size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be a 3D simulation when a 'LumpedElement' is present."
            )
        return val

    @pydantic.validator("grid_spec", always=True)
    @abstractmethod
    def _validate_auto_grid_wavelength(cls, val, values):
        """Check that wavelength can be defined if there is auto grid spec."""
        pass

    def _monitor_num_cells(self, monitor: Monitor) -> int:
        """Total number of cells included in monitor based on simulation grid."""

        def num_cells_in_monitor(monitor: Monitor) -> int:
            """Get the number of measurement cells in a monitor given the simulation grid and
            downsampling."""
            if not self.intersects(monitor):
                # Monitor is outside of simulation domain; can happen e.g. for integration surfaces
                return 0
            num_cells = self.discretize_monitor(monitor).num_cells
            # take monitor downsampling into account
            num_cells = monitor.downsampled_num_cells(num_cells)
            return np.prod(np.array(num_cells, dtype=np.int64))

        if isinstance(monitor, SurfaceIntegrationMonitor):
            return sum(num_cells_in_monitor(mnt) for mnt in monitor.integration_surfaces)
        return num_cells_in_monitor(monitor)

    @cached_property
    def _subpixel(self) -> SubpixelSpec:
        """Subpixel averaging method evaluated based on self.subpixel."""
        if isinstance(self.subpixel, SubpixelSpec):
            return self.subpixel

        # self.subpixel is boolean
        # 1) if it's true, use the default dielectric=True, metal=Staircasing, PEC=Benkler
        if self.subpixel:
            return SubpixelSpec()
        # 2) if it's false, apply staircasing on all material boundaries
        return SubpixelSpec.staircasing()

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        lumped_element_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
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
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        lumped_element_alpha : float = None
            Opacity of the lumped elements. If ``None``, uses Tidy3d default.
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

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Materials <../../notebooks/VizSimulation.html#Plotting-Materials>`_

        """
        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_lumped_elements(
            ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        lumped_element_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
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
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        lumped_element_alpha : float = None
            Opacity of the lumped elements. If ``None``, uses Tidy3d default.
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

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        ax = self.plot_structures_eps(
            freq=freq,
            cbar=True,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
        )
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_lumped_elements(
            ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=lumped_element_alpha
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
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
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
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

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return self.scene.plot_structures_eps(
            freq=freq,
            cbar=cbar,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            grid=self.grid,
            reverse=reverse,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_pml(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's absorbing boundaries
        on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        pml_boxes = self._make_pml_boxes(normal_axis=normal_axis)
        for pml_box in pml_boxes:
            pml_box.plot(x=x, y=y, z=z, ax=ax, **plot_params_pml.to_kwargs())
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        return ax

    # candidate for removal in 3.0
    @cached_property
    def bounds_pml(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        log.warning(
            "'Simulation.bounds_pml' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.simulation_bounds' instead."
        )
        return self.simulation_bounds

    @cached_property
    def simulation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        pml_thick = self.pml_thicknesses
        bounds_in = self.bounds
        bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
        bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

        return (bounds_min, bounds_max)

    def _make_pml_boxes(self, normal_axis: Axis) -> List[Box]:
        """make a list of Box objects representing the pml to plot on plane."""
        pml_boxes = []
        pml_thicks = self.pml_thicknesses
        for pml_axis, num_layers_dim in enumerate(self.num_pml_layers):
            if pml_axis == normal_axis:
                continue
            for sign, pml_height, num_layers in zip((-1, 1), pml_thicks[pml_axis], num_layers_dim):
                if num_layers == 0:
                    continue
                pml_box = self._make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
                pml_boxes.append(pml_box)
        return pml_boxes

    def _make_pml_box(self, pml_axis: Axis, pml_height: float, sign: int) -> Box:
        """Construct a :class:`.Box` representing an arborbing boundary to be plotted."""
        rmin, rmax = (list(bounds) for bounds in self.simulation_bounds)
        if sign == -1:
            rmax[pml_axis] = rmin[pml_axis] + pml_height
        else:
            rmin[pml_axis] = rmax[pml_axis] - pml_height
        pml_box = Box.from_bounds(rmin=rmin, rmax=rmax)

        # if any dimension of the sim has size 0, set the PML to a very small size along that dim
        new_size = list(pml_box.size)
        for dim_index, sim_size in enumerate(self.size):
            if sim_size == 0.0:
                new_size[dim_index] = PML_HEIGHT_FOR_0_DIMS
        pml_box = pml_box.updated_copy(size=new_size)

        return pml_box

    # candidate for removal in 3.0
    def eps_bounds(self, freq: float = None) -> Tuple[float, float]:
        """Compute range of (real) permittivity present in the simulation at frequency "freq"."""

        log.warning(
            "'Simulation.eps_bounds()' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.eps_bounds()' instead."
        )
        return self.scene.eps_bounds(freq=freq)

    @cached_property
    def pml_thicknesses(self) -> List[Tuple[float, float]]:
        """Thicknesses (um) of absorbers in all three axes and directions (-, +)

        Returns
        -------
        List[Tuple[float, float]]
            List containing the absorber thickness (micron) in - and + boundaries.
        """
        num_layers = self.num_pml_layers
        pml_thicknesses = []
        for num_layer, boundaries in zip(num_layers, self.grid.boundaries.to_list):
            thick_l = boundaries[num_layer[0]] - boundaries[0]
            thick_r = boundaries[-1] - boundaries[-1 - num_layer[1]]
            pml_thicknesses.append((thick_l, thick_r))

        return pml_thicknesses

    @equal_aspect
    @add_ax_if_none
    def plot_lumped_elements(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's lumped elements on a plane defined by one
        nonzero x,y,z coordinate.

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
            Opacity of the lumped element, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.bounds
        for element in self.lumped_elements:
            kwargs = element.plot_params.include_kwargs(alpha=alpha).to_kwargs()
            ax = element.to_structure.plot(x=x, y=y, z=z, ax=ax, sim_bounds=bounds, **kwargs)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        return ax

    @add_ax_if_none
    def plot_grid(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

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
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        cell_boundaries = self.grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = cell_boundaries.dict()["xyz"[axis_x]]
        boundaries_y = cell_boundaries.dict()["xyz"[axis_y]]
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        # Plot bounding boxes of override structures
        plot_params = plot_params_override_structures.include_kwargs(
            linewidth=2 * kwargs["linewidth"], edgecolor=kwargs["colors"]
        )
        for structure in self.grid_spec.override_structures:
            bounds = list(zip(*structure.geometry.bounds))
            _, ((xmin, xmax), (ymin, ymax)) = structure.geometry.pop_axis(bounds, axis=axis)
            xmin, xmax, ymin, ymax = (self._evaluate_inf(v) for v in (xmin, xmax, ymin, ymax))
            rect = mpl.patches.Rectangle(
                xy=(xmin, ymin),
                width=(xmax - xmin),
                height=(ymax - ymin),
                **plot_params.to_kwargs(),
            )
            ax.add_patch(rect)

        # Plot snapping points
        for point in self.grid_spec.snapping_points:
            _, (x_point, y_point) = Geometry.pop_axis(point, axis=axis)
            x_point, y_point = (self._evaluate_inf(v) for v in (x_point, y_point))
            ax.scatter(x_point, y_point, color=plot_params.edgecolor)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the simulation boundary conditions as lines on a plane
           defined by one nonzero x,y,z coordinate.

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
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        def set_plot_params(boundary_edge, lim, side, thickness):
            """Return the line plot properties such as color and opacity based on the boundary"""
            if isinstance(boundary_edge, PECBoundary):
                plot_params = plot_params_pec.copy(deep=True)
            elif isinstance(boundary_edge, PMCBoundary):
                plot_params = plot_params_pmc.copy(deep=True)
            elif isinstance(boundary_edge, BlochBoundary):
                plot_params = plot_params_bloch.copy(deep=True)
            else:
                plot_params = PlotParams(alpha=0)

            # expand axis limit so that the axis ticks and labels aren't covered
            new_lim = lim
            if plot_params.alpha != 0:
                if side == -1:
                    new_lim = lim - thickness
                elif side == 1:
                    new_lim = lim + thickness

            return plot_params, new_lim

        boundaries = self.boundary_spec.to_list

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dim_u, dim_v) = self.pop_axis([0, 1, 2], axis=normal_axis)

        umin, umax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()

        size_factor = 1.0 / 35.0
        thickness_u = (umax - umin) * size_factor
        thickness_v = (vmax - vmin) * size_factor

        # boundary along the u axis, minus side
        plot_params, ulim_minus = set_plot_params(boundaries[dim_u][0], umin, -1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umin - thickness_u, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the u axis, plus side
        plot_params, ulim_plus = set_plot_params(boundaries[dim_u][1], umax, 1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umax, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, minus side
        plot_params, vlim_minus = set_plot_params(boundaries[dim_v][0], vmin, -1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmin - thickness_v),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, plus side
        plot_params, vlim_plus = set_plot_params(boundaries[dim_v][1], vmax, 1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmax),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax.set_xlim([ulim_minus, ulim_plus])
        ax.set_ylim([vlim_minus, vlim_plus])

        return ax

    # TODO: not yet supported
    # def plot_3d(self, width=800, height=800) -> None:
    #    """Render 3D plot of ``Simulation`` (in jupyter notebook only).
    #    Parameters
    #    ----------
    #    width : float = 800
    #        width of the 3d view dom's size
    #    height : float = 800
    #        height of the 3d view dom's size
    #
    #    """
    #    return plot_sim_3d(self, width=width, height=height)

    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.structures

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # This would AutoGrid the in-plane directions of the 2D materials
        # return self._grid_corrections_2dmaterials(grid)
        return grid

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

    def _subgrid(self, span_inds: np.ndarray, grid: Grid = None):
        """Take a subgrid of the simulation grid with cell span defined by ``span_inds`` along the
        three dimensions. Optionally, a grid different from the simulation grid can be provided.
        The ``span_inds`` can also extend beyond the grid, in which case the grid is padded based
        on the boundary conditions of the simulation along the different dimensions."""

        if not grid:
            grid = self.grid

        boundary_dict = {}
        for idim, (dim, periodic) in enumerate(zip("xyz", self._periodic)):
            ind_beg, ind_end = span_inds[idim]
            # ind_end + 1 because we are selecting cell boundaries not cells
            boundary_dict[dim] = grid.extended_subspace(idim, ind_beg, ind_end + 1, periodic)
        return Grid(boundaries=Coords(**boundary_dict))

    @cached_property
    def _periodic(self) -> Tuple[bool, bool, bool]:
        """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
        We check on both sides but in practice there should be no cases in which a periodic/Bloch
        BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
        periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
        boundary with another PEC/PMC plane upon initialization."""
        periodic = []
        for bcs_1d in self.boundary_spec.to_list:
            periodic.append(all(isinstance(bcs, (Periodic, BlochBoundary)) for bcs in bcs_1d))
        return periodic

    @cached_property
    def num_pml_layers(self) -> List[Tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """
        num_layers = [[0, 0], [0, 0], [0, 0]]

        for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
            for idx_j, boundary in enumerate(boundary1d):
                if isinstance(boundary, (PML, StablePML, Absorber)):
                    num_layers[idx_i][idx_j] = boundary.num_layers

        return num_layers

    def _snap_zero_dim(self, grid: Grid):
        """Snap a grid to the simulation center along any dimension along which simulation is
        effectively 0D, defined as having a single pixel. This is more general than just checking
        size = 0."""
        size_snapped = [
            size if num_cells > 1 else 0 for num_cells, size in zip(self.grid.num_cells, self.size)
        ]
        return grid.snap_to_box_zero_dim(Box(center=self.center, size=size_snapped))

    def _discretize_grid(self, box: Box, grid: Grid, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        As opposed to ``Simulation.discretize``, this function operates on a ``grid``
        which may not be the grid of the simulation.
        """

        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize.")

        span_inds = grid.discretize_inds(box=box, extend=extend)
        return self._subgrid(span_inds=span_inds, grid=grid)

    def _discretize_inds_monitor(self, monitor: Monitor):
        """Start and stopping indexes for the cells where data needs to be recorded to fully cover
        a ``monitor``. This is used during the solver run. The final grid on which a monitor data
        lives is computed in ``discretize_monitor``, with the difference being that 0-sized
        dimensions of the monitor or the simulation are snapped in post-processing."""

        # Expand monitor size slightly to break numerical precision in favor of always having
        # enough data to span the full monitor.
        expand_size = [size + fp_eps if size > fp_eps else size for size in monitor.size]
        box_expanded = Box(center=monitor.center, size=expand_size)
        # Discretize without extension for now
        span_inds = np.array(self.grid.discretize_inds(box_expanded, extend=False))

        if any(ind[0] >= ind[1] for ind in span_inds):
            # At least one dimension has no indexes inside the grid, e.g. monitor is entirely
            # outside of the grid
            return span_inds

        # Now add extensions, which are specific for monitors and are determined such that data
        # colocated to grid boundaries can be interpolated anywhere inside the monitor.
        # We always need to expand on the right.
        span_inds[:, 1] += 1
        # Non-colocating monitors also need to expand on the left.
        if not monitor.colocate:
            span_inds[:, 0] -= 1
        return span_inds

    def discretize_monitor(self, monitor: Monitor) -> Grid:
        """Grid on which monitor data corresponding to a given monitor will be computed."""
        span_inds = self._discretize_inds_monitor(monitor)
        grid_snapped = self._subgrid(span_inds=span_inds).snap_to_box_zero_dim(monitor)
        grid_snapped = self._snap_zero_dim(grid=grid_snapped)
        return grid_snapped

    def discretize(self, box: Box, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`.Box`.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry within simulation to discretize.
        extend : bool = False
            If ``True``, ensure that the returned indexes extend sufficiently in every direction to
            be able to interpolate any field component at any point within the ``box``, for field
            components sampled on the Yee grid.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """
        return self._discretize_grid(box=box, grid=self.grid, extend=extend)

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float = None,
    ) -> xr.DataArray:
        """Get array of permittivity at volume specified by box and freq.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.

        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.

        See Also
        --------

        **Notebooks**
            * `First walkthrough: permittivity data <../../notebooks/Simulation.html#Permittivity-data>`_
        """

        sub_grid = self.discretize(box)
        return self.epsilon_on_grid(grid=sub_grid, coord_key=coord_key, freq=freq)

    def epsilon_on_grid(
        self,
        grid: Grid,
        coord_key: str = "centers",
        freq: float = None,
    ) -> xr.DataArray:
        """Get array of permittivity at a given freq on a given grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Grid specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.
        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.
        """

        grid_cells = np.prod(grid.num_cells)
        num_structures = len(self.structures)
        if grid_cells > NUM_CELLS_WARN_EPSILON:
            log.warning(
                f"Requested grid contains {int(grid_cells):.2e} grid cells. "
                "Epsilon calculation may be slow."
            )
        if num_structures > NUM_STRUCTURES_WARN_EPSILON:
            log.warning(
                f"Simulation contains {num_structures:.2e} structures. "
                "Epsilon calculation may be slow."
            )

        def get_eps(structure: Structure, frequency: float, coords: Coords):
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] != "E":
                return np.mean(structure.eps_diagonal(frequency, coords), axis=0)
            row = ["x", "y", "z"].index(coord_key[1])
            if len(coord_key) == 2:  # diagonal component in case of Ex, Ey, and Ez
                col = row
            else:  # off-diagonal component in case of Exy, Exz, Eyx, etc
                col = ["x", "y", "z"].index(coord_key[2])
            return structure.eps_comp(row, col, frequency, coords)

        def make_eps_data(coords: Coords):
            """returns epsilon data on grid of points defined by coords"""
            arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
            eps_background = get_eps(
                structure=self.scene.background_structure, frequency=freq, coords=coords
            )
            shape = tuple(len(array) for array in arrays)
            eps_array = eps_background * np.ones(shape, dtype=complex)
            # replace 2d materials with volumetric equivalents
            with log as consolidated_logger:
                for structure in self.volumetric_structures:
                    # Indexing subset within the bounds of the structure

                    inds = structure.geometry._inds_inside_bounds(*arrays)

                    # Get permittivity on meshgrid over the reduced coordinates
                    coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                    if any(coords.size == 0 for coords in coords_reduced):
                        continue

                    red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                    eps_structure = get_eps(structure=structure, frequency=freq, coords=red_coords)

                    if structure.medium.nonlinear_spec is not None:
                        consolidated_logger.warning(
                            "Evaluating permittivity of a nonlinear "
                            "medium ignores the nonlinearity."
                        )

                    if isinstance(structure.geometry, TriangleMesh):
                        consolidated_logger.warning(
                            "Client-side permittivity of a 'TriangleMesh' may be "
                            "inaccurate if the mesh is not unionized. We recommend unionizing "
                            "all meshes before import. A 'PermittivityMonitor' can be used to "
                            "obtain the true permittivity and check that the surface mesh is "
                            "loaded correctly."
                        )

                    # Update permittivity array at selected indexes within the geometry
                    is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
                    eps_array[inds][is_inside] = (eps_structure * is_inside)[is_inside]

            coords = dict(zip("xyz", arrays))
            return xr.DataArray(eps_array, coords=coords, dims=("x", "y", "z"))

        # combine all data into dictionary
        if coord_key[0] == "E":
            # off-diagonal components are sampled at respective locations (eg. `eps_xy` at `Ex`)
            coords = grid[coord_key[0:2]]
        else:
            coords = grid[coord_key]
        return make_eps_data(coords)

    def _volumetric_structures_grid(self, grid: Grid) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents, using ``grid`` as the simulation grid."""

        if (
            not any(isinstance(medium, Medium2D) for medium in self.scene.mediums)
            and not self.lumped_elements
        ):
            return self.structures

        def get_dls(geom: Geometry, axis: Axis, num_dls: int) -> List[float]:
            """Get grid size around the 2D material."""
            dls = self._discretize_grid(Box.from_bounds(*geom.bounds), grid=grid).sizes.to_list[
                axis
            ]
            # When 1 dl is requested it is assumed that only an approximate value is needed
            # before the 2D material has been snapped to the grid
            if num_dls == 1:
                return [np.mean(dls)]

            # When 2 dls are requested the 2D geometry should have been snapped to grid,
            # so this represents the exact adjacent grid spacing
            if len(dls) != num_dls:
                raise Tidy3dError(
                    "Failed to detect grid size around the 2D material. "
                    "Can't generate volumetric equivalent for this simulation. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )
            return dls

        def snap_to_grid(geom: Geometry, axis: Axis) -> Geometry:
            """Snap a 2D material to the Yee grid."""
            center = get_bounds(geom, axis)[0]
            if get_bounds(geom, axis)[0] != get_bounds(geom, axis)[1]:
                raise AssertionError(
                    "Unexpected error encountered while processing 2D material. "
                    "The upper and lower bounds of the geometry in the normal direction are not equal. "
                    "If you encounter this error, please create an issue in the Tidy3D github repository."
                )
            snapped_center = snap_coordinate_to_grid(self.grid, center, axis)
            return geom._update_from_bounds(bounds=(snapped_center, snapped_center), axis=axis)

        # Convert lumped elements into structures
        lumped_structures = []
        for lumped_element in self.lumped_elements:
            lumped_structures.append(lumped_element.to_structure)

        # Begin volumetric structures grid
        all_structures = list(self.static_structures) + lumped_structures

        # For 1D and 2D simulations, a nonzero size is needed for the polygon operations in subdivide
        placeholder_size = tuple(i if i > 0 else inf for i in self.geometry.size)
        simulation_placeholder_geometry = self.geometry.updated_copy(
            center=self.geometry.center, size=placeholder_size
        )

        simulation_background = Structure(
            geometry=simulation_placeholder_geometry, medium=self.medium
        )
        background_structures = [simulation_background]
        new_structures = []
        for structure in all_structures:
            if not isinstance(structure.medium, Medium2D):
                # found a 3D material; keep it
                background_structures.append(structure)
                new_structures.append(structure)
                continue
            # otherwise, found a 2D material; replace it with volumetric equivalent
            axis = structure.geometry._normal_2dmaterial
            geometry = structure.geometry

            # subdivide
            avg_axis_dl = get_dls(geometry, axis, 1)[0]
            subdivided_geometries = subdivide(geometry, axis, avg_axis_dl, background_structures)
            # Create and add volumetric equivalents
            for subdivided_geometry in subdivided_geometries:
                # Snap to the grid and create volumetric equivalent
                snapped_geometry = snap_to_grid(subdivided_geometry[0], axis)
                snapped_center = get_bounds(snapped_geometry, axis)[0]
                dls = get_dls(get_thickened_geom(snapped_geometry, axis, avg_axis_dl), axis, 2)
                adjacent_media = [subdivided_geometry[1].medium, subdivided_geometry[2].medium]

                # Create the new volumetric medium
                new_medium = structure.medium.volumetric_equivalent(
                    axis=axis, adjacent_media=adjacent_media, adjacent_dls=dls
                )

                new_bounds = (snapped_center, snapped_center)
                new_geometry = snapped_geometry._update_from_bounds(bounds=new_bounds, axis=axis)
                new_structure = structure.updated_copy(geometry=new_geometry, medium=new_medium)

                new_structures.append(new_structure)

        return tuple(new_structures)

    @cached_property
    def volumetric_structures(self) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents."""
        return self._volumetric_structures_grid(self.grid)

    def suggest_mesh_overrides(self, **kwargs) -> List[MeshOverrideStructure]:
        """Generate a :class:.`MeshOverrideStructure` `List` which is automatically generated
        from structures in the simulation.
        """
        mesh_overrides = []

        # For now we can suggest MeshOverrideStructures for lumped elements.
        for lumped_element in self.lumped_elements:
            mesh_overrides.extend(lumped_element.to_mesh_overrides())

        return mesh_overrides

    def subsection(
        self,
        region: Box,
        boundary_spec: BoundarySpec = None,
        grid_spec: Union[GridSpec, Literal["identical"]] = None,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry] = None,
        sources: Tuple[SourceType, ...] = None,
        monitors: Tuple[MonitorType, ...] = None,
        remove_outside_structures: bool = True,
        remove_outside_custom_mediums: bool = False,
        include_pml_cells: bool = False,
        **kwargs,
    ) -> AbstractYeeGridSimulation:
        """Generate a simulation instance containing only the ``region``.

        Parameters
        ----------
        region : :class:.`Box`
            New simulation domain.
        boundary_spec : :class:.`BoundarySpec` = None
            New boundary specification. If ``None``, then it is inherited from the original
            simulation.
        grid_spec : :class:.`GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:.`CustomGrid`. Note that in the latter case the region of the new simulation is
            snapped to the original grid lines.
        symmetry : Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
            New simulation symmetry. If ``None``, then it is inherited from the original
            simulation. Note that in this case the size and placement of new simulation domain
            must be commensurate with the original symmetry.
        sources : Tuple[SourceType, ...] = None
            New list of sources. If ``None``, then the sources intersecting the new simulation
            domain are inherited from the original simulation.
        monitors : Tuple[MonitorType, ...] = None
            New list of monitors. If ``None``, then the monitors intersecting the new simulation
            domain are inherited from the original simulation.
        remove_outside_structures : bool = True
            Remove structures outside of the new simulation domain.
        remove_outside_custom_mediums : bool = True
            Remove custom medium data outside of the new simulation domain.
        include_pml_cells : bool = False
            Keep PML cells in simulation boundaries. Note that retained PML cells will be converted
            to regular cells, and the simulation domain boundary will be moved accordingly.
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
                        new_bounds[0][dim] < self.center[dim]
                        and new_bounds[1][dim] > self.center[dim]
                    )

                    # inherit symmetry only if we cross symmetry plane, otherwise we don't impose
                    # symmetry even if the original simulation had symmetry
                    if crosses_symmetry:
                        symmetry[dim] = self.symmetry[dim]
                        center = (new_bounds[0][dim] + new_bounds[1][dim]) / 2

                        if not math.isclose(center, self.center[dim]):
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

        # inheritance of structures, sources, monitors, and boundary specs
        if remove_outside_structures:
            new_structures = [strc for strc in self.structures if new_box.intersects(strc.geometry)]
        else:
            new_structures = list(self.structures)

        if sources is None:
            sources = [src for src in self.sources if new_box.intersects(src)]

        # some nonlinear materials depend on the central frequency
        # we update them with hardcoded freq0
        freqs = np.array([source.source_time.freq0 for source in self.sources])
        for i, structure in enumerate(new_structures):
            medium = structure.medium
            nonlinear_spec = medium.nonlinear_spec
            if nonlinear_spec is not None:
                new_nonlinear_spec = nonlinear_spec._hardcode_medium_freqs(
                    medium=medium, freqs=freqs
                )
                new_structure = structure.updated_copy(
                    nonlinear_spec=new_nonlinear_spec, path="medium"
                )
                new_structures[i] = new_structure

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
                any(isinstance(edge, (PML, StablePML, Absorber)) for edge in boundary)
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
                    monitors=[],
                    sources=sources,  # need wavelength in case of auto grid
                    symmetry=symmetry,
                    structures=new_structures,
                )

                # then use its bounds as region for data cut off
                new_bounds = aux_sim.simulation_bounds

                # Note that this is not a full proof strategy. For example, if grid_spec is AutoGrid
                # then after outside custom medium data is removed the grid sizes and, thus,
                # pml extents can change as well

            # now cut out custom medium data
            new_structures_reduced_data = []

            for structure in new_structures:
                medium = structure.medium
                if isinstance(medium, AbstractCustomMedium):
                    new_structure_bounds = Box.bounds_intersection(
                        new_bounds, structure.geometry.bounds
                    )
                    new_medium = medium.sel_inside(bounds=new_structure_bounds)
                    new_structure = structure.updated_copy(medium=new_medium)
                    new_structures_reduced_data.append(new_structure)
                else:
                    new_structures_reduced_data.append(structure)

            new_structures = new_structures_reduced_data

            if isinstance(self.medium, AbstractCustomMedium):
                new_sim_medium = self.medium.sel_inside(bounds=new_bounds)

        # finally, create an updated copy with all modifications
        new_sim = self.updated_copy(
            center=new_box.center,
            size=new_box.size,
            medium=new_sim_medium,
            grid_spec=grid_spec,
            boundary_spec=boundary_spec,
            monitors=monitors,
            sources=sources,
            symmetry=symmetry,
            structures=new_structures,
            **kwargs,
        )

        return new_sim


class Simulation(AbstractYeeGridSimulation):
    """
    Custom implementation of Maxwell’s equations which represents the physical model to be solved using the FDTD
    method.

    Notes
    -----

        A ``Simulation`` defines a custom implementation of Maxwell's equations which represents the physical model
        to be solved using `the Finite-Difference Time-Domain (FDTD) method
        <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/>`_. ``tidy3d`` simulations
        run very quickly in the cloud through GPU parallelization.

        .. image:: ../../_static/img/field_update_fdtd.png
            :width: 50%
            :align: left

        FDTD is a method for simulating the interaction of electromagnetic waves with structures and materials. It is
        the most widely used method in photonics design. The Maxwell's
        equations implemented in the ``Simulation`` are solved per time-step in the order shown in this image.

        The simplified input to FDTD solver consists of the permittivity distribution defined by :attr:`structures`
        which describe the device and :attr:`sources` of electromagnetic excitation. This information is used to
        computate the time dynamics of the electric and magnetic fields in this system. From these time-domain
        results, frequency-domain information of the simulation can also be extracted, and used for device design and
        optimization.

        If you are new to the FDTD method, we recommend you get started with the `FDTD 101 Lecture Series
        <https://www.flexcompute.com/tidy3d/learning-center/fdtd101/>`_

        **Dimensions Selection**

        By default, simulations are defined as 3D. To make the simulation 2D, we can just set the simulation
        :attr:`size` in one of the dimensions to be 0. However, note that we still have to define a grid size (eg.
        ``tidy3d.Simulation(size=[size_x, size_y, 0])``) and specify a periodic boundary condition in that direction.

        .. TODO sort out inheritance problem https://aware-moon.cloudvent.net/tidy3d/examples/notebooks/RingResonator/

        See further parameter explanations below.

    Example
    -------
    >>> from tidy3d import Sphere, Cylinder, PolySlab
    >>> from tidy3d import UniformCurrentSource, GaussianPulse
    >>> from tidy3d import FieldMonitor, FluxMonitor
    >>> from tidy3d import GridSpec, AutoGrid
    >>> from tidy3d import BoundarySpec, Boundary
    >>> from tidy3d import Medium
    >>> sim = Simulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     grid_spec=GridSpec(
    ...         grid_x = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_y = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_z = AutoGrid(min_steps_per_wvl = 20)
    ...     ),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         UniformCurrentSource(
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
    ...         FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name='flux'),
    ...     ],
    ...     symmetry=(0, 0, 0),
    ...     boundary_spec=BoundarySpec(
    ...         x = Boundary.pml(num_layers=20),
    ...         y = Boundary.pml(num_layers=30),
    ...         z = Boundary.periodic(),
    ...     ),
    ...     shutoff=1e-6,
    ...     courant=0.8,
    ...     subpixel=False,
    ... )

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
        * See nearly all notebooks for :class:`Simulation` applications.

    **Lectures:**
        * `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/#presentation-slides>`_: Usage in a basic simulation flow.
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_

    **GUI:**
        * `FDTD Walkthrough <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-1-FDTD-Walkthrough/#presentation-slides>`_
    """

    boundary_spec: BoundarySpec = pydantic.Field(
        BoundarySpec(),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides.",
    )
    """Specification of boundary conditions along each dimension. If ``None``, :class:`PML` boundary conditions are
    applied on all sides.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             boundary_spec=BoundarySpec(
                x = Boundary.pml(num_layers=20),
                y = Boundary.pml(num_layers=30),
                z = Boundary.periodic(),
            ),
            ...
         )

    See Also
    --------

    :class:`PML`:
        A perfectly matched layer model.

    :class:`BoundarySpec`:
        Specifies boundary conditions on each side of the domain and along each dimension.

    `Index <../boundary_conditions.html>`_
        All boundary condition models.

    **Notebooks**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """

    courant: float = pydantic.Field(
        0.99,
        title="Normalized Courant Factor",
        description="Normalized Courant stability factor that is no larger than 1 when CFL "
        "stability condition is met. It controls time step to spatial step ratio. "
        "Lower values lead to more stable simulations for dispersive materials, "
        "but result in longer simulation times.",
        gt=0.0,
        le=1.0,
    )
    """The Courant-Friedrichs-Lewy (CFL) stability factor :math:`C`, controls time step to spatial step ratio.  A
    physical wave has to propagate slower than the numerical information propagation in a Yee-cell grid. This is
    because in this spatially-discrete grid, information propagates over 1 spatial step :math:`\\Delta x`
    over a time step :math:`\\Delta t`. This constraint enables the correct physics to be captured by the simulation.

    **1D Illustration**

    In a 1D model:

    .. image:: ../../_static/img/courant_instability.png

    Lower values lead to more stable simulations for dispersive materials, but result in longer simulation times. This
    factor is normalized to no larger than 1 when CFL stability condition is met in 3D.

    .. TODO finish this section for 1D, 2D and 3D references.

    For a 1D grid:

    .. math::

        C_{\\text{1D}} = \\frac{c \\Delta t}{\\Delta x} \\leq 1

    **2D Illustration**

    In a 2D uniform grid, where the :math:`E_z` field is at the red dot center surrounded by four green magnetic edge components
    in a square Yee cell grid:

    .. image:: ../../_static/img/courant_instability_2d.png

    .. math::

        C_{\\text{2D}} = \\frac{c\\Delta t}{\\Delta x} \\leq \\frac{1}{\\sqrt{2}}

    Hence, for the same spatial grid, the time step in 2D grid needs to be smaller than the time step in a 1D grid. Note we use
    a normalized Courant number in our simulation, which in 2D is :math:`\\sqrt{2}C_{\\text{2D}}`. CFL stability condition
    is met when the normalized Courant number is no larger than 1.

    **3D Illustration**

    For an isotropic medium with refractive index :math:`n`, the 3D time step condition can be derived to be:

    .. math::

        \\Delta t \\le \\frac{n}{c \\sqrt{\\frac{1}{\\Delta x^2} + \\frac{1}{\\Delta y^2} + \\frac{1}{\\Delta z^2}}}

    In this case, the number of spatial grid points scale by :math:`\\sim \\frac{1}{\\Delta x^3}` where :math:`\\Delta x`
    is the spatial discretization in the :math:`x` dimension. If the total simulation time is kept the same whilst
    maintaining the CFL condition, then the number of time steps required scale by :math:`\\sim \\frac{1}{\\Delta x}`.
    Hence, the spatial grid discretization influences the total time-steps required. The total simulation scaling per
    spatial grid size in this case is by :math:`\\sim \\frac{1}{\\Delta x^4}.`

    As an example, in this case, refining the mesh by a factor or 2 (reducing the spatial step size by half)
    :math:`\\Delta x \\to \\frac{\\Delta x}{2}` will increase the total simulation computational cost by 16.

    **Divergence Caveats**

    ``tidy3d`` uses a default Courant factor of 0.99. When a dispersive material with ``eps_inf < 1`` is used,
    the Courant factor will be automatically adjusted to be smaller than ``sqrt(eps_inf)`` to ensure stability. If
    your simulation still diverges despite addressing any other issues discussed above, reducing the Courant
    factor may help.

    See Also
    --------

    :attr:`grid_spec`
        Specifications for the simulation grid along each of the three directions.

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    lumped_elements: Tuple[LumpedElementType, ...] = pydantic.Field(
        (),
        title="Lumped Elements",
        description="Tuple of lumped elements in the simulation. "
        "Note: only :class:`tidy3d.LumpedResistor` is supported currently.",
    )
    """
    Tuple of lumped elements in the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            lumped_elements=[
                LumpedResistor(
                    size=(0, 3, 1),
                    center=(0, 0, 0),
                    voltage_axis=2,
                    resistance=50,
                    name="resistor_1",
                )
            ],
            ...
         )

    See Also
    --------

    `Index <../lumped_elements.html>`_:
        Available lumped element types.
    """

    grid_spec: GridSpec = pydantic.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )
    """
    Specifications for the simulation grid along each of the three directions.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             grid_spec=GridSpec(
                grid_x = AutoGrid(min_steps_per_wvl = 20),
                grid_y = AutoGrid(min_steps_per_wvl = 20),
                grid_z = AutoGrid(min_steps_per_wvl = 20)
            ),
            ...
         )

    **Usage Recommendations**

    In the *finite-difference* time domain method, the computational domain is discretized by a little cubes called
    the Yee cell. A discrete lattice formed by this Yee cell is used to describe the fields. In 3D, the electric
    fields are distributed on the edge of the Yee cell and the magnetic fields are distributed on the surface of the
    Yee cell.

    .. image:: ../../_static/img/yee_grid_illustration.png

    Note
    ----

        A typical rule of thumb is to choose the discretization to be about :math:`\\frac{\\lambda_m}{20}` where
        :math:`\\lambda_m` is the field wavelength.

    **Numerical Dispersion - 1D Illustration**

    Numerical dispersion is a form of numerical error dependent on the spatial and temporal discretization of the
    fields. In order to reduce it, it is necessary to improve the discretization of the simulation for particular
    frequencies and spatial features. This is an important aspect of defining the grid.

    Consider a standard 1D wave equation in vacuum:

    .. math::

        \\left( \\frac{\\delta ^2 }{\\delta x^2} - \\frac{1}{c^2} \\frac{\\delta^2}{\\delta t^2} \\right) E = 0

    which is ideally solved into a monochromatic travelling wave:

    .. math::

        E(x) = e^{j (kx - \\omega t)}

    This physical wave is described with a wavevector :math:`k` for the spatial field variations and the angular
    frequency :math:`\\omega` for temporal field variations. The spatial and temporal field variations are related by
    a dispersion relation.

    .. TODO explain the above more

    The ideal dispersion relation is:

    .. math::

        \\left( \\frac{\\omega}{c} \\right)^2 = k^2

    However, in the FDTD simulation, the spatial and temporal fields are discrete.

    .. TODO improve the ways figures are represented.

    .. image:: ../../_static/img/numerical_dispersion_grid_1d.png
        :width: 30%
        :align: right

    The same 1D monochromatic wave can be solved using the FDTD method where :math:`m` is the index in the grid:

    .. math::

        \\frac{\\delta^2}{\\delta x^2} E(x_i) \\approx \\frac{1}{\\Delta x^2} \\left[ E(x_i + \\Delta x) + E(x_i -
        \\Delta x) - 2 E(x_i) \\right]

    .. math::

        \\frac{\\delta^2}{\\delta t^2} E(t_{\\alpha}) \\approx \\frac{1}{\\Delta t^2} \\left[ E(t_{\\alpha} + \\Delta
        t) + E(t_{\\alpha} - \\Delta t) - 2 E(t_{\\alpha}) \\right]

    .. TODO define the alpha

    Hence, these discrete fields have this new dispersion relation:

    .. math::

        \\left( \\frac{1}{c \\Delta t} \\text{sin} \\left( \\frac{\\omega \\Delta t}{2} \\right)^2 \\right) = \\left(
        \\frac{1}{\\Delta x} \\text{sin} \\left( \\frac{k \\Delta x}{2} \\right) \\right)^2

    The ideal wave solution and the discrete solution have a mismatch illustrated below as a result of the numerical
    error introduced by numerical dispersion. This plot illustrates the angular frequency as a function of wavevector
    for both the physical ideal wave and the numerical discrete wave implemented in FDTD.

    .. image:: ../../_static/img/numerical_dispersion_discretization_1d.png

    .. TODO improve these images positions

    At lower frequencies, when the discretization of :math:`\\Delta x` is small compared to the wavelength the error
    between the solutions is very low. When this proportionality increases between the spatial step size and the
    angular wavelength, this introduces numerical dispersion errors.

    .. math::

        k \\Delta x = \\frac{2 \\pi}{\\lambda_k} \\Delta x


    **Usage Recommendations**

    *   It is important to understand the relationship between the time-step :math:`\\Delta t` defined by the
        :attr:`courant` factor, and the spatial grid distribution to guarantee simulation stability.

    *   If your structure has small features, consider using a spatially nonuniform grid. This guarantees finer
        spatial resolution near the features, but away from it you use have a larger (and computationally faster) grid.
        In this case, the time step :math:`\\Delta t` is defined by the smallest spatial grid size.

    See Also
    --------

    :attr:`courant`
        The Courant-Friedrichs-Lewy (CFL) stability factor

    :class:`GridSpec`
        Collective grid specification for all three dimensions.

    :class:`UniformGrid`
        Uniform 1D grid.

    :class:`AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    medium: MediumType3D = pydantic.Field(
        Medium(),
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
        discriminator=TYPE_TAG_STR,
    )
    """
    Background medium of simulation, defaults to vacuum if not specified.

    See Also
    --------

    `Material Library <../material_library.html>`_:
        The material library is a dictionary containing various dispersive models from real world materials.

    `Index <../mediums.html>`_:
        Dispersive and dispersionless Mediums models.

    **Notebooks:**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures:**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

    **GUI:**

    * `Mediums <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-2-Mediums/>`_

    """

    normalize_index: Union[pydantic.NonNegativeInt, None] = pydantic.Field(
        0,
        title="Normalization index",
        description="Index of the source in the tuple of sources whose spectrum will be used to "
        "normalize the frequency-dependent data. If ``None``, the raw field data is returned "
        "unnormalized.",
    )
    """
    Index of the source in the tuple of sources whose spectrum will be used to normalize the frequency-dependent
    data. If ``None``, the raw field data is returned. If ``None``, the raw field data is returned unnormalized.
    """

    monitors: Tuple[annotate_type(MonitorType), ...] = pydantic.Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )
    """
    Tuple of monitors in the simulation. Monitor names are used to access data after simulation is run.

    See Also
    --------

    `Index <../monitors.html>`_
        All the monitor implementations.
    """

    sources: Tuple[annotate_type(SourceType), ...] = pydantic.Field(
        (),
        title="Sources",
        description="Tuple of electric current sources injecting fields into the simulation.",
    )
    """
    Tuple of electric current sources injecting fields into the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            sources=[
                UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, 0.5, 0),
                    polarization="Hx",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=4e13,
                    ),
                )
            ],
            ...
         )

    See Also
    --------

    `Index <../sources.html>`_:
        Frequency and time domain source models.
    """

    shutoff: pydantic.NonNegativeFloat = pydantic.Field(
        1e-5,
        title="Shutoff Condition",
        description="Ratio of the instantaneous integrated E-field intensity to the maximum value "
        "at which the simulation will automatically terminate time stepping. "
        "Used to prevent extraneous run time of simulations with fully decayed fields. "
        "Set to ``0`` to disable this feature.",
    )
    """
    Ratio of the instantaneous integrated E-field intensity to the maximum value
    at which the simulation will automatically terminate time stepping.
    Used to prevent extraneous run time of simulations with fully decayed fields.
    Set to ``0`` to disable this feature.
    """

    structures: Tuple[Structure, ...] = pydantic.Field(
        (),
        title="Structures",
        description="Tuple of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )
    """
    Tuple of structures present in simulation. Structures defined later in this list override the simulation
    material properties in regions of spatial overlap.

    Example
    -------
    Simple application reference:

    .. code-block:: python

        Simulation(
            ...
            structures=[
                 Structure(
                 geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                 medium=Medium(permittivity=2.0),
                 ),
            ],
            ...
        )

    **Usage Caveats**

    It is very important to understand the way the dielectric permittivity of the :class:`Structure` list is resolved
    by the simulation grid. Without :attr:`subpixel` averaging, the structure geometry in relation to the
    grid points can lead to its features permittivity not being fully resolved by the
    simulation.

    For example, in the image below, two silicon slabs with thicknesses 150nm and 175nm centered in a grid with
    spatial discretization :math:`\\Delta z = 25\\text{nm}` will compute equivalently because that grid does
    not resolve the feature permittivity in between grid points without :attr:`subpixel` averaging.

    .. image:: ../../_static/img/permittivity_on_yee_grid.png

    See Also
    --------

    :class:`Structure`:
        Defines a physical object that interacts with the electromagnetic fields.

    :attr:`subpixel`
        Subpixel averaging of the permittivity based on structure definition, resulting in much higher
        accuracy for a given grid size.

    **Notebooks:**

    * `Visualizing geometries in Tidy3D <../../notebooks/VizSimulation.html>`_

    **Lectures:**

    * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`_
    *  `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_

    **GUI:**

    * `Structures <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-3-Structures/#presentation-slides>`_
    """

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pydantic.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or "
        "``-1`` (odd, i.e. 'PEC' symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )
    """
    You should set the ``symmetry`` parameter in your :class:`Simulation` object using a tuple of integers
    defining reflection symmetry across a plane bisecting the simulation domain normal to the x-, y-, and z-axis.
    Each element can be 0 (no symmetry), 1 (even, i.e. :class:`PMC` symmetry) or -1 (odd, i.e. :class:`PEC`
    symmetry). Note that the vectorial nature of the fields must be considered to determine the symmetry value
    correctly.

    The figure below illustrates how the electric and magnetic field components transform under :class:`PEC`- and
    :class:`PMC`-like symmetry planes. You can refer to this figure when considering whether a source field conforms
    to a :class:`PEC`- or :class:`PMC`-like symmetry axis. This would be helpful, especially when dealing with
    optical waveguide modes.

    .. image:: ../../notebooks/img/pec_pmc.png


    .. TODO maybe resize?
    """

    # TODO: at a later time (once well tested) we could consider making default of RunTimeSpec()
    run_time: Union[pydantic.PositiveFloat, RunTimeSpec] = pydantic.Field(
        ...,
        title="Run Time",
        description="Total electromagnetic evolution time in seconds. "
        "Note: If simulation 'shutoff' is specified, "
        "simulation will terminate early when shutoff condition met. "
        "Alternatively, user may supply a :class:`RunTimeSpec` to this field, which will auto-"
        "compute the ``run_time`` based on the contents of the spec. If this option is used, "
        "the evaluated ``run_time`` value is available in the ``Simulation._run_time`` property.",
        units=SECOND,
    )
    """
    Total electromagnetic evolution time in seconds. If simulation 'shutoff' is specified, simulation will
    terminate early when shutoff condition met.

    **How long to run a simulation?**

    The frequency-domain response obtained in the FDTD simulation only accurately represents the continuous-wave
    response of the system if the fields at the beginning and at the end of the time stepping are (very close to)
    zero. So, you should run the simulation for a time enough to allow the electromagnetic fields decay to negligible
    values within the simulation domain.

    When dealing with light propagation in a NON-RESONANT device, like a simple optical waveguide, a good initial
    guess to simulation run_time would be the a few times the largest domain dimension (:math:`L`) multiplied by the
    waveguide mode group index (:math:`n_g`), divided by the speed of light in a vacuum (:math:`c_0`),
    plus the ``source_time``:

    .. math::

        t_{sim} \\approx \\frac{n_g L}{c_0} + t_{source}

    By default, ``tidy3d`` checks periodically the total field intensity left in the simulation, and compares that to
    the maximum total field intensity recorded at previous times. If it is found that the ratio of these two values
    is smaller than the default :attr:`shutoff` value :math:`10^{-5}`, the simulation is terminated as the fields
    remaining in the simulation are deemed negligible. The shutoff value can be controlled using the :attr:`shutoff`
    parameter, or completely turned off by setting it to zero. In most cases, the default behavior ensures that
    results are correct, while avoiding unnecessarily long run times. The Flex Unit cost of the simulation is also
    proportionally scaled down when early termination is encountered.

    **Resonant Caveats**

    Should I make sure that fields have fully decayed by the end of the simulation?

    The main use case in which you may want to ignore the field decay warning is when you have high-Q modes in your
    simulation that would require an extremely long run time to decay. In that case, you can use the the
    :class:`tidy3d.plugins.resonance.ResonanceFinder` plugin to analyze the modes, as well as field monitors with
    vaporization to capture the modal profiles. The only thing to note is that the normalization of these modal
    profiles would be arbitrary, and would depend on the exact run time and apodization definition. An example of
    such a use case is presented in our case study.

    .. TODO add links to resonant plugins.

    See Also
    --------

    **Notebooks**

    *   `High-Q silicon resonator <../../notebooks/HighQSi.html>`_

    """

    """ Validating setup """

    @pydantic.root_validator(pre=True)
    def _update_simulation(cls, values):
        """Update the simulation if it is an earlier version."""

        # if no version, assume it's already updated
        if "version" not in values:
            return values

        # otherwise, call the updator to update the values dictionary
        updater = Updater(sim_dict=values)
        return updater.update_to_current()

    @pydantic.validator("grid_spec", always=True)
    @skip_if_fields_missing(["sources"])
    def _validate_auto_grid_wavelength(cls, val, values):
        """Check that wavelength can be defined if there is auto grid spec."""
        if val.wavelength is None and val.auto_grid_used:
            _ = val.wavelength_from_sources(sources=values.get("sources"))
        return val

    _sources_in_bounds = assert_objects_in_sim_bounds("sources", strict_inequality=True)
    _mode_sources_symmetries = validate_mode_objects_symmetry("sources")
    _mode_monitors_symmetries = validate_mode_objects_symmetry("monitors")

    # _few_enough_mediums = validate_num_mediums()
    # _structures_not_at_edges = validate_structure_bounds_not_at_edges()
    # _gap_size_ok = validate_pml_gap_size()
    # _medium_freq_range_ok = validate_medium_frequency_range()
    # _resolution_fine_enough = validate_resolution()
    # _plane_waves_in_homo = validate_plane_wave_intersections()

    @pydantic.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["symmetry"])
    def bloch_with_symmetry(cls, val, values):
        """Error if a Bloch boundary is applied with symmetry"""
        boundaries = val.to_list
        symmetry = values.get("symmetry")
        for dim, boundary in enumerate(boundaries):
            num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
            if num_bloch > 0 and symmetry[dim] != 0:
                raise SetupError(
                    f"Bloch boundaries cannot be used with a symmetry along dimension {dim}."
                )
        return val

    @pydantic.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["medium", "size", "structures", "sources"])
    def plane_wave_boundaries(cls, val, values):
        """Error if there are plane wave sources incompatible with boundary conditions."""
        boundaries = val.to_list
        sources = values.get("sources")
        size = values.get("size")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        for source_ind, source in enumerate(sources):
            if not isinstance(source, PlaneWave):
                continue

            _, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = Scene.intersecting_media(source, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # check the PML/absorber + angled plane wave case
                num_pml = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
                if num_pml > 0 and source.angle_theta != 0:
                    raise SetupError(
                        "Angled plane wave sources are not compatible with the absorbing boundary "
                        f"along dimension {tan_dir}. Either set the source ``angle_theta`` to "
                        "``0``, or use Bloch boundaries that match the source angle."
                    )

                # check the Bloch boundary + angled plane wave case
                num_bloch = sum(isinstance(bnd, (Periodic, BlochBoundary)) for bnd in boundary)
                if num_bloch > 0:
                    cls._check_bloch_vec(
                        source=source,
                        source_ind=source_ind,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                    )
        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["boundary_spec", "medium", "size", "structures", "sources"])
    def bloch_boundaries_diff_mnt(cls, val, values):
        """Error if there are diffraction monitors incompatible with boundary conditions."""

        monitors = val

        if not val or not any(isinstance(mnt, DiffractionMonitor) for mnt in monitors):
            return val

        boundaries = values.get("boundary_spec").to_list
        sources = values.get("sources")
        size = values.get("size")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        for source_ind, source in enumerate(sources):
            if not isinstance(source, PlaneWave):
                continue

            _, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = Scene.intersecting_media(source, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # check the Bloch boundary + angled plane wave case
                num_bloch = sum(isinstance(bnd, (Periodic, BlochBoundary)) for bnd in boundary)
                if num_bloch > 0:
                    cls._check_bloch_vec(
                        source=source,
                        source_ind=source_ind,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                        has_diff_mnt=True,
                    )
        return val

    @pydantic.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["medium", "center", "size", "structures", "sources"])
    def tfsf_boundaries(cls, val, values):
        """Error if the boundary conditions are compatible with TFSF sources, if any."""
        boundaries = val.to_list
        sources = values.get("sources")
        size = values.get("size")
        center = values.get("center")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        sim_bounds = [
            [c - s / 2.0 for c, s in zip(center, size)],
            [c + s / 2.0 for c, s in zip(center, size)],
        ]
        for src_idx, source in enumerate(sources):
            if not isinstance(source, TFSF):
                continue

            norm_dir, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            src_bounds = source.bounds

            # make a dummy source that represents the injection surface to get the intersecting
            # medium, which is later used to test the Bloch vector for correctness
            temp_size = list(source.size)
            temp_size[source.injection_axis] = 0
            temp_src = Source(
                center=source.injection_plane_center,
                size=temp_size,
                source_time=source.source_time,
            )
            medium_set = Scene.intersecting_media(temp_src, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            # the source shouldn't touch or cross any boundary in the direction of injection
            if (
                src_bounds[0][norm_dir] <= sim_bounds[0][norm_dir]
                or src_bounds[1][norm_dir] >= sim_bounds[1][norm_dir]
            ):
                raise SetupError(
                    f"The TFSF source at index '{src_idx}' must not touch or cross the "
                    f"simulation boundary along its injection axis, '{['x', 'y', 'z'][norm_dir]}'."
                )

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # crossing may be allowed for periodic or Bloch boundaries, but not others
                if (
                    src_bounds[0][tan_dir] <= sim_bounds[0][tan_dir]
                    or src_bounds[1][tan_dir] >= sim_bounds[1][tan_dir]
                ):
                    # if the boundary is Bloch periodic, crossing is allowed, but check that the
                    # Bloch vector has been correctly set, similar to the check for plane waves
                    num_bloch = sum(isinstance(bnd, (Periodic, BlochBoundary)) for bnd in boundary)
                    if num_bloch == 2:
                        cls._check_bloch_vec(
                            source=source,
                            source_ind=src_idx,
                            bloch_vec=boundary[0].bloch_vec,
                            dim=tan_dir,
                            medium=medium,
                            domain_size=size[tan_dir],
                        )
                        continue

                    # for any other boundary, the source must not cross the boundary
                    raise SetupError(
                        f"The TFSF source at index '{src_idx}' must not touch or cross the "
                        f"simulation boundary in the '{['x', 'y', 'z'][tan_dir]}' direction, "
                        "unless that boundary is 'Periodic' or 'BlochBoundary'."
                    )

        return val

    @pydantic.validator("sources", always=True)
    @skip_if_fields_missing(["symmetry"])
    def tfsf_with_symmetry(cls, val, values):
        """Error if a TFSF source is applied with symmetry"""
        symmetry = values.get("symmetry")
        for source in val:
            if isinstance(source, TFSF) and not all(sym == 0 for sym in symmetry):
                raise SetupError("TFSF sources cannot be used with symmetries.")
        return val

    @pydantic.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["size", "symmetry"])
    def boundaries_for_zero_dims(cls, val, values):
        """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""
        boundaries = val.to_list
        size = values.get("size")
        symmetry = values.get("symmetry")
        axis_names = "xyz"

        for dim, (boundary, symmetry_dim, size_dim) in enumerate(zip(boundaries, symmetry, size)):
            if size_dim == 0:
                axis = axis_names[dim]
                num_absorbing_bdries = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
                num_bloch_bdries = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)

                if num_absorbing_bdries > 0:
                    raise SetupError(
                        f"The simulation has zero size along the {axis} axis, so "
                        "using a PML or absorbing boundary along that axis is incorrect. "
                        f"Use either 'Periodic' or 'BlochBoundary' along {axis}."
                    )

                if num_bloch_bdries > 0:
                    raise SetupError(
                        f"The simulation has zero size along the {axis} axis, "
                        "using a Bloch boundary along such an axis is not supported because of "
                        "the Bloch vector definition in units of '2 * pi / (size along dimension)'. Use a small "
                        "but nonzero size along the dimension instead."
                    )

                if symmetry_dim != 0:
                    raise SetupError(
                        f"The simulation has zero size along the {axis} axis, so "
                        "using symmetry along that axis is incorrect. Use 'PECBoundary' "
                        "or 'PMCBoundary' to select source polarization if needed and set "
                        f"Simulation.symmetry to 0 along {axis}."
                    )

                if boundary[0] != boundary[1]:
                    raise SetupError(
                        f"The simulation has zero size along the {axis} axis. "
                        f"The boundary condition for {axis} plus and {axis} "
                        "minus must be the same."
                    )

        return val

    @pydantic.validator("sources", always=True)
    def _validate_num_sources(cls, val):
        """Error if too many sources present."""

        if val is None:
            return val

        if len(val) > MAX_NUM_SOURCES:
            raise SetupError(
                f"Number of distinct sources exceeds the maximum allowed {MAX_NUM_SOURCES}. "
                "For a complex source setup, consider using 'CustomFieldSource' or "
                "'CustomCurrentSource' to combine multiple sources into one object."
            )

        return val

    @pydantic.validator("structures", always=True)
    def _validate_2d_geometry_has_2d_medium(cls, val, values):
        """Warn if a geometry bounding box has zero size in a certain dimension."""

        if val is None:
            return val

        with log as consolidated_logger:
            for i, structure in enumerate(val):
                if isinstance(structure.medium, Medium2D):
                    continue
                for geom in flatten_groups(structure.geometry):
                    zero_dims = geom.zero_dims
                    if len(zero_dims) > 0:
                        consolidated_logger.warning(
                            f"Structure at 'structures[{i}]' has geometry with zero size along "
                            f"dimensions {zero_dims}, and with a medium that is not a 'Medium2D'. "
                            "This is probably not correct, since the resulting simulation will "
                            "depend on the details of the numerical grid. Consider either "
                            "giving the geometry a nonzero thickness or using a 'Medium2D'."
                        )

        return val

    @pydantic.validator("structures", always=True)
    def _validate_incompatible_material_intersections(cls, val, values):
        """Check for intersections of incompatible materials."""
        structures = val
        incompatible_indices = []
        incompatible_structures = []
        # first just isolate the incompatible structures, to avoid unnecessary double looping
        # keep track of indices to give helpful error message
        for i, structure in enumerate(structures):
            if structure.medium._has_incompatibilities:
                incompatible_indices.append(i)
                incompatible_structures.append(structure)
        for i, (ind1, structure_ind1) in enumerate(
            zip(incompatible_indices, incompatible_structures)
        ):
            for ind2, structure_ind2 in zip(
                incompatible_indices[i + 1 :], incompatible_structures[i + 1 :]
            ):
                if not structure_ind1._compatible_with(structure_ind2):
                    raise ValidationError(
                        f"The structure at 'structures[{ind1}]' and the structure at "
                        f"'structures[{ind2}]' have incompatible medium types "
                        f"{structure_ind1.medium._incompatible_material_types} and "
                        f"{structure_ind2.medium._incompatible_material_types} "
                        "respectively, and so are not allowed to intersect. "
                        "Please ensure that the bounding boxes of the two geometries "
                        "do not intersect."
                    )
        return val

    @pydantic.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["sources", "center", "size", "structures"])
    def _structures_not_close_pml(cls, val, values):
        """Warn if any structures lie at the simulation boundaries."""

        sim_box = Box(size=values.get("size"), center=values.get("center"))
        sim_bound_min, sim_bound_max = sim_box.bounds

        boundaries = val.to_list
        structures = values.get("structures")
        sources = values.get("sources")

        if (not structures) or (not sources):
            return val

        with log as consolidated_logger:

            def warn(istruct, side):
                """Warning message for a structure too close to PML."""
                consolidated_logger.warning(
                    f"Structure at structures[{istruct}] was detected as being less "
                    f"than half of a central wavelength from a PML on side {side}. "
                    "To avoid inaccurate results or divergence, please increase gap between "
                    "any structures and PML or fully extend structure through the pml.",
                    custom_loc=["structures", istruct],
                )

            for istruct, structure in enumerate(structures):
                struct_bound_min, struct_bound_max = structure.geometry.bounds

                for source in sources:
                    lambda0 = C_0 / source.source_time.freq0

                    zipped = zip(["x", "y", "z"], sim_bound_min, struct_bound_min, boundaries)
                    for axis, sim_val, struct_val, boundary in zipped:
                        # The test is required only for PML and stable PML
                        if not isinstance(boundary[0], (PML, StablePML)):
                            continue
                        if (
                            boundary[0].num_layers > 0
                            and struct_val > sim_val
                            and abs(sim_val - struct_val) < lambda0 / 2
                        ):
                            warn(istruct, axis + "-min")

                    zipped = zip(["x", "y", "z"], sim_bound_max, struct_bound_max, boundaries)
                    for axis, sim_val, struct_val, boundary in zipped:
                        # The test is required only for PML and stable PML
                        if not isinstance(boundary[1], (PML, StablePML)):
                            continue
                        if (
                            boundary[1].num_layers > 0
                            and struct_val < sim_val
                            and abs(sim_val - struct_val) < lambda0 / 2
                        ):
                            warn(istruct, axis + "-max")

        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["medium", "structures"])
    def _warn_monitor_mediums_frequency_range(cls, val, values):
        """Warn user if any DFT monitors have frequencies outside of medium frequency range."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = structures or []
        medium_bg = values.get("medium")
        mediums = [medium_bg] + [structure.medium for structure in structures]

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor):
                    continue

                freqs = np.array(monitor.freqs)
                fmin_mon = freqs.min()
                fmax_mon = freqs.max()
                for medium_index, medium in enumerate(mediums):
                    # skip mediums that have no freq range (all freqs valid)
                    if medium.frequency_range is None:
                        continue

                    # make sure medium frequency range includes all monitor frequencies
                    fmin_med, fmax_med = medium.frequency_range
                    if fmin_mon < fmin_med or fmax_mon > fmax_med:
                        if medium_index == 0:
                            medium_str = "The simulation background medium"
                            custom_loc = ["medium", "frequency_range"]
                        else:
                            medium_str = (
                                f"The medium associated with structures[{medium_index - 1}]"
                            )
                            custom_loc = [
                                "structures",
                                medium_index - 1,
                                "medium",
                                "frequency_range",
                            ]

                        consolidated_logger.warning(
                            f"{medium_str} has a frequency range: ({fmin_med:2e}, {fmax_med:2e}) "
                            "(Hz) that does not fully cover the frequencies contained in "
                            f"monitors[{monitor_index}]. "
                            "This can cause inaccuracies in the recorded results.",
                            custom_loc=custom_loc,
                        )

        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["sources"])
    def _warn_monitor_simulation_frequency_range(cls, val, values):
        """Warn if any DFT monitors have frequencies outside of the simulation frequency range."""

        if val is None:
            return val

        source_ranges = [source.source_time.frequency_range() for source in values["sources"]]
        if not source_ranges:
            # Commented out to eliminate this message from Mode real time log in GUI
            # TODO: Bring it back when it doesn't interfere with mode solver
            # log.info("No sources in simulation.")
            return val

        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor):
                    continue

                freqs = np.array(monitor.freqs)
                if freqs.min() < freq_min or freqs.max() > freq_max:
                    consolidated_logger.warning(
                        f"monitors[{monitor_index}] contains frequencies "
                        f"outside of the simulation frequency range ({freq_min:2e}, {freq_max:2e})"
                        "(Hz) as defined by the sources.",
                        custom_loc=["monitors", monitor_index, "freqs"],
                    )
        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["boundary_spec"])
    def diffraction_monitor_boundaries(cls, val, values):
        """If any :class:`.DiffractionMonitor` exists, ensure boundary conditions in the
        transverse directions are periodic or Bloch."""
        monitors = val
        boundary_spec = values.get("boundary_spec")
        for monitor in monitors:
            if isinstance(monitor, DiffractionMonitor):
                _, (n_x, n_y) = monitor.pop_axis(["x", "y", "z"], axis=monitor.normal_axis)
                boundaries = [
                    boundary_spec[n_x].plus,
                    boundary_spec[n_x].minus,
                    boundary_spec[n_y].plus,
                    boundary_spec[n_y].minus,
                ]
                # make sure the transverse boundaries are either periodic or Bloch
                for boundary in boundaries:
                    if not isinstance(boundary, (Periodic, BlochBoundary)):
                        raise SetupError(
                            f"The 'DiffractionMonitor' {monitor.name} requires periodic "
                            f"or Bloch boundaries along dimensions {n_x} and {n_y}."
                        )
        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["medium", "center", "size", "structures"])
    def _projection_monitors_homogeneous(cls, val, values):
        """Error if any field projection monitor is not in a homogeneous region."""

        if val is None:
            return val

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=values.get("size"),
                center=values.get("center"),
            ),
            medium=values.get("medium"),
        )

        structures = values.get("structures") or []
        total_structures = [structure_bg] + list(structures)

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if isinstance(monitor, (AbstractFieldProjectionMonitor, DiffractionMonitor)):
                    mediums = Scene.intersecting_media(monitor, total_structures)
                    # make sure there is no more than one medium in the returned list
                    if len(mediums) > 1:
                        raise SetupError(
                            f"{len(mediums)} different mediums detected on plane "
                            f"intersecting a {monitor.type}. Plane must be homogeneous."
                        )
                    # 0 medium, something is wrong
                    if len(mediums) < 1:
                        raise SetupError(
                            f"No medium detected on plane intersecting a {monitor.type}, "
                            "indicating an unexpected error. Please create a github issue so "
                            "that the problem can be investigated."
                        )
                    # 1 medium, check if the medium is spatially uniform
                    if not list(mediums)[0].is_spatially_uniform:
                        consolidated_logger.warning(
                            f"Nonuniform custom medium detected on plane intersecting a {monitor.type}. "
                            "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                            custom_loc=["monitors", monitor_ind],
                        )

        return val

    @pydantic.validator("monitors", always=True)
    def _projection_direction(cls, val, values):
        """Warn if field projection observation points are behind surface projection monitors."""
        # This validator is in simulation.py rather than monitor.py because volume monitors are
        # eventually converted to their bounding surface projection monitors, in which case we
        # do not want this validator to be triggered.

        if val is None:
            return val

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if monitor.size.count(0.0) != 1:
                        continue

                    normal_dir = monitor.projection_surfaces[0].normal_dir
                    normal_ind = monitor.size.index(0.0)

                    projecting_backwards = False
                    if isinstance(monitor, FieldProjectionAngleMonitor):
                        r, theta, phi = np.meshgrid(
                            monitor.proj_distance,
                            monitor.theta,
                            monitor.phi,
                            indexing="ij",
                        )
                        x, y, z = Geometry.sph_2_car(r=r, theta=theta, phi=phi)
                    elif isinstance(monitor, FieldProjectionKSpaceMonitor):
                        uxs, uys, _ = np.meshgrid(
                            monitor.ux,
                            monitor.uy,
                            monitor.proj_distance,
                            indexing="ij",
                        )
                        theta, phi = monitor.kspace_2_sph(uxs, uys, monitor.proj_axis)
                        x, y, z = Geometry.sph_2_car(r=monitor.proj_distance, theta=theta, phi=phi)
                    else:
                        pts = monitor.unpop_axis(
                            monitor.proj_distance, (monitor.x, monitor.y), axis=normal_ind
                        )
                        x, y, z = pts

                    center = np.array(monitor.center) - np.array(monitor.local_origin)
                    pts = [np.array(i) for i in [x, y, z]]
                    normal_displacement = pts[normal_ind] - center[normal_ind]
                    if np.any(normal_displacement < 0) and normal_dir == "+":
                        projecting_backwards = True
                    elif np.any(normal_displacement > 0) and normal_dir == "-":
                        projecting_backwards = True

                    if projecting_backwards:
                        consolidated_logger.warning(
                            f"Field projection monitor '{monitor.name}' has observation points set "
                            "up such that the monitor is projecting backwards with respect to its "
                            "'normal_dir'. If this was not intentional, please take a look at the "
                            "documentation associated with this type of projection monitor to "
                            "check how the observation point coordinate system is defined.",
                            custom_loc=["monitors", monitor_ind],
                        )

        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["size"])
    def proj_distance_for_approx(cls, val, values):
        """Warn if projection distance for projection monitors is not large compared to monitor or,
        simulation size, yet far_field_approx is True."""
        if val is None:
            return val

        sim_size = values.get("size")

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if not isinstance(monitor, AbstractFieldProjectionMonitor):
                    continue

                name = monitor.name
                max_size = min(np.max(monitor.size), np.max(sim_size))

                if monitor.far_field_approx and np.abs(monitor.proj_distance) < 10 * max_size:
                    consolidated_logger.warning(
                        f"Monitor {name} projects to a distance comparable to the size of the "
                        "monitor; we recommend setting ``far_field_approx=False`` to disable "
                        "far-field approximations for this monitor, because the approximations "
                        "are valid only when the observation points are very far compared to the "
                        "size of the monitor that records near fields.",
                        custom_loc=["monitors", monitor_ind],
                    )
        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["center", "size"])
    def _integration_surfaces_in_bounds(cls, val, values):
        """Error if all of the integration surfaces are outside of the simulation domain."""

        if val is None:
            return val

        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        for mnt in (mnt for mnt in val if isinstance(mnt, SurfaceIntegrationMonitor)):
            if not any(sim_box.intersects(surf) for surf in mnt.integration_surfaces):
                raise SetupError(
                    f"All integration surfaces of monitor '{mnt.name}' are outside of the "
                    "simulation bounds."
                )

        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["size"])
    def _projection_monitors_distance(cls, val, values):
        """Warn if the projection distance is large for exact projections."""

        if val is None:
            return val

        sim_size = values.get("size")

        with log as consolidated_logger:
            for idx, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if (
                        np.abs(monitor.proj_distance) > 1.0e4 * np.max(sim_size)
                        and not monitor.far_field_approx
                    ):
                        monitor = monitor.copy(update={"far_field_approx": True})
                        val = list(val)
                        val[idx] = monitor
                        val = tuple(val)
                        consolidated_logger.warning(
                            "A very large projection distance was set for the field projection "
                            f"monitor '{monitor.name}'. Using exact field projections may result "
                            "in precision loss for large distances; automatically enabling "
                            "far-field approximations ('far_field_approx = True') for better "
                            "precision. To insist on exact projections, consider using client-side "
                            "projections via the 'FieldProjector' class, where higher precision is "
                            "available.",
                            custom_loc=["monitors", idx, "proj_distance"],
                        )
        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["size"])
    def _projection_mnts_2d(cls, val, values):
        """
        Validate if the field projection monitor is set up for a 2D simulation and
        ensure the observation parameters are configured correctly.

        - For a 2D simulation in the x-y plane, ``theta`` should be set to ``pi/2``.
        - For a 2D simulation in the y-z plane, ``phi`` should be set to ``pi/2`` or ``3*pi/2``.
        - For a 2D simulation in the x-z plane, ``phi`` should be set to ``0`` or ``pi``.

        Note: Exact far field projection is not available yet. Currently, only
        'far_field_approx = True' is supported.
        """

        if val is None:
            return val

        sim_size = values.get("size")

        # Validation if is 3D simulation
        non_zero_dims = sum(1 for size in sim_size if size != 0)
        if non_zero_dims == 3:
            return val

        plane = None

        if sim_size[0] == 0:
            plane = "y-z"
        elif sim_size[1] == 0:
            plane = "x-z"
        elif sim_size[2] == 0:
            plane = "x-y"

        for monitor in val:
            if isinstance(monitor, AbstractFieldProjectionMonitor):
                if non_zero_dims == 1:
                    raise SetupError(
                        f"Monitor '{monitor.name}' is not supported in 1D simulations."
                    )

                if isinstance(monitor, FieldProjectionAngleMonitor):
                    config = {
                        "y-z": {"valid_value": [np.pi / 2, 3 * np.pi / 2], "coord": "phi"},
                        "x-z": {"valid_value": [0, np.pi], "coord": "phi"},
                        "x-y": {"valid_value": [np.pi / 2], "coord": "theta"},
                    }[plane]

                    coord = getattr(monitor, config["coord"])
                    if not all(value in config["valid_value"] for value in coord):
                        replacements = {
                            np.pi: "np.pi",
                            np.pi / 2: "np.pi/2",
                            3 * np.pi / 2: "3*np.pi/2",
                            0: "0",
                        }
                        valid_values_str = ", ".join(
                            replacements.get(val) for val in config["valid_value"]
                        )
                        raise SetupError(
                            f"For a 2D simulation in the {plane} plane, the observation "
                            f"angle '{config['coord']}' of monitor "
                            f"'{monitor.name}' should be set to "
                            f"'{valid_values_str}'"
                        )

                    continue

                if isinstance(monitor, (FieldProjectionCartesianMonitor)):
                    config = {
                        "y-z": {"valid_proj_axes": [1, 2], "coord": ["x", "x"]},
                        "x-z": {"valid_proj_axes": [0, 2], "coord": ["x", "y"]},
                        "x-y": {"valid_proj_axes": [0, 1], "coord": ["y", "y"]},
                    }[plane]
                elif isinstance(monitor, (FieldProjectionKSpaceMonitor)):
                    config = {
                        "y-z": {"valid_proj_axes": [1, 2], "coord": ["ux", "ux"]},
                        "x-z": {"valid_proj_axes": [0, 2], "coord": ["ux", "uy"]},
                        "x-y": {"valid_proj_axes": [0, 1], "coord": ["uy", "uy"]},
                    }[plane]

                valid_proj_axes = config["valid_proj_axes"]
                invalid_proj_axis = [i for i in range(3) if i not in valid_proj_axes]

                if monitor.proj_axis in invalid_proj_axis:
                    raise SetupError(
                        f"For a 2D simulation in the {plane} plane, the 'proj_axis' of "
                        f"monitor '{monitor.name}' should be set to one of {valid_proj_axes}."
                    )

                for idx, axis in enumerate(valid_proj_axes):
                    coord = getattr(monitor, config["coord"][idx])
                    if monitor.proj_axis == axis and not all(value in [0] for value in coord):
                        raise SetupError(
                            f"For a 2D simulation in the {plane} plane with "
                            f"'proj_axis = {monitor.proj_axis}', '{config['coord'][idx]}' of monitor "
                            f"'{monitor.name}' should be set to '[0]'."
                        )

        return val

    @pydantic.validator("monitors", always=True)
    @skip_if_fields_missing(["medium", "structures"])
    def diffraction_monitor_medium(cls, val, values):
        """If any :class:`.DiffractionMonitor` exists, ensure is does not lie in a lossy medium."""
        monitors = val
        structures = values.get("structures")
        medium = values.get("medium")
        for monitor in monitors:
            if isinstance(monitor, DiffractionMonitor):
                medium_set = Scene.intersecting_media(monitor, structures)
                medium = medium_set.pop() if medium_set else medium
                freqs = np.array(monitor.freqs)
                if isinstance(medium, AbstractCustomMedium) and len(freqs) > 1:
                    freqs = 0.5 * (np.min(freqs) + np.max(freqs))
                _, index_k = medium.nk_model(frequency=freqs)
                if not np.all(index_k == 0):
                    raise SetupError("Diffraction monitors must not lie in a lossy medium.")
        return val

    @pydantic.validator("grid_spec", always=True)
    @skip_if_fields_missing(["medium", "sources", "structures"])
    def _warn_grid_size_too_small(cls, val, values):
        """Warn user if any grid size is too large compared to minimum wavelength in material."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = structures or []
        medium_bg = values.get("medium")
        mediums = [medium_bg] + [structure.to_static().medium for structure in structures]

        with log as consolidated_logger:
            for source_index, source in enumerate(values.get("sources")):
                freq0 = source.source_time.freq0

                for medium_index, medium in enumerate(mediums):
                    # min wavelength in PEC is meaningless and we'll get divide by inf errors
                    if medium.is_pec:
                        continue
                    # min wavelength in Medium2D is meaningless
                    if isinstance(medium, Medium2D):
                        continue

                    eps_material = medium.eps_model(freq0)
                    n_material, _ = medium.eps_complex_to_nk(eps_material)

                    for comp, (key, grid_spec) in enumerate(
                        zip("xyz", (val.grid_x, val.grid_y, val.grid_z))
                    ):
                        if medium.is_pec or (
                            isinstance(medium, AnisotropicMedium) and medium.is_comp_pec(comp)
                        ):
                            n_material = 1.0
                        lambda_min = C_0 / freq0 / n_material

                        if (
                            isinstance(grid_spec, UniformGrid)
                            and grid_spec.dl > lambda_min / MIN_GRIDS_PER_WVL
                        ):
                            if medium_index == 0:
                                medium_str = "the simulation background medium"
                            else:
                                medium_str = (
                                    f"the medium associated with structures[{medium_index - 1}]"
                                )

                            consolidated_logger.warning(
                                f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                                ", which was detected as being large when compared to the "
                                f"central wavelength of sources[{source_index}] "
                                f"within {medium_str}, given by "
                                f"{lambda_min:.4f} (um). To avoid inaccuracies, "
                                "it is recommended the grid size is reduced. ",
                                custom_loc=["grid_spec", f"grid_{key}", "dl"],
                            )
                            # TODO: warn about custom grid spec

        return val

    @pydantic.validator("sources", always=True)
    @skip_if_fields_missing(["medium", "center", "size", "structures"])
    def _source_homogeneous_isotropic(cls, val, values):
        """Error if a plane wave or gaussian beam source is not in a homogeneous and isotropic
        region.
        """

        if val is None:
            return val

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=values.get("size"),
                center=values.get("center"),
            ),
            medium=values.get("medium"),
        )

        structures = values.get("structures") or []
        total_structures = [structure_bg] + list(structures)

        # for each plane wave in the sources list
        with log as consolidated_logger:
            for source_id, source in enumerate(val):
                if isinstance(source, (PlaneWave, GaussianBeam, AstigmaticGaussianBeam)):
                    mediums = Scene.intersecting_media(source, total_structures)
                    # make sure there is no more than one medium in the returned list
                    if len(mediums) > 1:
                        raise SetupError(
                            f"{len(mediums)} different mediums detected on plane "
                            f"intersecting a {source.type} source. Plane must be homogeneous."
                        )
                    # 0 medium, something is wrong
                    if len(mediums) < 1:
                        raise SetupError(
                            f"No medium detected on plane intersecting a {source.type}, "
                            "indicating an unexpected error. Please create a github issue so "
                            "that the problem can be investigated."
                        )
                    if isinstance(list(mediums)[0], (AnisotropicMedium, FullyAnisotropicMedium)):
                        raise SetupError(
                            f"An anisotropic medium is detected on plane intersecting a {source.type} "
                            f"source. Injection of {source.type} into anisotropic media currently is "
                            "not supported."
                        )

                    # check if the medium is spatially uniform
                    if not list(mediums)[0].is_spatially_uniform:
                        consolidated_logger.warning(
                            f"Nonuniform custom medium detected on plane intersecting a {source.type}. "
                            "Plane must be homogeneous. Make sure custom medium is uniform on the plane.",
                            custom_loc=["sources", source_id],
                        )

        return val

    @pydantic.validator("normalize_index", always=True)
    @skip_if_fields_missing(["sources"])
    def _check_normalize_index(cls, val, values):
        """Check validity of normalize index in context of simulation.sources."""

        # not normalizing
        if val is None:
            return val

        sources = values.get("sources")
        num_sources = len(sources)
        if num_sources > 0:
            # No check if no sources, but it should be irrelevant anyway
            if val >= num_sources:
                raise ValidationError(
                    f"'normalize_index' {val} out of bounds for number of sources {num_sources}."
                )

            # Also error if normalizing by a zero-amplitude source
            if sources[val].source_time.amplitude == 0:
                raise ValidationError("Cannot set 'normalize_index' to source with zero amplitude.")

            # Warn if normalizing by a ContinuousWave or CustomSourceTime source, if frequency-domain monitors are present.
            if isinstance(sources[val].source_time, ContinuousWave):
                log.warning(
                    f"'normalize_index' {val} is a source with 'ContinuousWave' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is not meaningful because field decay does not occur. "
                    "Consider setting 'normalize_index' to 'None' instead."
                )
            if isinstance(sources[val].source_time, CustomSourceTime):
                log.warning(
                    f"'normalize_index' {val} is a source with 'CustomSourceTime' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is only meaningful if field decay occurs."
                )

        return val

    """ Post-init validators """

    def _post_init_validators(self) -> None:
        """Call validators taking z`self` that get run after init."""
        _ = self.scene
        self._validate_no_structures_pml()
        self._validate_tfsf_nonuniform_grid()
        self._validate_nonlinear_specs()
        self._validate_custom_source_time()

    def _validate_custom_source_time(self):
        """Warn if all simulation times are outside CustomSourceTime definition range."""
        run_time = self._run_time
        for idx, source in enumerate(self.sources):
            if isinstance(source.source_time, CustomSourceTime):
                if source.source_time._all_outside_range(run_time=run_time):
                    data_times = source.source_time.data_times
                    mint = np.min(data_times)
                    maxt = np.max(data_times)
                    log.warning(
                        f"'CustomSourceTime' at 'sources[{idx}]' is defined over a time range "
                        f"'({mint}, {maxt})' which does not include any of the 'Simulation' "
                        f"times '({0, run_time})'. The envelope will be constant extrapolated "
                        "from the first or last value in the 'CustomSourceTime', which may not "
                        "be the desired outcome."
                    )

    def _validate_no_structures_pml(self) -> None:
        """Ensure no structures terminate / have bounds inside of PML."""

        pml_thicks = np.array(self.pml_thicknesses).T
        sim_bounds = self.bounds
        bound_spec = self.boundary_spec.to_list

        with log as consolidated_logger:
            for i, structure in enumerate(self.structures):
                geo_bounds = structure.geometry.bounds
                warn = False  # will only warn once per structure
                for sim_bound, geo_bound, pml_thick, bound_dim, pm_val in zip(
                    sim_bounds, geo_bounds, pml_thicks, bound_spec, (-1, 1)
                ):
                    for sim_pos, geo_pos, pml, bound_edge in zip(
                        sim_bound, geo_bound, pml_thick, bound_dim
                    ):
                        sim_pos_pml = sim_pos + pm_val * pml
                        in_pml_plus = (pm_val > 0) and (sim_pos < geo_pos <= sim_pos_pml)
                        in_pml_mnus = (pm_val < 0) and (sim_pos > geo_pos >= sim_pos_pml)
                        if not isinstance(bound_edge, Absorber) and (in_pml_plus or in_pml_mnus):
                            warn = True
                if warn:
                    consolidated_logger.warning(
                        f"A bound of Simulation.structures[{i}] was detected as being "
                        "within the simulation PML. We recommend extending structures to "
                        "infinity or completely outside of the simulation PML to avoid "
                        "unexpected effects when the structures are not translationally "
                        "invariant within the PML.",
                        custom_loc=["structures", i],
                    )

    def _validate_tfsf_nonuniform_grid(self) -> None:
        """Warn if the grid is nonuniform along the directions tangential to the injection plane,
        inside the TFSF box.
        """
        # if the grid is uniform in all directions, there's no need to proceed
        if not (self.grid_spec.auto_grid_used or self.grid_spec.custom_grid_used):
            return

        with log as consolidated_logger:
            for source_ind, source in enumerate(self.sources):
                if not isinstance(source, TFSF):
                    continue

                centers = self.grid.centers.to_list
                sizes = self.grid.sizes.to_list
                tfsf_bounds = source.bounds
                _, plane_inds = source.pop_axis([0, 1, 2], axis=source.injection_axis)
                grid_list = [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
                for ind in plane_inds:
                    grid_type = grid_list[ind]
                    if isinstance(grid_type, UniformGrid):
                        continue

                    sizes_in_tfsf = [
                        size
                        for size, center in zip(sizes[ind], centers[ind])
                        if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
                    ]

                    # check if all the grid sizes are sufficiently unequal
                    if not np.all(np.isclose(sizes_in_tfsf, sizes_in_tfsf[0])):
                        consolidated_logger.warning(
                            f"The grid is nonuniform along the '{'xyz'[ind]}' axis, which may lead "
                            "to sub-optimal cancellation of the incident field in the "
                            "scattered-field region for the total-field scattered-field (TFSF) "
                            f"source '{source.name}'. For best results, we recommended ensuring a "
                            "uniform grid in both directions tangential to the TFSF injection "
                            f"axis, '{'xyz'[source.injection_axis]}'.",
                            custom_loc=["sources", source_ind],
                        )

    def _validate_nonlinear_specs(self) -> None:
        """Run :class:`.NonlinearSpec` validators that depend on knowing the central
        frequencies of the sources."""
        freqs = np.array([source.source_time.freq0 for source in self.sources])
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                for model in medium._nonlinear_models:
                    model._validate_medium_freqs(medium, freqs)

    """ Pre submit validation (before web.upload()) """

    def validate_pre_upload(self, source_required: bool = True) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers.

        Parameters
        ----------
        source_required: bool = True
            If ``True``, validation will fail in case no sources are found in the simulation.
        """
        log.begin_capture()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._validate_num_cells_in_mode_objects()
        self._validate_datasets_not_none()
        self._validate_tfsf_structure_intersections()
        self._warn_time_monitors_outside_run_time()
        self._validate_time_monitors_num_steps()
        _ = self.volumetric_structures
        log.end_capture(self)
        if source_required and len(self.sources) == 0:
            raise SetupError("No sources in simulation.")

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""

        num_comp_cells = self.num_cells / 2 ** (np.sum(np.abs(self.symmetry)))
        if num_comp_cells > MAX_GRID_CELLS:
            raise SetupError(
                f"Simulation has {num_comp_cells:.2e} computational cells, "
                f"a maximum of {MAX_GRID_CELLS:.2e} are allowed."
            )

        num_time_steps = self.num_time_steps
        if num_time_steps > MAX_TIME_STEPS:
            raise SetupError(
                f"Simulation has {num_time_steps:.2e} time steps, "
                f"a maximum of {MAX_TIME_STEPS:.2e} are allowed."
            )
        if num_time_steps > WARN_TIME_STEPS:
            log.warning(
                f"Simulation has {num_time_steps:.2e} time steps. The 'run_time' may be "
                "unnecessarily large, unless there are very long-lived resonances.",
                custom_loc=["run_time"],
            )

        num_cells_times_steps = num_time_steps * num_comp_cells
        if num_cells_times_steps > MAX_CELLS_TIMES_STEPS:
            raise SetupError(
                f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
                f"a maximum of {MAX_CELLS_TIMES_STEPS:.2e} are allowed."
            )

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors aren't storing too much data before simulation is uploaded."""

        total_size_gb = 0
        with log as consolidated_logger:
            datas = self.monitors_data_size
            for monitor_ind, (monitor_name, monitor_size) in enumerate(datas.items()):
                monitor_size_gb = monitor_size / 1e9
                if monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Monitor '{monitor_name}' estimated storage is {monitor_size_gb:1.2f}GB. "
                        "Consider making it smaller, using fewer frequencies, or spatial or "
                        "temporal downsampling using 'interval_space' and 'interval', respectively.",
                        custom_loc=["monitors", monitor_ind],
                    )

                total_size_gb += monitor_size_gb

        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            raise SetupError(
                f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed."
            )

        # Some monitors store much less data than what is needed internally. Make sure that the
        # internal storage also does not exceed the limit.
        for monitor in self.monitors:
            num_cells = self._monitor_num_cells(monitor)
            # intermediate storage needed, in GB
            solver_data = monitor._storage_size_solver(num_cells=num_cells, tmesh=self.tmesh) / 1e9
            if solver_data > MAX_MONITOR_INTERNAL_DATA_SIZE_GB:
                raise SetupError(
                    f"Estimated internal storage of monitor '{monitor.name}' is "
                    f"{solver_data:1.2f}GB, which is larger than the maximum allowed "
                    f"{MAX_MONITOR_INTERNAL_DATA_SIZE_GB:.2f}GB. Consider making it smaller, "
                    "using fewer frequencies, or spatial or temporal downsampling using "
                    "'interval_space' and 'interval', respectively."
                )

    def _validate_modes_size(self) -> None:
        """Warn if mode sources or monitors have a large number of points."""

        def warn_mode_size(monitor: AbstractModeMonitor, msg_header: str, custom_loc: List):
            """Warn if a mode component has a large number of points."""
            num_cells = np.prod(self.discretize_monitor(monitor).num_cells)
            if num_cells > WARN_MODE_NUM_CELLS:
                consolidated_logger.warning(
                    msg_header + f"has a large number ({num_cells:1.2e}) of grid points. "
                    "This can lead to solver slow-down and increased cost. "
                    "Consider making the size of the component smaller, as long as the modes "
                    "of interest decay by the plane boundaries.",
                    custom_loc=custom_loc,
                )

        with log as consolidated_logger:
            for src_ind, source in enumerate(self.sources):
                if isinstance(source, ModeSource):
                    # Make a monitor so we can call ``discretize_monitor``
                    monitor = FieldMonitor(
                        center=source.center,
                        size=source.size,
                        name="tmp",
                        freqs=[source.source_time.freq0],
                        colocate=False,
                    )
                    msg_header = f"Mode source at sources[{src_ind}] "
                    custom_loc = ["sources", src_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

        with log as consolidated_logger:
            for mnt_ind, monitor in enumerate(self.monitors):
                if isinstance(monitor, AbstractModeMonitor):
                    msg_header = f"Mode monitor '{monitor.name}' "
                    custom_loc = ["monitors", mnt_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    def _validate_num_cells_in_mode_objects(self) -> None:
        """Raise an error if mode sources or monitors intersect with a very small number
        of grid cells in their transverse dimensions."""

        def check_num_cells(
            mode_object: Tuple[ModeSource, ModeMonitor], normal_axis: Axis, msg_header: str
        ):
            disc_grid = self.discretize(mode_object)
            _, check_axes = Box.pop_axis([0, 1, 2], axis=normal_axis)
            for axis in check_axes:
                sim_size = self.size[axis]
                dim_cells = disc_grid.num_cells[axis]
                if sim_size > 0 and dim_cells <= 2:
                    small_dim = "xyz"[axis]
                    raise SetupError(
                        msg_header + f"is too small along the "
                        f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                        f"Increase the size of the object along '{small_dim}'."
                    )

        for source in self.sources:
            if isinstance(source, ModeSource):
                msg_header = f"Mode source '{source.name}' "
                check_num_cells(source, source.injection_axis, msg_header)

        for monitor in self.monitors:
            if isinstance(monitor, ModeMonitor):
                msg_header = f"Mode monitor '{monitor.name}' "
                check_num_cells(monitor, monitor.normal_axis, msg_header)

    def _validate_time_monitors_num_steps(self) -> None:
        """Raise an error if non-0D time monitors have too many time steps."""
        for monitor in self.monitors:
            if not isinstance(monitor, FieldTimeMonitor) or len(monitor.zero_dims) == 3:
                continue
            num_time_steps = monitor.num_steps(self.tmesh)
            if num_time_steps > MAX_TIME_MONITOR_STEPS:
                raise SetupError(
                    f"Time monitor '{monitor.name}' records at {num_time_steps} time steps, which "
                    f"is larger than the maximum allowed value of {MAX_TIME_MONITOR_STEPS} when "
                    "the monitor is not zero-dimensional. Change the geometry to a point monitor, "
                    "or use 'start', 'stop', and 'interval' to reduce the number of time steps "
                    "at which the monitor stores data."
                )

    @cached_property
    def static_structures(self) -> list[Structure]:
        """Structures in simulation with all autograd tracers removed."""
        return [structure.to_static() for structure in self.structures]

    @cached_property
    def monitors_data_size(self) -> Dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self.monitors:
            num_cells = self._monitor_num_cells(monitor)
            storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=self.tmesh))
            data_size[monitor.name] = storage_size
        return data_size

    def _validate_datasets_not_none(self) -> None:
        """Ensures that all custom datasets are defined."""
        if any(dataset is None for dataset in self.custom_datasets):
            raise SetupError(
                "Data for a custom data component is missing. This can happen for example if the "
                "Simulation has been loaded from json. To save and load simulations with custom "
                "data, use hdf5 format instead."
            )

    def _validate_tfsf_structure_intersections(self) -> None:
        """Error if the 4 sidewalls of a TFSF box don't all intersect the same structures.
        This validator may need to compute permittivities on the grid, so it is called
        pre-upload rather than at the time of definition. Also errors if any side wall
        intersects with a custom medium or a fully anisotropic media.
        """
        for source in self.sources:
            if not isinstance(source, TFSF):
                continue
            # get all TFSF surfaces
            tfsf_surfaces = Source.surfaces(
                center=source.center, size=source.size, source_time=source.source_time
            )
            sidewall_surfaces = []
            sidewall_structs = []
            # get the structures that intersect each sidewall
            for surface in tfsf_surfaces:
                # ignore the sidewall surface if it falls outside the simulation domain
                if not self.intersects(surface):
                    continue

                if surface.name[-2] != "xyz"[source.injection_axis]:
                    sidewall_surfaces.append(surface)
                    intersecting_structs = Scene.intersecting_structures(
                        test_object=surface, structures=self.structures
                    )

                    if any(
                        isinstance(struct.medium, (AbstractCustomMedium, FullyAnisotropicMedium))
                        for struct in intersecting_structs
                    ):
                        raise SetupError(
                            f"The surfaces of TFSF source '{source.name}' must not intersect any "
                            "structures containing a 'CustomMedium' or a 'FullyAnisotropicMedium'."
                        )

                    # if no structures intersect, just add a phantom associated with the simulation
                    # background, to prevent false positives below
                    if not intersecting_structs:
                        sidewall_structs.append(
                            [
                                Structure(
                                    geometry=Box(center=self.center, size=self.size),
                                    medium=self.medium,
                                )
                            ]
                        )
                    else:
                        sidewall_structs.append(intersecting_structs)

            # let the first wall be a reference, and compare the rest of them to the structures
            # intersected by that reference wall
            if len(sidewall_structs) > 1:
                ref_structs = sidewall_structs[0]
                test_structs = sidewall_structs[1:]
                if all(structs == ref_structs for structs in test_structs):
                    continue

                # if the == test doesn't pass, that doesn't mean the materials are necessarily
                # different, because it's possible that the sidewalls encounter different
                # `Structure` objects but with an identical material profile, which is still
                # a valid setup; in this case, compute the epsilon profile on the grid for each
                # side wall - the profiles must be the same along the injection axis, so we take
                # a single "stripe" of epsilon as the reference and subtract it from all other
                # stripes, which should result in zero if all the epsilon profiles are the same
                freq0 = source.source_time.freq0
                _, plane_axs = source.pop_axis("xyz", axis=source.injection_axis)
                ref_eps = self.epsilon(box=sidewall_surfaces[0], coord_key="centers", freq=freq0)
                kwargs = {plane_axs[0]: 0, plane_axs[1]: 0}
                ref_eps = ref_eps.isel(**kwargs)
                for surface in sidewall_surfaces:
                    test_eps = self.epsilon(box=surface, coord_key="centers", freq=freq0) - ref_eps
                    if not np.allclose(test_eps.to_numpy(), 0):
                        raise SetupError(
                            f"All sidewalls of the TFSF source '{source.name}' must intersect "
                            "the same media along the injection axis "
                            f" '{'xyz'[source.injection_axis]}'."
                        )

    def _warn_time_monitors_outside_run_time(self) -> None:
        """Warn if time monitors start after the simulation run_time.
        TODO: (remove this comment later) this is done as a pre-upload validator in view of a
        planned change to allow ``run_time`` to accept a ``RunTimeSpec`` which would automatically
        determine a run time based on simulation details. Then, we would have to access the
        dynamically computed run_time e.g. through a ``_run_time`` cached property.
        """
        with log as consolidated_logger:
            for monitor in self.monitors:
                if isinstance(monitor, TimeMonitor) and monitor.start > self._run_time:
                    consolidated_logger.warning(
                        f"Monitor {monitor.name} has a start time {monitor.start:1.2e}s exceeding"
                        f"the simulation run time {self._run_time:1.2e}s. No data will be recorded."
                    )

    """ Autograd adjoint support """

    def with_adjoint_monitors(self, sim_fields_keys: list) -> Simulation:
        """Copy of self with adjoint field and permittivity monitors for every traced structure."""

        # set of indices in the structures needing adjoint monitors
        structure_indices = {index for (_, index, *_) in sim_fields_keys}

        mnts_fld, mnts_eps = self.make_adjoint_monitors(structure_indices=structure_indices)
        monitors = list(self.monitors) + list(mnts_fld) + list(mnts_eps)
        return self.copy(update=dict(monitors=monitors))

    def make_adjoint_monitors(self, structure_indices: set[int]) -> tuple[list, list]:
        """Get lists of field and permittivity monitors for this simulation."""

        freqs = self.freqs_adjoint

        adjoint_monitors_fld = []
        adjoint_monitors_eps = []

        # make a field and permittivity monitor for every structure needing one
        for i in structure_indices:
            structure = self.structures[i]

            mnt_fld, mnt_eps = structure.make_adjoint_monitors(freqs=freqs, index=i)

            adjoint_monitors_fld.append(mnt_fld)
            adjoint_monitors_eps.append(mnt_eps)

        return adjoint_monitors_fld, adjoint_monitors_eps

    @property
    def freqs_adjoint(self) -> list[float]:
        """Unique list of all frequencies. For now should be only one."""

        freqs = set()
        for mnt in self.monitors:
            if isinstance(mnt, FreqMonitor):
                freqs.update(mnt.freqs)
        freqs = sorted(freqs)
        return freqs

    """ Accounting """

    @cached_property
    def _run_time(self) -> float:
        """Run time evaluated based on self.run_time."""

        if not isinstance(self.run_time, RunTimeSpec):
            return self.run_time

        run_time_spec = self.run_time

        # contribution from the time of of the source pulses
        if not self.sources:
            source_time = 0.0
        else:
            source_times = [src.source_time.end_time() for src in self.sources]
            source_times = [x for x in source_times if x is not None]
            if not source_times:
                raise SetupError(
                    "Could not compute source contributions to run time from 'RunTimeSpec'."
                    "Please check all of your 'Source.source_time' and ensure that at least one "
                    "has a decaying (non-DC) pulse profile to be able to compute the 'run_time'."
                )
            source_time_max = np.max(source_times)
            source_time = run_time_spec.source_factor * source_time_max

        # contribution from field decay out of the simulation
        propagation_lengths = np.array(self.bounds[1]) - np.array(self.bounds[0])
        max_propagation_length = np.max(propagation_lengths)

        # get the maximum refractive index evaluated over each of all the source central frequencies
        all_ref_inds = [self.get_refractive_indices(src.source_time.freq0) for src in self.sources]
        avg_ref_inds = [np.mean(np.array(n)) for n in all_ref_inds]
        max_ref_ind = np.max(avg_ref_inds)

        propagation_time = run_time_spec.quality_factor * max_ref_ind * max_propagation_length / C_0

        return source_time + propagation_time

    # candidate for removal in 3.0
    @cached_property
    def mediums(self) -> Set[MediumType]:
        """Returns set of distinct :class:`.AbstractMedium` in simulation.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums in the simulation.
        """
        log.warning(
            "'Simulation.mediums' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.mediums' instead."
        )
        return self.scene.mediums

    # candidate for removal in 3.0
    @cached_property
    def medium_map(self) -> Dict[MediumType, pydantic.NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`.AbstractMedium`
        in simulation.

        Returns
        -------
        Dict[:class:`.AbstractMedium`, int]
            Mapping between distinct mediums to index in simulation.
        """

        log.warning(
            "'Simulation.medium_map' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.medium_map' instead."
        )
        return self.scene.medium_map

    # candidate for removal in 3.0
    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.Simulation`."""

        log.warning(
            "'Simulation.background_structure' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.background_structure' instead."
        )
        return self.scene.background_structure

    # candidate for removal in 3.0
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

        log.warning(
            "'Simulation.intersecting_media()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_media()' instead."
        )
        return Scene.intersecting_media(test_object=test_object, structures=structures)

    # candidate for removal in 3.0
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

        log.warning(
            "'Simulation.intersecting_structures()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_structures()' instead."
        )
        return Scene.intersecting_structures(test_object=test_object, structures=structures)

    def monitor_medium(self, monitor: MonitorType):
        """Return the medium in which the given monitor resides.

        Parameters
        -------
        monitor : :class:`.Monitor`
            Monitor whose associated medium is to be returned.

        Returns
        -------
        :class:`.AbstractMedium`
            Medium associated with the given :class:`.Monitor`.
        """
        medium_set = Scene.intersecting_media(monitor, self.structures)
        if len(medium_set) > 1:
            raise SetupError(f"Monitor '{monitor.name}' intersects more than one medium.")
        medium = medium_set.pop() if medium_set else self.medium
        return medium

    @staticmethod
    def _check_bloch_vec(
        source: SourceType,
        source_ind: int,
        bloch_vec: float,
        dim: Axis,
        medium: MediumType,
        domain_size: float,
        has_diff_mnt: bool = False,
    ):
        """Helper to check if a given Bloch vector is consistent with a given source."""

        # make a dummy Bloch boundary to check for correctness
        dummy_bnd = BlochBoundary.from_source(
            source=source, domain_size=domain_size, axis=dim, medium=medium
        )
        expected_bloch_vec = dummy_bnd.bloch_vec

        if bloch_vec != expected_bloch_vec:
            test_val = np.real(expected_bloch_vec - bloch_vec)

            test_val_is_int = np.isclose(test_val, np.round(test_val))
            src_name = f" '{source.name}'" if source.name else ""

            if has_diff_mnt and test_val_is_int and not np.isclose(test_val, 0):
                # the given Bloch vector is offset by an integer
                log.warning(
                    f"The wave vector of source{src_name} along dimension "
                    f"'{dim}' is equal to the Bloch vector of the simulation "
                    "boundaries along that dimension plus an integer reciprocal "
                    "lattice vector. If using a 'DiffractionMonitor', diffraction "
                    "order 0 will not correspond to the angle of propagation "
                    "of the source. Consider using 'BlochBoundary.from_source()'.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )

            if not test_val_is_int:
                # the given Bloch vector is neither equal to the expected value, nor
                # off by an integer
                log.warning(
                    f"The Bloch vector along dimension '{dim}' may be incorrectly "
                    f"set with respect to the source{src_name}. The absolute "
                    "difference between the expected and provided values in "
                    "bandstructure units, up to an integer offset, is greater than "
                    "1e-6. Consider using ``BlochBoundary.from_source()``, or "
                    "double-check that it was defined correctly.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )

    def to_gdstk(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.

        Return
        ------
        List
            List of `gdstk.Polygon`.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, bmin = self.pop_axis(self.bounds[0], axis)
        _, bmax = self.pop_axis(self.bounds[1], axis)

        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            bmin = (0, bmin[1])
        if symmetry[1] != 0:
            bmin = (bmin[0], 0)
        clip = gdstk.rectangle(bmin, bmax)

        polygons = []
        for structure in self.structures:
            gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
            for polygon in structure.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
            ):
                pmin, pmax = polygon.bounding_box()
                if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                    polygons.extend(
                        gdstk.boolean(clip, polygon, "and", layer=gds_layer, datatype=gds_dtype)
                    )
                else:
                    polygons.append(polygon)

        return polygons

    def to_gdspy(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.

        Return
        ------
        List
            List of `gdspy.Polygon` and `gdspy.PolygonSet`.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, bmin = self.pop_axis(self.bounds[0], axis)
        _, bmax = self.pop_axis(self.bounds[1], axis)

        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            bmin = (0, bmin[1])
        if symmetry[1] != 0:
            bmin = (bmin[0], 0)
        clip = gdspy.Rectangle(bmin, bmax)

        polygons = []
        for structure in self.structures:
            gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
            for polygon in structure.to_gdspy(
                x=x,
                y=y,
                z=z,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
            ):
                pmin, pmax = polygon.get_bounding_box()
                if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                    polygon = gdspy.boolean(
                        clip, polygon, "and", layer=gds_layer, datatype=gds_dtype
                    )
                polygons.append(polygon)
        return polygons

    def to_gds(
        self,
        cell,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> None:
        """Append the simulation structures to a .gds cell.

        Parameters
        ----------
        cell : ``gdstk.Cell`` or ``gdspy.Cell``
            Cell object to which the generated polygons are added.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        if gdstk_available and isinstance(cell, gdstk.Cell):
            polygons = self.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer_dtype_map=gds_layer_dtype_map,
            )
            if len(polygons) > 0:
                cell.add(*polygons)

        elif gdspy_available and isinstance(cell, gdspy.Cell):
            polygons = self.to_gdspy(x=x, y=y, z=z, gds_layer_dtype_map=gds_layer_dtype_map)
            if len(polygons) > 0:
                cell.add(polygons)

        elif "gdstk" in cell.__class__ and not gdstk_available:
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
            )
        elif "gdspy" in cell.__class__ and not gdspy_available:
            raise Tidy3dImportError(
                "Module 'gdspy' not found. It is required to export shapes to gdspy cells."
            )
        else:
            raise Tidy3dError(
                "Argument 'cell' must be an instance of 'gdstk.Cell' or 'gdspy.Cell'."
            )

    def to_gds_file(
        self,
        fname: str,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
        gds_cell_name: str = "MAIN",
    ) -> None:
        """Append the simulation structures to a .gds cell.

        Parameters
        ----------
        fname : str
            Full path to the .gds file to save the :class:`Simulation` slice to.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1
            Permittivity value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluation in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        gds_cell_name : str = 'MAIN'
            Name of the cell created in the .gds file to store the geometry.
        """
        if gdstk_available:
            library = gdstk.Library()
            reference = gdstk.Reference
            rotation = np.pi
        elif gdspy_available:
            library = gdspy.GdsLibrary()
            reference = gdspy.CellReference
            rotation = 180
        else:
            raise Tidy3dImportError(
                "Python modules 'gdspy' and 'gdstk' not found. To export geometries to .gds "
                "files, please install one of those those modules."
            )
        cell = library.new_cell(gds_cell_name)

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_X")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, rotation=rotation, x_reflection=True))
        if symmetry[1] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_Y")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, x_reflection=True))

        self.to_gds(
            cell,
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
        )
        pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
        library.write_gds(fname)

    """ Plotting """

    @cached_property
    def frequency_range(self) -> FreqBound:
        """Range of frequencies spanning all sources' frequency dependence.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum frequencies of the power spectrum of the sources.
        """
        source_ranges = [source.source_time.frequency_range() for source in self.sources]
        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        return (freq_min, freq_max)

    def plot_3d(self, width=800, height=800) -> None:
        """Render 3D plot of ``Simulation`` (in jupyter notebook only).
        Parameters
        ----------
        width : float = 800
            width of the 3d view dom's size
        height : float = 800
            height of the 3d view dom's size

        """
        return plot_sim_3d(self, width=width, height=height)

    """ Discretization """

    @cached_property
    def scaled_courant(self) -> float:
        """When conformal mesh is applied, courant number is scaled down depending on `conformal_mesh_spec`."""

        mediums = self.scene.mediums
        contain_pec_structures = any(medium.is_pec for medium in mediums)
        return self.courant * self._subpixel.courant_ratio(
            contain_pec_structures=contain_pec_structures
        )

    @cached_property
    def dt(self) -> float:
        """Simulation time step (distance).

        Returns
        -------
        float
            Time step (seconds).
        """
        dl_mins = [
            np.min(sizes)
            for dim, sizes in enumerate(self.grid.sizes.to_list)
            if self.grid.num_cells[dim] > 1
        ]
        dl_sum_inv_sq = sum(1 / dl**2 for dl in dl_mins)
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        # material factor
        n_cfl = min(min(mat.n_cfl for mat in self.scene.mediums), 1)
        return n_cfl * self.scaled_courant * dl_avg / C_0

    @cached_property
    def tmesh(self) -> Coords1D:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self._run_time + dt, dt)

    @cached_property
    def num_time_steps(self) -> int:
        """Number of time steps in simulation."""

        return len(self.tmesh)

    @cached_property
    def self_structure(self) -> Structure:
        """The simulation background as a ``Structure``."""
        geometry = Box(size=(inf, inf, inf), center=self.center)
        return Structure(geometry=geometry, medium=self.medium)

    @cached_property
    def all_structures(self) -> List[Structure]:
        """List of all structures in the simulation (including the ``Simulation.medium``)."""
        return [self.self_structure] + list(self.structures)

    def _grid_corrections_2dmaterials(self, grid: Grid) -> Grid:
        """Correct the grid if 2d materials are present, using their volumetric equivalents."""
        if not any(isinstance(structure.medium, Medium2D) for structure in self.structures):
            return grid

        # when there are 2D materials, need to make grid again with volumetric_structures
        # generated using the first grid

        volumetric_structures = [Structure(geometry=self.geometry, medium=self.medium)]
        volumetric_structures += self._volumetric_structures_grid(grid)

        volumetric_grid = self.grid_spec.make_grid(
            structures=volumetric_structures,
            symmetry=self.symmetry,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # Handle 2D materials if ``AutoGrid`` is used for in-plane directions
        # must use original grid for the normal directions of all 2d materials
        grid_axes = [False, False, False]
        # must use volumetric grid for the ``AutoGrid`` in-plane directions of 2d materials
        volumetric_grid_axes = [False, False, False]
        with log as consolidated_logger:
            for structure in self.structures:
                if isinstance(structure.medium, Medium2D):
                    normal = structure.geometry._normal_2dmaterial
                    grid_axes[normal] = True
                    for axis, grid_axis in enumerate(
                        [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
                    ):
                        if isinstance(grid_axis, AutoGrid):
                            if axis != normal:
                                volumetric_grid_axes[axis] = True
                            else:
                                consolidated_logger.warning(
                                    "Using 'AutoGrid' for the normal direction of a 2D material "
                                    "may generate a grid that is not sufficiently fine."
                                )
        coords_all = [None, None, None]
        for axis in range(3):
            if grid_axes[axis] and volumetric_grid_axes[axis]:
                raise ValidationError(
                    "Unable to generate grid. Cannot use 'AutoGrid' for "
                    "an axis that is in-plane to one 2D material and normal to another."
                )
            if volumetric_grid_axes[axis]:
                coords_all[axis] = volumetric_grid.boundaries.to_list[axis]
            else:
                coords_all[axis] = grid.boundaries.to_list[axis]

        return Grid(boundaries=Coords(**dict(zip("xyz", coords_all))))

    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]

        structures += self.static_structures

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # This would AutoGrid the in-plane directions of the 2D materials
        # return self._grid_corrections_2dmaterials(grid)
        return grid

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation grid.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

    @property
    def _num_computational_grid_points_dim(self):
        """Number of cells in the computational domain for this simulation along each dimension."""
        num_cells = self.grid.num_cells
        num_cells_comp_domain = []
        # symmetry overrides other boundaries so should be checked first
        for sym, npts, boundary in zip(self.symmetry, num_cells, self.boundary_spec.to_list):
            if sym != 0:
                num_cells_comp_domain.append(npts // 2 + 2)
            elif isinstance(boundary[0], Periodic):
                num_cells_comp_domain.append(npts)
            else:
                num_cells_comp_domain.append(npts + 2)
        return num_cells_comp_domain

    @property
    def num_computational_grid_points(self):
        """Number of cells in the computational domain for this simulation. This is usually
        different from ``num_cells`` due to the boundary conditions. Specifically, all boundary
        conditions apart from ``Periodic`` require an extra pixel at the end of the simulation
        domain. On the other hand, if a symmetry is present along a given dimension, only half of
        the grid cells along that dimension will be in the computational domain.

        Returns
        -------
        int
            Number of yee cells in the computational domain corresponding to the simulation.
        """
        return np.prod(self._num_computational_grid_points_dim, dtype=np.int64)

    def get_refractive_indices(self, freq: float) -> list[float]:
        """List of refractive indices in the simulation at a given frequency."""

        eps_values = [structure.medium.eps_model(freq) for structure in self.static_structures]
        eps_values.append(self.medium.eps_model(freq))

        return [AbstractMedium.eps_complex_to_nk(eps)[0] for eps in eps_values]

    @cached_property
    def n_max(self) -> float:
        """Maximum refractive index in the ``Simulation``."""
        eps_max = max(abs(struct.medium.eps_model(self.freq_max)) for struct in self.all_structures)
        n_max, _ = AbstractMedium.eps_complex_to_nk(eps_max)
        return n_max

    @cached_property
    def wvl_mat_min(self) -> float:
        """Minimum wavelength in the materials present throughout the simulation.

        Returns
        -------
        float
            Minimum wavelength in the material (microns).
        """
        freq_max = max(source.source_time.freq0 for source in self.sources)
        wvl_min = C_0 / freq_max

        n_values = self.get_refractive_indices(freq_max)
        n_max = max(n_values)
        return wvl_min / n_max

    @cached_property
    def complex_fields(self) -> bool:
        """Whether complex fields are used in the simulation. Currently this only happens when there
        are Bloch boundaries.

        Returns
        -------
        bool
            Whether the time-stepping fields are real or complex.
        """
        if any(isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list):
            return True
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                if any(model.complex_fields for model in medium._nonlinear_models):
                    return True
        return False

    @cached_property
    def nyquist_step(self) -> int:
        """Maximum number of discrete time steps to keep sampling below Nyquist limit.

        Returns
        -------
        int
            The largest ``N`` such that ``N * self.dt`` is below the Nyquist limit.
        """

        # source frequency upper bound
        freq_source_max = self.frequency_range[1]
        # monitor frequency upper bound
        freq_monitor_max = max(
            (
                monitor.frequency_range[1]
                for monitor in self.monitors
                if isinstance(monitor, FreqMonitor) and not isinstance(monitor, PermittivityMonitor)
            ),
            default=0.0,
        )
        # combined frequency upper bound
        freq_max = max(freq_source_max, freq_monitor_max)

        if freq_max > 0:
            nyquist_step = int(1 / (2 * freq_max) / self.dt) - 1
            nyquist_step = max(1, nyquist_step)
        else:
            nyquist_step = 1

        return nyquist_step

    @property
    def custom_datasets(self) -> List[Dataset]:
        """List of custom datasets for verification purposes. If the list is not empty, then
        the simulation needs to be exported to hdf5 to store the data.
        """
        datasets_source_time = [
            src.source_time.source_time_dataset
            for src in self.sources
            if isinstance(src.source_time, CustomSourceTime)
        ]
        datasets_field_source = [
            src.field_dataset for src in self.sources if isinstance(src, CustomFieldSource)
        ]
        datasets_current_source = [
            src.current_dataset for src in self.sources if isinstance(src, CustomCurrentSource)
        ]
        datasets_medium = [
            mat
            for mat in self.scene.mediums
            if isinstance(mat, AbstractCustomMedium) or mat.is_time_modulated
        ]
        datasets_geometry = []

        for struct in self.structures:
            for geometry in traverse_geometries(struct.geometry):
                if isinstance(geometry, TriangleMesh):
                    datasets_geometry += [geometry.mesh_dataset]

        return (
            datasets_source_time
            + datasets_field_source
            + datasets_current_source
            + datasets_medium
            + datasets_geometry
        )

    @cached_property
    def allow_gain(self) -> bool:
        """``True`` if any of the mediums in the simulation allows gain."""

        for medium in self.scene.mediums:
            if isinstance(medium, AnisotropicMedium):
                if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                    return True
            elif medium.allow_gain:
                return True
        return False

    def perturbed_mediums_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Simulation:
        """Return a copy of the simulation with heat and/or charge data applied to all mediums
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
        Simulation
            Simulation after application of heat and/or charge data.
        """

        sim_dict = self.dict()
        structures = self.structures
        sim_bounds = self.simulation_bounds
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
                s_bounds = np.array(structure.geometry.bounds)

                bounds = [
                    np.max([sim_bounds[0], s_bounds[0]], axis=0),
                    np.min([sim_bounds[1], s_bounds[1]], axis=0),
                ]

                # skip structure if it's completely outside of sim box
                if any(bmin > bmax for bmin, bmax in zip(*bounds)):
                    new_structures.append(structure)
                else:
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

                    new_medium = med.perturbed_copy(
                        **restricted_arrays, interp_method=interp_method
                    )
                    new_structure = structure.updated_copy(medium=new_medium)
                    new_structures.append(new_structure)
            else:
                new_structures.append(structure)

        sim_dict["structures"] = new_structures

        # do the same for background medium if it a medium with perturbation models.
        med = self.medium
        if isinstance(med, AbstractPerturbationMedium):
            # get simulation's bounding box
            bounds = sim_bounds

            # for each structure select a minimal subset of data that covers it
            restricted_arrays = {}

            for name, array in array_dict.items():
                if array is not None:
                    restricted_arrays[name] = array.sel_inside(bounds)

                    # check provided data fully cover simulation
                    if not array.does_cover(bounds):
                        log.warning(f"Provided '{name}' does not fully cover simulation domain.")

            sim_dict["medium"] = med.perturbed_copy(
                **restricted_arrays, interp_method=interp_method
            )

        return Simulation.parse_obj(sim_dict)

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> Simulation:
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
        >>> from tidy3d import Scene, Medium, Box, Structure, GridSpec
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
        return Simulation(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )
