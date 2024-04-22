"""Defines EME simulation class."""
from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Literal

import pydantic.v1 as pd
import matplotlib as mpl
import numpy as np

from ..types import Ax
from ..simulation import AbstractYeeGridSimulation
from ..viz import add_ax_if_none, equal_aspect
from ..boundary import BoundarySpec, PECBoundary
from ..grid.grid_spec import GridSpec
from ..grid.grid import Grid
from ..types import Axis, annotate_type
from ..base import cached_property
from ..scene import Scene
from ..medium import FullyAnisotropicMedium
from ..structure import Structure
from ..source import ModeSource, GaussianPulse
from ..monitor import ModeSolverMonitor, AbstractModeMonitor, FieldMonitor, Monitor
from ...exceptions import SetupError
from ...log import log

from .monitor import EMEMonitor, EMEMonitorType, EMEFieldMonitor, EMEModeSolverMonitor
from .grid import EMEGridSpecType, EMEGrid, EMEGridSpec, EMECompositeGrid
from .sweep import EMESweepSpecType, EMEModeSweep, EMELengthSweep


# maximum numbers of simulation parameters
MAX_GRID_CELLS = 20e9
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5


# eme specific simulation parameters
MAX_NUM_FREQS = 20
MAX_NUM_SWEEP = 100


class EMESimulation(AbstractYeeGridSimulation):
    """EigenMode Expansion (EME) simulation.

    Notes
    -----

        EME is a frequency-domain method for propagating the electromagnetic field along a
        specified axis. The method is well-suited for propagation of guided waves.
        The electromagnetic fields are expanded locally in the basis of eigenmodes of the
        waveguide; they are then propagated by imposing continuity conditions in this basis.

        The EME simulation is performed along the propagation axis ``axis`` at frequencies ``freqs``.
        The simulation is divided into cells along the propagation axis, as defined by
        ``eme_grid_spec``. Mode solving is performed at cell centers, and boundary conditions are
        imposed between cells. The EME ports are defined to be the boundaries of the first and last
        cell in the EME grid. These can be moved using ``port_offsets``.

        An EME simulation always computes the full scattering matrix of the structure.
        Additional data can be recorded by adding 'monitors' to the simulation.

        **Other Bases**

        By default, the scattering matrix is expressed in the basis of EME modes at the two ports. It is sometimes useful to use alternative bases. For example, in a waveguide splitter, we might want the scattering matrix in the basis of modes of the individual waveguides. The functions `smatrix_in_basis` and `field_in_basis` in :class:`.EMESimulationData` can be used for this purpose after the simulation has been run.

        **Frequency Sweeps**

        Frequency sweeps are supported by including multiple frequencies in the `freqs` field. However, our EME solver repeats the mode solving for each new frequency, so frequency sweeps involving a large number of frequencies can be slow and expensive. If a large number of frequencies are required, consider using our FDTD solver instead.

        **Passivity and Unitarity Constraints**

        Passivity and unitarity constraints can be imposed via the `constraint` field. These constraints are imposed at interfaces between cells, possibly at the expense of field continuity. Passivity means that the interface can only dissipate energy, and unitarity means the interface will conserve energy (energy may still be dissipated inside cells when the propagation constant is complex). Adding constraints can slow down the simulation significantly, especially for a large number of modes (more than 30 or 40).

        **Too Many Modes**

        It is important to use enough modes to capture the physics of the device and to ensure that the results have converged (see below). However, using too many modes can slow down the simulation and result in numerical issues. If too many modes are used, it is common to see a warning about invalid modes in the simulation log. While these modes are not included in the EME propagation, this can indicate some other issue with the setup, especially if the results have not converged. In this case, extending the simulation size in the transverse directions and increasing the grid resolution may help by creating more valid modes that can be used in convergence testing.

        **Mode Convergence Sweeps**

        It is a good idea to check that the number of modes is large enough by running a mode convergence sweep. This can be done using :class:`.EMEModeSweep`.

    Example
    -------
    >>> from tidy3d import Box, Medium, Structure, C_0, inf
    >>> from tidy3d import EMEModeSpec, EMEUniformGrid, GridSpec
    >>> from tidy3d import EMEFieldMonitor
    >>> lambda0 = 1
    >>> freq0 = C_0 / lambda0
    >>> sim_size = 3*lambda0, 3*lambda0, 3*lambda0
    >>> waveguide_size = (lambda0/2, lambda0, inf)
    >>> waveguide = Structure(
    ...     geometry=Box(center=(0,0,0), size=waveguide_size),
    ...     medium=Medium(permittivity=2)
    ... )
    >>> eme_grid_spec = EMEUniformGrid(num_cells=5, mode_spec=EMEModeSpec(num_modes=10))
    >>> grid_spec = GridSpec(wavelength=lambda0)
    >>> field_monitor = EMEFieldMonitor(
    ...     size=(0, sim_size[1], sim_size[2]),
    ...     name="field_monitor"
    ... )
    >>> sim = EMESimulation(
    ...     size=sim_size,
    ...     monitors=[field_monitor],
    ...     structures=[waveguide],
    ...     axis=2,
    ...     freqs=[freq0],
    ...     eme_grid_spec=eme_grid_spec,
    ...     grid_spec=grid_spec
    ... )

    See Also
    --------

    **Notebooks:**
        * `EME Solver Demonstration <../../notebooks/docs/features/eme.rst>`_
    """

    freqs: List[pd.PositiveFloat] = pd.Field(
        ..., title="Frequencies", description="Frequencies for the EME simulation.", min_items=1
    )

    axis: Axis = pd.Field(
        ...,
        title="Propagation Axis",
        description="Propagation axis (0, 1, or 2) for the EME simulation.",
    )

    eme_grid_spec: EMEGridSpecType = pd.Field(
        ...,
        title="EME Grid Specification",
        description="Specification for the EME propagation grid. "
        "The simulation is divided into cells in the propagation direction; "
        "this parameter specifies the layout of those cells. "
        "Mode solving is performed in each cell, and then propagation between cells "
        "is performed to determine the complete solution. "
        "This is distinct from 'grid_spec', which defines the grid in the two "
        "tangential directions, as well as the grid used for field monitors.",
    )

    monitors: Tuple[annotate_type(EMEMonitorType), ...] = pd.Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )

    boundary_spec: BoundarySpec = pd.Field(
        BoundarySpec.all_sides(PECBoundary()),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides. NOTE: for EME simulations, this "
        "is required to be PECBoundary on all sides. To capture radiative effects, "
        "move the boundary farther away from the waveguide in the tangential directions, "
        "and increase the number of modes. The 'ModeSpec' can also be used to try "
        "different boundary conditions.",
    )

    sources: Tuple[None, ...] = pd.Field(
        (),
        title="Sources",
        description="Sources in the simulation. NOTE: sources are not currently supported "
        "for EME simulations. Instead, the simulation performs full bidirectional "
        "propagation in the 'port_mode' basis. After running the simulation, "
        "use 'smatrix_in_basis' to use another set of modes or input field.",
    )

    grid_spec: GridSpec = pd.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions. "
        "This is distinct from 'eme_grid_spec', which defines the 1D EME grid in the "
        "propagation direction.",
        validate_default=True,
    )

    subpixel: bool = pd.Field(
        True,
        title="Subpixel Averaging",
        description="If ``True``, uses subpixel averaging of the permittivity "
        "based on structure definition, resulting in much higher accuracy for a given grid size.",
    )

    store_port_modes: bool = pd.Field(
        True,
        title="Store Port Modes",
        description="Whether to store the modes associated with the two ports. "
        "Required to find scattering matrix in basis besides the computational basis.",
    )

    port_offsets: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, 0),
        title="Port Offsets",
        description="Offsets for the two ports, relative to the simulation bounds "
        "along the propagation axis.",
    )

    sweep_spec: Optional[EMESweepSpecType] = pd.Field(
        None,
        title="EME Sweep Specification",
        description="Specification for a parameter sweep to be performed during the EME "
        "propagation step. The results are stored "
        "in 'sim_data.smatrix'. Other simulation monitor data is not included in the sweep.",
    )

    constraint: Optional[Literal["passive", "unitary"]] = pd.Field(
        None,
        title="EME Constraint",
        description="Constraint for EME propagation, imposed at cell interfaces. "
        "A constraint of 'passive' means that energy can be dissipated but not created at "
        "interfaces. A constraint of 'unitary' means that energy is conserved at interfaces "
        "(but not necessarily within cells). The option 'none' may be faster "
        "for a large number of modes. The option 'passive' can serve as regularization "
        "for the field continuity requirement and give more physical results.",
    )

    @pd.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        # this is handled instead post-init to ensure freqs is defined
        return val

    @pd.validator("port_offsets", always=True)
    def _validate_port_offsets(cls, val, values):
        """Port offsets cannot jointly exceed simulation length."""
        total_offset = val[0] + val[1]
        size = values["size"]
        axis = values["axis"]
        if size[axis] < total_offset:
            raise SetupError(
                "The sum of the two 'port_offset' fields "
                "cannot exceed the simulation 'size' in the 'axis' direction."
            )
        return val

    @pd.validator("freqs", always=True)
    def _validate_freqs(cls, val):
        """Freqs cannot contain duplicates."""
        if len(set(val)) != len(val):
            raise SetupError(f"Simulation 'freqs={val}' cannot " "contain duplicate frequencies.")
        return val

    @pd.validator("structures", always=True)
    def _validate_structures(cls, val):
        """Validate and warn for certain medium types."""
        for ind, structure in enumerate(val):
            medium = structure.medium
            if isinstance(medium, FullyAnisotropicMedium):
                raise SetupError(
                    f"Structure at 'structures[{ind}]' has a medium which is a "
                    "'FullyAnisotropicMedium'. This medium class is not yet supported in EME."
                )
            if medium.is_time_modulated:
                log.warning(
                    f"Structure at 'structures[{ind}]' is time-modulated. The "
                    "time modulation is ignored in the EME solver."
                )
            if medium.is_nonlinear:
                log.warning(
                    f"Structure at 'structures[{ind}] is nonlinear. The nonlinearity "
                    "is ignored in the EME solver."
                )
        return val

    @equal_aspect
    @add_ax_if_none
    def plot_eme_ports(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """Plot the EME ports."""
        kwargs.setdefault("linewidth", 0.4)
        kwargs.setdefault("colors", "black")
        rmin = self.geometry.bounds[0][self.axis]
        rmax = self.geometry.bounds[1][self.axis]
        ports = np.array([rmin + self.port_offsets[0], rmax - self.port_offsets[1]])
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = ports
        if axis_y == self.axis:
            boundaries_y = ports
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eme_subgrid_boundaries(
        self,
        eme_grid_spec: EMEGridSpec,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """Plot the EME subgrid boundaries.
        Does nothing if ``eme_grid_spec`` is not :class:`.EMECompositeGrid`.
        Operates recursively on subgrids.
        """
        if not isinstance(eme_grid_spec, EMECompositeGrid):
            return ax
        kwargs.setdefault("linewidth", 0.4)
        kwargs.setdefault("colors", "black")
        subgrid_boundaries = np.array(eme_grid_spec.subgrid_boundaries)
        subgrids = eme_grid_spec.subgrids
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = subgrid_boundaries
        if axis_y == self.axis:
            boundaries_y = subgrid_boundaries
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        for subgrid in subgrids:
            ax = self.plot_eme_subgrid_boundaries(
                eme_grid_spec=subgrid, x=x, y=y, z=z, ax=ax, hlim=hlim, vlim=vlim, **kwargs
            )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eme_grid(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """Plot the EME grid."""
        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        cell_boundaries = self.eme_grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = []
        boundaries_y = []
        if axis_x == self.axis:
            boundaries_x = cell_boundaries
        if axis_y == self.axis:
            boundaries_y = cell_boundaries
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

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

        ax = self.scene.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        ax = self.plot_eme_grid(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_eme_subgrid_boundaries(
            eme_grid_spec=self.eme_grid_spec, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_eme_ports(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        return ax

    @cached_property
    def eme_grid(self) -> EMEGrid:
        """The EME grid as defined by 'eme_grid_spec'.
        An EME grid is a 1D grid aligned with the propagation axis,
        dividing the simulation into cells. Modes and mode coefficients
        are defined at the central plane of each cell. Typically,
        cell boundaries are aligned with interfaces between structures
        in the simulation.

        This is distinct from 'grid', which is the grid used in the tangential directions
        as well as the grid used for field monitors.
        """
        center = list(self.center)
        size = list(self.size)
        axis = self.axis
        rmin = center[axis] - size[axis] / 2
        rmax = center[axis] + size[axis] / 2
        rmin += self.port_offsets[0]
        rmax -= self.port_offsets[1]
        center[axis] = (rmax + rmin) / 2
        size[axis] = rmax - rmin
        return self.eme_grid_spec.make_grid(center=center, size=size, axis=self.axis)

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> EMESimulation:
        """Create an EME simulation from a :class:.`Scene` instance. Must provide additional parameters
        to define a valid EME simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:.`Scene`
            Scene containing structures information.
        **kwargs
            Other arguments

        """
        return cls(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )

    @property
    def mode_solver_monitors(self) -> List[ModeSolverMonitor]:
        """A list of mode solver monitors at the cell centers.
        Each monitor has a mode spec. The cells and mode specs
        are specified by 'eme_grid_spec'."""
        monitors = []
        freqs = self.freqs
        mode_planes = self.eme_grid.mode_planes
        mode_specs = [eme_mode_spec._to_mode_spec() for eme_mode_spec in self.eme_grid.mode_specs]
        for i in range(self.eme_grid.num_cells):
            monitor = ModeSolverMonitor(
                center=mode_planes[i].center,
                size=mode_planes[i].size,
                name=f"_eme_mode_solver_monitor_{i}",
                freqs=freqs,
                mode_spec=mode_specs[i],
                colocate=False,
            )
            monitors.append(monitor)
        return monitors

    @property
    def port_modes_monitor(self) -> EMEModeSolverMonitor:
        """EME Mode solver monitor for only the port modes."""
        return EMEModeSolverMonitor(
            center=self.center,
            size=self.size,
            eme_cell_interval_space=self.eme_grid.num_cells,
            name="_eme_port_modes_monitor",
            colocate=False,
            num_modes=self.max_port_modes,
        )

    def _post_init_validators(self) -> None:
        """Call validators taking `self` that get run after init."""
        _ = self.grid
        _ = self.eme_grid
        _ = self.mode_solver_monitors
        log.begin_capture()
        self._validate_monitor_setup()
        self._validate_sources_and_boundary()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._validate_sweep_spec()
        self._validate_symmetry()
        # self._warn_monitor_interval()
        log.end_capture(self)

    def _validate_symmetry(self):
        """Symmetry in propagation direction is not supported."""
        if self.symmetry[self.axis] != 0:
            raise SetupError("Symmetry in the propagation diretion is not currently supported.")

    # uncomment once interval_space != 1 is supported in any monitors
    # def _warn_monitor_interval(self):
    #    """EMEModeSolverMonitor does not use interval_space in propagation direction."""
    #    for monitor in self.monitors:
    #        if isinstance(monitor, EMEModeSolverMonitor):
    #            if monitor.interval_space[self.axis] != 1:
    #                log.warning(
    #                    "'EMEModeSolverMonitor' has 'interval_space != 1' "
    #                    "in the propagation axis. This value is not used; "
    #                    "it always monitors every EME cell."
    #                )

    def _validate_sweep_spec(self):
        """Validate sweep spec."""
        if self.sweep_spec is None:
            return
        # mode sweep can't exceed max num modes
        if isinstance(self.sweep_spec, EMEModeSweep):
            if len(self.sweep_spec.num_modes) > MAX_NUM_SWEEP:
                raise SetupError(
                    "Simulation 'sweep_spec' has "
                    f"'len(num_modes)={len(self.sweep_spec.num_modes)}', which exceeds "
                    f"the maximum allowed '{MAX_NUM_SWEEP}'."
                )
            if any(self.sweep_spec.num_modes > self.max_num_modes):
                raise SetupError(
                    "Simulation 'sweep_spec' is an 'EMEModeSweep'. "
                    "The number of modes should not exceed the maximum number of "
                    "modes in any EME cell. Provided "
                    f"'num_modes={self.sweep_spec.num_modes}'; the maximum "
                    f"number of EME modes is '{self.max_num_modes}'."
                )
        if isinstance(self.sweep_spec, EMELengthSweep):
            if len(self.sweep_spec.scale_factors) > MAX_NUM_SWEEP:
                raise SetupError(
                    "Simulation 'sweep_spec' has "
                    f"'len(scale_factors)={len(self.sweep_spec.scale_factors)}', which exceeds "
                    f"the maximum allowed '{MAX_NUM_SWEEP}'."
                )

    def _validate_monitor_setup(self):
        """Check monitor setup."""
        for monitor in self.monitors:
            if (
                hasattr(monitor, "freqs")
                and monitor.freqs is not None
                and not (len(set(monitor.freqs)) == len(monitor.freqs))
            ):
                raise SetupError(f"Monitor 'freqs={monitor.freqs}' cannot contain duplicates.")
            if (
                hasattr(monitor, "freqs")
                and monitor.freqs is not None
                and not (set(monitor.freqs).issubset(set(self.freqs)))
            ):
                raise SetupError(
                    f"Monitor 'freqs={monitor.freqs}' "
                    f"must be a subset of simulation 'freqs={self.freqs}'."
                )
            if (
                hasattr(monitor, "num_modes")
                and monitor.num_modes is not None
                and not (monitor.num_modes <= self.max_num_modes)
            ):
                raise SetupError(
                    f"Monitor has 'num_modes={monitor.num_modes}', which exceeds the "
                    "maximum number of modes in the 'eme_grid', which is "
                    f"'mode_spec.num_modes={self.max_num_modes}'."
                )
            if (
                isinstance(monitor, EMEFieldMonitor)
                and monitor.num_modes is not None
                and not (monitor.num_modes <= self.max_port_modes)
            ):
                raise SetupError(
                    f"EMEFieldMonitor has 'num_modes={monitor.num_modes}', which exceeds the "
                    "max number of modes of the two EME ports, which is "
                    f"'mode_spec.num_modes={self.max_port_modes}'."
                )

    def _validate_sources_and_boundary(self):
        """Disallow sources and boundary."""
        if self.boundary_spec != BoundarySpec.all_sides(PECBoundary()):
            raise SetupError(
                "In an EME simulation, the 'boundary_spec' must be `PECBoundary` "
                "on all sides (the default value). The boundary condition along "
                "the propagation axis is always transparent; boundary conditions "
                "in the tangential directions are imposed in 'mode_spec' in the "
                "EME grid."
            )
        # commented because sources has type Tuple[None, ...]
        # if self.sources != ():
        #    raise SetupError(
        #        "EME simulations do not currently support sources. "
        #        "The simulation performs full bidirectional propagation in the "
        #        "'port_mode' basis. After running the simulation, use "
        #        "'smatrix_in_basis' to use another set of modes. "
        #    )

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""

        num_comp_cells = self.num_cells / 2 ** (np.sum(np.abs(self.symmetry)))
        if num_comp_cells > MAX_GRID_CELLS:
            raise SetupError(
                f"Simulation has {num_comp_cells:.2e} computational cells, "
                f"a maximum of {MAX_GRID_CELLS:.2e} are allowed."
            )

        num_freqs = len(self.freqs)
        if num_freqs > MAX_NUM_FREQS:
            raise SetupError(
                f"Simulation has {num_freqs:.2e} frequencies, "
                f"a maximum of {MAX_NUM_FREQS:.2e} are allowed. Currently, mode solving "
                f"is repeated at each frequency, so EME simulations with too many frequencies "
                f"can be slower and more expensive than FDTD simulations. In the future, "
                f"we will support efficient frequency sweeps via perturbative mode solving."
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
            if isinstance(monitor, EMEMonitor):
                num_eme_cells = self._monitor_num_eme_cells(monitor)
                num_freqs = self._monitor_num_freqs(monitor)
                num_modes = self._monitor_num_modes(monitor)
                solver_data = (
                    monitor.storage_size(
                        num_cells=num_cells,
                        num_eme_cells=num_eme_cells,
                        num_freqs=num_freqs,
                        num_modes=num_modes,
                    )
                    / 1e9
                )
            else:
                solver_data = (monitor.storage_size(num_cells=num_cells, tmesh=0)) / 1e9
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

    @cached_property
    def monitors_data_size(self) -> Dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self.monitors:
            num_cells = self._monitor_num_cells(monitor)
            if isinstance(monitor, EMEMonitor):
                num_eme_cells = self._monitor_num_eme_cells(monitor)
                num_freqs = self._monitor_num_freqs(monitor)
                num_modes = self._monitor_num_modes(monitor)
                storage_size = float(
                    monitor.storage_size(
                        num_cells=num_cells,
                        num_eme_cells=num_eme_cells,
                        num_freqs=num_freqs,
                        num_modes=num_modes,
                    )
                )
            else:
                storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=0))
            data_size[monitor.name] = storage_size
        return data_size

    def _monitor_eme_cell_indices(self, monitor: EMEMonitor) -> List[pd.NonNegativeInt]:
        """EME cell indices inside monitor. Takes into account 'eme_cell_interval_space'."""
        cell_indices_full = self.eme_grid.cell_indices_in_box(box=monitor.geometry)
        if len(cell_indices_full) == 0:
            raise SetupError(f"Monitor '{monitor.name}' does not intersect any EME cells.")
        cell_indices = cell_indices_full[:: monitor.eme_cell_interval_space]
        # make sure last index is included
        if cell_indices[-1] != cell_indices_full[-1]:
            cell_indices.append(cell_indices_full[-1])
        return cell_indices

    def _monitor_num_eme_cells(self, monitor: EMEMonitor) -> int:
        """Total number of EME cells included in monitor based on simulation grid."""
        return len(self._monitor_eme_cell_indices(monitor=monitor))

    def _monitor_freqs(self, monitor: Monitor) -> List[pd.NonNegativeFloat]:
        """Monitor frequencies."""
        if monitor.freqs is None:
            return self.freqs
        return np.intersect1d(self.freqs, monitor.freqs)

    def _monitor_num_freqs(self, monitor: Monitor) -> int:
        """Total number of freqs included in monitor."""
        return len(self._monitor_freqs(monitor=monitor))

    def _monitor_num_modes(self, monitor: Monitor) -> int:
        """Total number of modes included in monitor."""
        sim_max_num_modes = (
            self.max_port_modes if isinstance(monitor, EMEFieldMonitor) else self.max_num_modes
        )
        if not hasattr(monitor, "num_modes") or monitor.num_modes is None:
            return sim_max_num_modes
        return min(monitor.num_modes, sim_max_num_modes)

    def _monitor_num_modes_cell(self, monitor: Monitor, cell_index: int) -> int:
        """Number of modes included in monitor at certain cell_index."""
        return min(
            self.eme_grid.mode_specs[cell_index].num_modes, self._monitor_num_modes(monitor=monitor)
        )

    @cached_property
    def max_num_modes(self) -> int:
        """Max number of modes in the simulation."""
        return np.max([mode_spec.num_modes for mode_spec in self.eme_grid.mode_specs])

    @cached_property
    def max_port_modes(self) -> int:
        """Max number of modes at the two ports."""
        return max(self.eme_grid.mode_specs[0].num_modes, self.eme_grid.mode_specs[-1].num_modes)

    @cached_property
    def grid(self) -> Grid:
        """Grid spatial locations and information as defined by `grid_spec`.
        This is the grid used in the tangential directions
        as well as the grid used for field monitors.
        This is distinct from 'eme_grid', which is the grid
        used for mode solving and EME propagation.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # TODO: add option (true by default) to make Yee grid conformal to EME grid

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.structures

        # make source for autogrid if needed
        freqs = self.freqs
        grid_spec = self.grid_spec
        sources = []
        if grid_spec.auto_grid_used and grid_spec.wavelength is None:
            if not np.all(np.isclose(freqs, freqs[0])):
                raise SetupError(
                    "Multiple 'sim.freqs' are supplied. Please supply "
                    "a 'wavelength' value for 'grid_spec' to control automatic "
                    "grid generation."
                )
            # this is already logged in auto mesher
            # wavelength = C_0 / freqs[0]
            # log.info(f"Auto meshing using wavelength {wavelength:1.4f} defined from freqs.")
            plane = self.eme_grid.mode_planes[0]
            sources.append(
                ModeSource(
                    center=plane.center,
                    size=plane.size,
                    source_time=GaussianPulse(freq0=freqs[0], fwidth=0.1 * freqs[0]),
                    direction="+",
                    mode_spec=self.eme_grid.mode_specs[0],
                )
            )

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=sources,
            num_pml_layers=self.num_pml_layers,
        )

        # This would AutoGrid the in-plane directions of the 2D materials
        # return self._grid_corrections_2dmaterials(grid)
        return grid
