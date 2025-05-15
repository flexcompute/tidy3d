"""Defines EME simulation class."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

try:
    import matplotlib as mpl
except ImportError:
    pass
import numpy as np
import pydantic.v1 as pd

from ...constants import C_0
from ...exceptions import SetupError, ValidationError
from ...log import log
from ..base import cached_property
from ..boundary import BoundarySpec, PECBoundary
from ..geometry.base import Box
from ..grid.grid import Grid
from ..grid.grid_spec import GridSpec
from ..medium import FullyAnisotropicMedium
from ..monitor import AbstractModeMonitor, ModeSolverMonitor, Monitor, MonitorType
from ..scene import Scene
from ..simulation import AbstractYeeGridSimulation, Simulation, validate_boundaries_for_zero_dims
from ..types import Ax, Axis, FreqArray, Symmetry, annotate_type
from ..validators import (
    MIN_FREQUENCY,
    validate_freqs_min,
    validate_freqs_not_empty,
)
from ..viz import add_ax_if_none, equal_aspect
from .grid import EMECompositeGrid, EMEExplicitGrid, EMEGrid, EMEGridSpec, EMEGridSpecType
from .monitor import (
    EMECoefficientMonitor,
    EMEFieldMonitor,
    EMEModeSolverMonitor,
    EMEMonitor,
    EMEMonitorType,
)
from .sweep import EMEFreqSweep, EMELengthSweep, EMEModeSweep, EMEPeriodicitySweep, EMESweepSpecType

# maximum numbers of simulation parameters
MAX_GRID_CELLS = 20e9
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5


# eme specific simulation parameters
WARN_NUM_FREQS = 20
MAX_NUM_FREQS = 500
MAX_NUM_SWEEP = 100


# constraint can be slow with too many modes
WARN_CONSTRAINT_NUM_MODES = 50

# dummy run time for conversion to FDTD sim
# should be very small -- otherwise, generating tmesh will fail or take a long time
RUN_TIME = 1e-30

EME_SIM_YEE_SIM_SHARED_ATTRS = [
    "center",
    "size",
    "medium",
    "structures",
    "symmetry",
    "boundary_spec",
    "version",
    "plot_length_units",
    "lumped_elements",
    "subpixel",
    "simulation_type",
    "post_norm",
]


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
    >>> lambda0 = 1550e-9
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

    freqs: FreqArray = pd.Field(
        ...,
        title="Frequencies",
        description="Frequencies for the EME simulation. "
        "The field is propagated independently at each provided frequency. "
        "This can be slow when the number of frequencies is large. In this case, "
        "consider using the approximate 'EMEFreqSweep' as the 'sweep_spec' "
        "instead of providing all desired frequencies here.",
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
        description="Specification of boundary conditions along each dimension. "
        "By default, PEC boundary conditions are applied on all sides. "
        "This field is for consistency with FDTD simulations; however, please note that "
        "regardless of the 'boundary_spec', the mode solver terminates the mode plane "
        "with PEC boundary. The 'EMEModeSpec' can be used to "
        "apply PML layers in the mode solver.",
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

    store_port_modes: bool = pd.Field(
        True,
        title="Store Port Modes",
        description="Whether to store the modes associated with the two ports. "
        "Required to find scattering matrix in basis besides the computational basis.",
    )

    normalize: bool = pd.Field(
        True,
        title="Normalize Scattering Matrix",
        description="Whether to normalize the port modes to unity flux, "
        "thereby normalizing the scattering matrix and expansion coefficients.",
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
        "passive",
        title="EME Constraint",
        description="Constraint for EME propagation, imposed at cell interfaces. "
        "A constraint of 'passive' means that energy can be dissipated but not created at "
        "interfaces. A constraint of 'unitary' means that energy is conserved at interfaces "
        "(but not necessarily within cells). The option 'none' may be faster "
        "for a large number of modes. The option 'passive' can serve as regularization "
        "for the field continuity requirement and give more physical results.",
    )

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    @pd.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        # this is handled instead post-init to ensure freqs is defined
        return val

    @pd.validator("freqs", always=True)
    def _validate_freqs(cls, val):
        """Freqs cannot contain duplicates."""
        if len(set(val)) != len(val):
            raise SetupError(f"'EMESimulation' 'freqs={val}' cannot contain duplicate frequencies.")
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
        """Create an EME simulation from a :`.Scene` instance. Must provide additional parameters
        to define a valid EME simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:`.Scene`
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
        freqs = list(self.freqs)
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
            num_sweep=None,
            normalize=self.normalize,
        )

    def _post_init_validators(self) -> None:
        """Call validators taking `self` that get run after init."""
        self._validate_port_offsets()
        _ = self.grid
        _ = self.eme_grid
        _ = self.mode_solver_monitors
        _ = self._cell_index_pairs
        self._validate_too_close_to_edges()
        self._validate_sweep_spec()
        self._validate_symmetry()
        self._validate_monitor_setup()

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized EME simulation is ok for upload to our servers."""
        log.begin_capture()
        self._validate_sweep_spec_size()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._validate_constraint()
        # self._warn_monitor_interval()
        log.end_capture(self)

    def _validate_too_close_to_edges(self):
        """Can't have mode planes closer to boundary than extreme Yee grid center."""
        cell_centers = self.eme_grid.centers
        yee_centers = list(self.grid.centers.to_dict.values())[self.axis]
        if cell_centers[0] < yee_centers[0]:
            raise SetupError(
                "The first EME cell center must be further from the boundary "
                "than the first Yee cell center, "
                f"currently {cell_centers[0]} compared to {yee_centers[0]}."
            )
        if cell_centers[-1] > yee_centers[-1]:
            raise SetupError(
                "The last EME cell center must be further from the boundary "
                "than the last Yee cell center, "
                f"currently {cell_centers[-1]} compared to {yee_centers[-1]}."
            )
        for ind, monitor in enumerate(self.monitors):
            if isinstance(monitor, ModeSolverMonitor) and monitor.normal_axis == self.axis:
                center = monitor.center[monitor.normal_axis]
                if center < yee_centers[0] or center > yee_centers[-1]:
                    raise SetupError(
                        f"'ModeSolverMonitor' at 'monitors[{ind}]' has "
                        f"center {center}, which is within half a Yee cell "
                        "of the simulation boundary along the propagation axis. "
                        "Please move the monitor further from the boundary."
                    )

    def _validate_constraint(self):
        """Constraint can be slow with too many modes. Warn in this case."""
        constraint = self.constraint
        max_num_modes = self.max_num_modes
        if constraint is not None and max_num_modes > WARN_CONSTRAINT_NUM_MODES:
            log.warning(
                f"The simulation has 'constraint={constraint}', and the maximum "
                f"number of EME modes in the simulation is '{max_num_modes}'. "
                f"Using more than '{WARN_CONSTRAINT_NUM_MODES}' modes together with "
                "a constraint can significantly slow down the simulation. Consider "
                "reducing the number of modes or setting 'constraint=None'."
            )

    def _validate_port_offsets(self):
        """Port offsets cannot jointly exceed simulation length."""
        total_offset = self.port_offsets[0] + self.port_offsets[1]
        size = self.size
        axis = self.axis
        if size[axis] < total_offset:
            raise ValidationError(
                "The sum of the two 'port_offset' fields "
                "cannot exceed the simulation 'size' in the 'axis' direction."
            )

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

    def _validate_sweep_spec_size(self):
        """Make sure sweep spec is not too large."""
        if self.sweep_spec is None:
            return
        num_sweep = self.sweep_spec.num_sweep
        if num_sweep > MAX_NUM_SWEEP:
            raise SetupError(
                f"Simulation 'sweep_spec' has 'num_sweep={num_sweep}, "
                f"which exceeds the maximum allowed '{MAX_NUM_SWEEP}'."
            )

    def _validate_sweep_spec(self):
        """Validate sweep spec."""
        if self.sweep_spec is None:
            return
        num_sweep = self.sweep_spec.num_sweep
        if num_sweep == 0:
            raise SetupError("Simulation 'sweep_spec' has 'num_sweep=0'.")
        if isinstance(self.sweep_spec, EMEModeSweep):
            if any(self.sweep_spec.num_modes > self.max_num_modes):
                raise SetupError(
                    "Simulation 'sweep_spec' is an 'EMEModeSweep'. "
                    "The number of modes should not exceed the maximum number of "
                    "modes in any EME cell. Provided "
                    f"'num_modes={self.sweep_spec.num_modes}'; the maximum "
                    f"number of EME modes is '{self.max_num_modes}'."
                )
        elif isinstance(self.sweep_spec, EMELengthSweep):
            scale_factors_shape = self.sweep_spec.scale_factors.shape
            if len(scale_factors_shape) > 2:
                raise SetupError(
                    "Simulation 'sweep_spec.scale_factors' must "
                    "have either one or two dimensions."
                )
            if len(scale_factors_shape) == 2:
                num_scale_factors = scale_factors_shape[1]
                if num_scale_factors != self.eme_grid.num_cells:
                    raise SetupError(
                        "Simulation 'sweep_spec.scale_factors' has shape "
                        f"'{scale_factors_shape}'. The size of the second dimension "
                        "must equal the number of EME cells in the simulation, which is "
                        f"'{self.eme_grid.num_cells}'."
                    )
        elif isinstance(self.sweep_spec, EMEFreqSweep):
            for i, scale_factor in enumerate(self.sweep_spec.freq_scale_factors):
                scaled_freqs = np.array(self.freqs) * scale_factor
                if np.min(scaled_freqs) < MIN_FREQUENCY:
                    raise SetupError(
                        f"Simulation 'sweep_spec' at sweep index {i} results in "
                        f"scaled frequencies {scaled_freqs}; the minimum allowed is "
                        f"{MIN_FREQUENCY:.0e} Hz."
                    )
        elif isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for i, monitor in enumerate(self.monitors):
                if isinstance(monitor, EMEFieldMonitor):
                    raise SetupError(
                        f"Monitor at 'monitors[{i}]' is an 'EMEFieldMonitor', "
                        "which is not compatible with 'EMEPeriodicitySweep'."
                    )
                elif isinstance(monitor, EMECoefficientMonitor):
                    raise SetupError(
                        f"Monitor at 'monitors[{i}]' is an 'EMECoefficientMonitor', "
                        "which is not compatible with 'EMEPeriodicitySweep'."
                    )

    def _validate_monitor_setup(self):
        """Check monitor setup."""
        for i, monitor in enumerate(self.monitors):
            if isinstance(monitor, EMEMonitor):
                _ = self._monitor_eme_cell_indices(monitor=monitor)
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
                hasattr(monitor, "num_sweep")
                and monitor.num_sweep is not None
                and self.sweep_spec is not None
                and not (monitor.num_sweep <= self.sweep_spec.num_sweep)
            ):
                raise SetupError(
                    f"Monitor has 'num_sweep={monitor.num_sweep}', which exceeds the "
                    "number of sweep indices in the simulation 'sweep_spec', which is "
                    f"'{self.sweep_spec.num_sweep}'."
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
            if isinstance(monitor, EMEFieldMonitor):
                if not np.array_equal(
                    self.eme_grid_spec.virtual_cell_indices, self.eme_grid_spec.real_cell_indices
                ):
                    raise SetupError(
                        f"Monitor at 'monitors[{i}]' is an 'EMEFieldMonitor', "
                        "which is not compatible with periodic repetition "
                        "('num_reps != 1' in any 'EMEGridSpec'.)"
                    )

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
                f"a maximum of {MAX_NUM_FREQS:.2e} are allowed. Mode solving "
                f"is repeated at each frequency, so EME simulations with too many frequencies "
                f"can be slower and more expensive than FDTD simulations. "
                f"Consider using an 'EMEFreqSweep' instead for a faster approximate solution."
            )
        if num_freqs > WARN_NUM_FREQS:
            log.warning(
                f"Simulation has {num_freqs:.2e} frequencies. Mode solving "
                f"is repeated at each frequency, so EME simulations with too many frequencies "
                f"can be slower and more expensive than FDTD simulations. "
                f"Consider using an 'EMEFreqSweep' instead for a faster approximate solution."
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
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed. Note that "
                "this estimate includes the port modes if 'store_port_modes' is 'True'."
            )

        # Make sure that internal storage from mode solvers also does not exceed the limit.
        for monitor in self.mode_solver_monitors:
            num_cells = self._monitor_num_cells(monitor)
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
            for mnt_ind, monitor in enumerate(self.monitors):
                if isinstance(monitor, AbstractModeMonitor):
                    msg_header = f"Mode monitor '{monitor.name}' "
                    custom_loc = ["monitors", mnt_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)
            for mnt_ind, monitor in enumerate(self.mode_solver_monitors):
                msg_header = f"Internal mode solver monitor '{monitor.name}' "
                custom_loc = ["mode_solver_monitors", mnt_ind]
                warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    @property
    def _monitors_full(self) -> Tuple[EMEMonitorType, ...]:
        """All monitors, including port modes monitor."""
        if self.store_port_modes:
            return list(self.monitors) + [self.port_modes_monitor]
        return list(self.monitors)

    @cached_property
    def monitors_data_size(self) -> Dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self._monitors_full:
            num_cells = self._monitor_num_cells(monitor)
            if isinstance(monitor, EMEMonitor):
                num_transverse_cells = self._monitor_num_transverse_cells(monitor)
                num_eme_cells = self._monitor_num_eme_cells(monitor)
                num_freqs = self._monitor_num_freqs(monitor)
                num_modes = self._monitor_num_modes(monitor)
                num_sweep = self._monitor_num_sweep(monitor)
                storage_size = float(
                    monitor.storage_size(
                        num_cells=num_cells,
                        num_transverse_cells=num_transverse_cells,
                        num_eme_cells=num_eme_cells,
                        num_freqs=num_freqs,
                        num_modes=num_modes,
                        num_sweep=num_sweep,
                    )
                )
            else:
                storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=0))
            data_size[monitor.name] = storage_size
        return data_size

    @property
    def _num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""
        if self.sweep_spec is None:
            return 1
        return self.sweep_spec.num_sweep

    @property
    def _sweep_modes(self) -> bool:
        """Whether the sweep changes the modes."""
        return self.sweep_spec is not None and isinstance(self.sweep_spec, EMEFreqSweep)

    @property
    def _num_sweep_modes(self) -> pd.PositiveInt:
        """Number of sweep indices for modes."""
        if self._sweep_modes:
            return self._num_sweep
        return 1

    @property
    def _sweep_interfaces(self) -> bool:
        """Whether the sweep changes the cell interface scattering matrices."""
        return self.sweep_spec is not None and isinstance(
            self.sweep_spec, (EMEFreqSweep, EMEModeSweep)
        )

    @property
    def _num_sweep_interfaces(self) -> pd.PositiveInt:
        """Number of sweep indices for interfaces."""
        if self._sweep_interfaces:
            return self._num_sweep
        return 1

    @property
    def _sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return self.sweep_spec is not None and isinstance(
            self.sweep_spec, (EMELengthSweep, EMEFreqSweep, EMEModeSweep)
        )

    @property
    def _num_sweep_cells(self) -> pd.PositiveInt:
        """Number of sweep indices for cells."""
        if self._sweep_cells:
            return self._num_sweep
        return 1

    def _monitor_num_sweep(self, monitor: EMEMonitor) -> pd.PositiveInt:
        """Number of sweep indices for a certain monitor."""
        if self.sweep_spec is None:
            return 1
        # only freq sweep changes the modes
        if isinstance(monitor, EMEModeSolverMonitor) and not self._sweep_modes:
            return 1
        if monitor.num_sweep is None:
            return self.sweep_spec.num_sweep
        return min(self.sweep_spec.num_sweep, monitor.num_sweep)

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
            return list(self.freqs)
        return list(monitor.freqs)

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

        return self._as_fdtd_sim.grid

    def _monitor_num_transverse_cells(self, monitor: Monitor) -> int:
        """Total number of cells transverse to propagation axis
        included in monitor based on simulation grid."""

        def num_transverse_cells_in_monitor(monitor: Monitor) -> int:
            """Get the number of transverse measurement cells in a
            monitor given the simulation grid and downsampling."""
            num_cells = self.discretize_monitor(monitor).num_cells
            # take monitor downsampling into account
            num_cells = list(monitor.downsampled_num_cells(num_cells))
            # pop propagation axis
            num_cells.pop(self.axis)
            return np.prod(np.array(num_cells, dtype=np.int64))

        return num_transverse_cells_in_monitor(monitor)

    @cached_property
    def _as_fdtd_sim(self) -> Simulation:
        """Convert :class:`.EMESimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""
        return self._to_fdtd_sim()

    def _to_fdtd_sim(self) -> Simulation:
        """Convert :class:`.EMESimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""

        grid_spec = self.grid_spec
        if grid_spec.auto_grid_used and grid_spec.wavelength is None:
            min_wvl = C_0 / np.max(self.freqs)
            log.info(
                f"Auto meshing using wavelength {min_wvl:1.4f} defined from "
                "largest of 'EMESimulation.freqs'."
            )
            grid_spec = grid_spec.updated_copy(wavelength=min_wvl)

        # copy over all FDTD monitors too
        monitors = [monitor for monitor in self.monitors if not isinstance(monitor, EMEMonitor)]

        kwargs = {key: getattr(self, key) for key in EME_SIM_YEE_SIM_SHARED_ATTRS}
        return Simulation(
            **kwargs,
            run_time=RUN_TIME,
            grid_spec=grid_spec,
            monitors=monitors,
        )

    def subsection(
        self,
        region: Box,
        grid_spec: Union[GridSpec, Literal["identical"]] = None,
        eme_grid_spec: Union[EMEGridSpec, Literal["identical"]] = None,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry] = None,
        monitors: Tuple[MonitorType, ...] = None,
        remove_outside_structures: bool = True,
        remove_outside_custom_mediums: bool = False,
        **kwargs,
    ) -> EMESimulation:
        """Generate a simulation instance containing only the ``region``.
        Same as in :class:`.AbstractYeeGridSimulation`, except also restricting EME grid.

        Parameters
        ----------
        region : :class:`.Box`
            New simulation domain.
        grid_spec : :class:`.GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:`.CustomGrid`. Note that in the latter case the region of the new simulation is
            snapped to the original grid lines.
        eme_grid_spec: :class:`.EMEGridSpec` = None
            New EME grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:`.EMEExplicitGrid`. Noe that in the latter case the region of the new simulation
            is expanded to contain full EME cells.
        symmetry : Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
            New simulation symmetry. If ``None``, then it is inherited from the original
            simulation. Note that in this case the size and placement of new simulation domain
            must be commensurate with the original symmetry.
        monitors : Tuple[MonitorType, ...] = None
            New list of monitors. If ``None``, then the monitors intersecting the new simulation
            domain are inherited from the original simulation.
        remove_outside_structures : bool = True
            Remove structures outside of the new simulation domain.
        remove_outside_custom_mediums : bool = True
            Remove custom medium data outside of the new simulation domain.
        **kwargs
            Other arguments passed to new simulation instance.
        """

        new_region = region
        if eme_grid_spec is None:
            eme_grid_spec = self.eme_grid_spec
        elif isinstance(eme_grid_spec, str) and eme_grid_spec == "identical":
            axis = self.axis
            mode_specs = self.eme_grid.mode_specs
            boundaries = self.eme_grid.boundaries
            indices = self.eme_grid.cell_indices_in_box(box=region)

            new_boundaries = boundaries[indices[0] : indices[-1] + 2]
            new_mode_specs = mode_specs[indices[0] : indices[-1] + 1]

            rmin = list(region.bounds[0])
            rmax = list(region.bounds[1])
            rmin[axis] = min(rmin[axis], new_boundaries[0])
            rmax[axis] = max(rmax[axis], new_boundaries[-1])
            new_region = Box.from_bounds(rmin=rmin, rmax=rmax)

            # remove outer boundaries for explicit grid
            new_boundaries = new_boundaries[1:-1]

            eme_grid_spec = EMEExplicitGrid(mode_specs=new_mode_specs, boundaries=new_boundaries)

        new_sim = super().subsection(
            region=new_region,
            grid_spec=grid_spec,
            symmetry=symmetry,
            monitors=monitors,
            remove_outside_structures=remove_outside_structures,
            remove_outside_custom_mediums=remove_outside_custom_mediums,
            **kwargs,
        )

        new_sim = new_sim.updated_copy(eme_grid_spec=eme_grid_spec)

        return new_sim

    @property
    def _cell_index_pairs(self) -> List[pd.NonNegativeInt]:
        """All the pairs of adjacent EME cells needed, taken over all sweep indices."""
        pairs = set()
        if isinstance(self.sweep_spec, EMEPeriodicitySweep):
            for num_reps in self.sweep_spec.num_reps:
                eme_grid_spec = self.eme_grid_spec._updated_copy_num_reps(num_reps=num_reps)
                pairs = pairs | set(eme_grid_spec._cell_index_pairs)
        else:
            pairs = set(self.eme_grid_spec._cell_index_pairs)
        return list(pairs)

    _boundaries_for_zero_dims = validate_boundaries_for_zero_dims()
