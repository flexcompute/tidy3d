"""Defines EME simulation class."""
from __future__ import annotations

from typing import Tuple, List, Dict

import pydantic.v1 as pd
import matplotlib as mpl
import numpy as np

from ..types import Ax
from ..base_sim.simulation import AbstractSimulation
from ..simulation import Simulation
from ..viz import add_ax_if_none, equal_aspect
from ..boundary import BoundarySpec, Periodic, BlochBoundary, PECBoundary
from ..boundary import PML, StablePML, Absorber
from ..grid.grid_spec import GridSpec
from ..grid.grid import Grid, Coords
from ..types import Axis
from ..base import cached_property
from ..scene import Scene
from ..structure import Structure
from ..medium import Medium
from ..monitor import ModeSolverMonitor, AbstractModeMonitor, FieldMonitor, Monitor
from ..source import ModeSource
from ..viz import plot_params_structure
from ..geometry.base import Box
from ...exceptions import ValidationError, SetupError
from ...log import log
from ...constants import C_0, fp_eps

from .monitor import EMEMonitor, EMEMonitorType, EMEModeSolverMonitor
from .grid import EMEGridSpecType, EMEGrid


# dummy run_time for converting to FDTD simulation
# should be fairly small, or else we run into issues constructing tmesh
RUN_TIME = 1e-30


# maximum numbers of simulation parameters
MAX_GRID_CELLS = 20e9
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5


# eme specific simulation parameters
MAX_NUM_FREQS = 100


class EMESimulation(AbstractSimulation):
    """EME simulation.
    An EME simulation always computes the full scattering matrix of the structure.
    Additional data can be recorded by adding 'monitors' to the simulation.

    Example
    -------
    >>> from tidy3d import Box, Medium, Structure, C_0, inf
    >>> from tidy3d import ModeSpec, EMEUniformGrid, GridSpec
    >>> from tidy3d import EMEFieldMonitor
    >>> lambda0 = 1
    >>> freq0 = C_0 / lambda0
    >>> sim_size = 3*lambda0, 3*lambda0, 3*lambda0
    >>> waveguide_size = (lambda0/2, lambda0, inf)
    >>> waveguide = Structure(
    ...     geometry=Box(center=(0,0,0), size=waveguide_size),
    ...     medium=Medium(permittivity=2)
    ... )
    >>> eme_grid_spec = EMEUniformGrid(num_cells=5, mode_spec=ModeSpec(num_modes=10))
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

    monitors: Tuple[EMEMonitorType, ...] = pd.Field(
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
        "Required for use with scattering matrix plugin.",
    )

    @pd.root_validator(pre=False)
    def _validate_grid_spec(cls, values):
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        grid_spec = values["grid_spec"]
        # in this case, we must try to determine the wavelength
        if grid_spec is None or (grid_spec.auto_grid_used and grid_spec.wavelength is None):
            freqs = values.get("freqs")
            if freqs is None:
                raise ValidationError(
                    "Automatic grid generation requires the input of 'wavelength' or "
                    "'freqs' in the simulation."
                )
            if not np.all(np.isclose(freqs, freqs[0])):
                raise ValidationError(
                    "The 'grid_spec' has 'AutoGrid' components but its "
                    "'wavelength' is not provided. Multiple 'freqs' are provided, "
                    "so the wavelength for FDTD grid generation cannot be determined. "
                    "Please set the 'wavelength' field of the 'grid_spec'."
                )
            wavelength = C_0 / freqs[0]
            log.warning(
                "The 'grid_spec' has 'AutoGrid' components but its "
                "'wavelength' is not provided. Using 'C_0 / freqs[0]' as "
                "the wavelength for FDTD grid generation."
            )
        if grid_spec is None:
            grid_spec = GridSpec(wavelength=wavelength)
        elif grid_spec.auto_grid_used and grid_spec.wavelength is None:
            grid_spec = grid_spec.updated_copy(wavelength=wavelength)
        values["grid_spec"] = grid_spec
        return values

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
        """Plot boundaries"""
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

        structures = [
            Structure(geometry=geometry, medium=Medium()) for geometry in self.eme_grid.cells
        ]
        scene = Scene(structures=structures)
        medium_shapes = scene._get_structures_2dbox(
            structures=structures, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        plot_params_struct = plot_params_structure.updated_copy(linewidth=0.2, alpha=0.5)
        for _medium, shape in medium_shapes:
            ax = scene.box.plot_shape(shape=shape, plot_params=plot_params_struct, ax=ax)
        ax = scene._set_plot_bounds(bounds=self.bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        # clean up the axis display
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = scene.box.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

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
        return self.eme_grid_spec.make_grid(center=self.center, size=self.size, axis=self.axis)

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

    def _to_fdtd_sim(self, **kwargs) -> Simulation:
        """Construct an FDTD simulation from an EME simulation. Must provide additional
        parameters to define a valid FDTD simulation (for example, ``run_time``).

        Used to generate the FDTD grid.
        """
        return Simulation(
            center=self.center,
            size=self.size,
            medium=self.medium,
            structures=self.structures,
            symmetry=self.symmetry,
            grid_spec=self.grid_spec,
            boundary_spec=self.boundary_spec,
            version=self.version,
            subpixel=self.subpixel,
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
        mode_specs = self.eme_grid.mode_specs
        for i in range(self.eme_grid.num_cells):
            monitor = ModeSolverMonitor(
                center=mode_planes[i].center,
                size=mode_planes[i].size,
                name=f"mode_solver_monitor_{i}",
                freqs=freqs,
                mode_spec=mode_specs[i],
                colocate=False,
            )
            monitors.append(monitor)
        return monitors

    def _post_init_validators(self) -> None:
        """Call validators taking `self` that get run after init."""
        _ = self.eme_grid
        _ = self.mode_solver_monitors
        # _ = self._to_fdtd_sim(run_time=RUN_TIME)
        log.begin_capture()
        self._validate_monitor_setup()
        self._validate_sources_and_boundary()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._warn_monitor_interval()
        log.end_capture(self)

    def _warn_monitor_interval(self):
        """EMEModeSolverMonitor does not use interval_space in propagation direction."""
        for monitor in self.monitors:
            if isinstance(monitor, EMEModeSolverMonitor):
                if monitor.interval_space[self.axis] != 1:
                    log.warning(
                        "'EMEModeSolverMonitor' has 'interval_space != 1' "
                        "in the propagation axis. This value is not used; "
                        "it always monitors every EME cell."
                    )

    def _validate_monitor_setup(self):
        """Check monitor setup."""
        for monitor in self.monitors:
            if (
                hasattr(monitor, "freqs")
                and monitor.freqs is not None
                and not (set(monitor.freqs).issubset(set(self.freqs)))
            ):
                raise SetupError(
                    f"Monitor 'freqs={monitor.freqs}' "
                    f"must be a subset of simulation =freqs={self.freqs}'."
                )
            if (
                hasattr(monitor, "mode_indices")
                and monitor.mode_indices is not None
                and not (set(monitor.mode_indices).issubset(set(np.arange(self.max_num_modes))))
            ):
                raise SetupError(
                    f"Values of monitor 'mode_indices={monitor.mode_indices}' "
                    f"must be between zero and the maximum number of modes in the 'eme_grid', "
                    f"which is 'mode_spec.num_modes={self.max_num_modes}'"
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
        if self.sources != ():
            raise SetupError(
                "EME simulations do not currently support sources. "
                "The simulation performs full bidirectional propagation in the "
                "'port_mode' basis. After running the simulation, use "
                "'smatrix_in_basis' to use another set of modes. "
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
                f"a maximum of {MAX_NUM_FREQS:.2e} are allowed."
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
            num_eme_cells = self._monitor_num_eme_cells(monitor)
            num_freqs = self._monitor_num_freqs(monitor)
            num_modes = self._monitor_num_modes(monitor)
            # intermediate storage needed, in GB
            if isinstance(monitor, EMEMonitor):
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
            num_eme_cells = self._monitor_num_eme_cells(monitor)
            num_freqs = self._monitor_num_freqs(monitor)
            num_modes = self._monitor_num_modes(monitor)
            if isinstance(monitor, EMEMonitor):
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

        # if isinstance(monitor, SurfaceIntegrationMonitor):
        #    return sum(num_cells_in_monitor(mnt) for mnt in monitor.integration_surfaces)
        return num_cells_in_monitor(monitor)

    def _monitor_num_eme_cells(self, monitor: Monitor) -> int:
        """Total number of EME cells included in monitor based on simulation grid."""
        return len(self.eme_grid.cell_indices_in_box(monitor.geometry))

    def _monitor_num_freqs(self, monitor: Monitor) -> int:
        """Total number of freqs included in monitor."""
        if monitor.freqs is None:
            return len(self.freqs)
        return len(np.intersect1d(self.freqs, monitor.freqs))

    def _monitor_num_modes(self, monitor: Monitor) -> int:
        """Total number of modes included in monitor."""
        if not hasattr(monitor, "mode_indices") or monitor.mode_indices is None:
            return self.max_num_modes
        return len(np.intersect1d(np.arange(self.max_num_modes), monitor.mode_indices))

    @cached_property
    def max_num_modes(self) -> int:
        """Max number of modes in the simulation."""
        return np.max([mode_spec.num_modes for mode_spec in self.eme_grid.mode_specs])

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

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

        return grid

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
