"""Defines mode solver simulation class."""

from __future__ import annotations

from math import isclose
from typing import List, Literal, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...exceptions import SetupError, ValidationError
from ...log import log
from ..base import cached_property, skip_if_fields_missing
from ..boundary import BoundarySpec, PECBoundary
from ..geometry.base import Box
from ..grid.grid import Grid
from ..grid.grid_spec import GridSpec
from ..medium import FullyAnisotropicMedium
from ..mode_spec import ModeSpec
from ..monitor import ModeMonitor, ModeSolverMonitor
from ..simulation import AbstractYeeGridSimulation, Simulation
from ..source import GaussianPulse, ModeSource, PointDipole, SourceTime
from ..structure import Structure
from ..types import (
    ArrayComplex3D,
    ArrayComplex4D,
    Axis,
    Direction,
    FreqArray,
    Symmetry,
)
from ..validators import validate_freqs_min, validate_freqs_not_empty

# dummy run time for conversion to FDTD sim
# should be very small -- otherwise, generating tmesh will fail or take a long time
RUN_TIME = 1e-30

# Maximum allowed size of the field data produced by the mode solver
MAX_MODES_DATA_SIZE_GB = 20

MODE_MONITOR_NAME = "<<<MODE_SOLVER_MONITOR>>>"


class ModeSimulation(AbstractYeeGridSimulation):
    """Mode solver simulation.

    Example
    -------
    >>> from tidy3d import C_0, ModeSpec
    >>> lambda0 = 1
    >>> freq0 = C_0 / lambda0
    >>> freqs = [freq0]
    >>> sim_size = lambda0, lambda0, 0
    >>> mode_spec = ModeSpec(num_modes=4)
    >>> sim = ModeSimulation(
    ...     size=sim_size,
    ...     freqs=freqs,
    ...     mode_spec=mode_spec
    ... )

    """

    mode_spec: ModeSpec = pd.Field(
        ...,
        title="Mode specification",
        description="Container with specifications about the modes to be solved for.",
    )

    freqs: FreqArray = pd.Field(
        ..., title="Frequencies", description="A list of frequencies at which to solve."
    )

    direction: Direction = pd.Field(
        "+",
        title="Propagation direction",
        description="Direction of waveguide mode propagation along the axis defined by its normal "
        "dimension.",
    )

    colocate: bool = pd.Field(
        True,
        title="Colocate fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default is ``True``.",
    )

    boundary_spec: BoundarySpec = pd.Field(
        BoundarySpec.all_sides(PECBoundary()),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. "
        "The default is PEC on all sides. Other boundary conditions may be specified for "
        "consistency with FDTD simulations; however, please note that the mode solver "
        "terminates the mode plane with PEC boundary. The 'ModeSpec' can be used to "
        "apply PML layers in the mode solver.",
    )

    monitors: Tuple[None, ...] = pd.Field(
        (),
        title="Monitors",
        description="Monitors in the simulation. Note: monitors are not supported in mode "
        "simulations.",
    )

    sources: Tuple[None, ...] = pd.Field(
        (),
        title="Sources",
        description="Sources in the simulation. Note: sources are not supported in mode "
        "simulations.",
    )

    grid_spec: GridSpec = pd.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
        validate_default=True,
    )

    plane: Box = pd.Field(
        None,
        title="Plane",
        description="Cross-sectional plane in which the mode will be computed. "
        "If ``None``, the simulation must be 2D, and the plane will be the entire "
        "simulation geometry.",
    )

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    @pd.validator("plane", always=True)
    def is_plane(cls, val, values):
        """Raise validation error if not planar."""
        if val is None:
            val = Box(size=values["size"], center=values["center"])
            if val.size.count(0.0) != 1:
                raise ValidationError(
                    "If the 'ModeSimulation' geometry is not planar, "
                    "then 'plane' must be specified."
                )
            return val
        if val.size.count(0.0) != 1:
            raise ValidationError(f"'ModeSimulation' 'plane' must be planar, given 'size={val}'")
        return val

    @pd.validator("plane", always=True)
    @skip_if_fields_missing(["center", "size"])
    def plane_in_sim_bounds(cls, val, values):
        """Check that the plane is at least partially inside the simulation bounds."""
        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        if not sim_box.intersects(val):
            raise SetupError("'ModeSimulation.plane' must intersect 'ModeSimulation.geometry'.")
        return val

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.plane.normal_axis

    @cached_property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        return self._solver_symmetry(plane=self.plane)

    @pd.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        # this is handled instead post-init to ensure freqs is defined
        return val

    @cached_property
    def grid(self) -> Grid:
        """Grid spatial locations and information as defined by `grid_spec`.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

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
            plane = self.plane
            sources.append(
                PointDipole(
                    center=plane.center,
                    source_time=GaussianPulse(freq0=freqs[0], fwidth=0.1 * freqs[0]),
                    polarization="Ez",
                )
            )

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=sources,
            num_pml_layers=self.num_pml_layers,
        )

        return grid

    def _to_fdtd_sim(self) -> Simulation:
        """Convert :class:`.ModeSimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""

        # source to silence warnings
        # note that freqs[0] is fine for autogrid, since we already validate that
        # wavelength is provided to autogrid if freqs are not consistent
        freq0 = self.freqs[0]
        source_time = GaussianPulse(freq0=freq0, fwidth=0.1 * freq0)
        source = PointDipole(
            center=self.plane.center,
            source_time=source_time,
            polarization="Ez",
        )
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
            run_time=RUN_TIME,
            sources=[source],
            monitors=[],
        )

    def _post_init_validators(self) -> None:
        """Call validators taking `self` that get run after init."""
        _ = self.grid
        log.begin_capture()
        self._validate_symmetry()
        # self._validate_boundary_spec()
        log.end_capture(self)

    # this is not used because we want to allow PML boundary
    def _validate_boundary_spec(self):
        """Disallow boundary."""
        if self.boundary_spec != BoundarySpec.all_sides(PECBoundary()):
            log.warning(
                "In a 'ModeSimulation', the 'boundary_spec' is ignored and taken to be "
                "`PECBoundary` on all sides (the default value). The current value is "
                f"'{self.boundary_spec}'."
                "The field 'mode_spec' can be used to add pml layers."
            )

    def _validate_symmetry(self):
        """Symmetry in normal direction is not supported."""
        if self.symmetry[self.normal_axis] != 0:
            raise SetupError("Symmetry along normal axis is not supported.")

    def _validate_modes_size(self):
        """Make sure that the total size of the modes fields is not too large."""
        monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        num_cells = self._monitor_num_cells(monitor)
        # size in GB
        total_size = monitor._storage_size_solver(num_cells=num_cells, tmesh=[]) / 1e9
        if total_size > MAX_MODES_DATA_SIZE_GB:
            raise SetupError(
                f"Mode solver has {total_size:.2f}GB of estimated storage, "
                f"a maximum of {MAX_MODES_DATA_SIZE_GB:.2f}GB is allowed. Consider making the "
                "mode plane smaller, or decreasing the resolution or number of requested "
                "frequencies or modes."
            )

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pd.NonNegativeInt = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`ModeSimulation` instance plus additional
        specifications.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode solver.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.ModeSource`
            Mode source with specifications taken from the ModeSimulation instance and the method
            inputs.
        """

        if direction is None:
            direction = self.direction

        return ModeSource(
            center=self.plane.center,
            size=self.plane.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=direction,
        )

    def to_monitor(self, freqs: List[float] = None, name: str = None) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`ModeSimulation` instance plus additional
        specifications.

        Parameters
        ----------
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
            If not specified, passes ``self.freqs``.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.ModeMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        if freqs is None:
            freqs = self.freqs

        if name is None:
            raise ValueError(
                "A 'name' must be passed to 'ModeSimulation.to_monitor'. "
                "The default value of 'None' is for backwards compatibility and is not accepted."
            )

        return ModeMonitor(
            center=self.plane.center,
            size=self.plane.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=name,
        )

    def to_mode_solver_monitor(self, name: str, colocate: bool = None) -> ModeSolverMonitor:
        """Creates :class:`.ModeSolverMonitor` from a :class:`.ModeSimulation` instance.

        Parameters
        ----------
        name : str
            Name of the monitor.
        colocate : bool
            Whether to colocate fields or compute on the Yee grid. If not provided, the value
            set in the :class:`ModeSimulation` instance is used.

        Returns
        -------
        :class:`.ModeSolverMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and ``name``.
        """

        if colocate is None:
            colocate = self.colocate

        return ModeSolverMonitor(
            size=self.plane.size,
            center=self.plane.center,
            mode_spec=self.mode_spec,
            freqs=self.freqs,
            direction=self.direction,
            colocate=colocate,
            name=name,
        )

    def validate_pre_upload(self, source_required: bool = False):
        self._validate_modes_size()

    @classmethod
    def from_simulation(
        cls,
        simulation: AbstractYeeGridSimulation,
        plane: Box,
        mode_spec: ModeSpec,
        freqs: FreqArray,
        grid_spec: Union[GridSpec, Literal["identical"]] = "identical",
        reduced: bool = False,
        **kwargs,
    ) -> ModeSimulation:
        """Creates :class:`.ModeSimulation` from a :class:`.AbstractYeeGridSimulation`.

        Parameters
        ----------
        simulation: :class:`.AbstractYeeGridSimulation`
            Starting simulation defining structures, grid, etc.
        plane: :class:`.Box`
            Plane defining the mode solving region inside the simulation.
        mode_spec: :class:`.ModeSpec`
            Mode spec defining parameters for the mode solving.
        freqs: FreqArray
            Frequencies at which to solve for modes.
        grid_spec : :class:.`GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:.`CustomGrid`.
        reduced: bool
            Whether to strip objects not used by the mode solver from the simulation object.
            This might significantly reduce upload time in the presence of custom mediums.
        **kwargs
            Other arguments passed to new mode simulation instance.

        Returns
        -------
        :class:`.ModeSimulation`
            Mode simulation meant to reproduce the material properties of the original simulation.
            If ``reduced`` is ``True``, then the mode simulation may be smaller and contain fewer
            objects than the original simulation.
        """
        symmetry = plane.unpop_axis(
            0, simulation._solver_symmetry(plane=plane), axis=plane.normal_axis
        )

        # grid spec inheritace
        if grid_spec is None:
            grid_spec = simulation.grid_spec
        elif isinstance(grid_spec, str) and grid_spec == "identical":
            # create a custom grid from existing one
            grid_spec = GridSpec.from_grid(simulation.grid)

        # this logic was needed to get the grid right when we didn't allow PML boundary
        # but sometimes we still need to extend mode solver into PML for consistency with FDTD
        ## we preserve extra cells to ensure there is enough data for
        ## subpixel
        # extended_grid = simulation._get_solver_grid(
        #    plane=simulation.geometry, keep_additional_layers=True, truncate_symmetry=False
        # )
        # grids_1d = extended_grid.boundaries
        # rmin = [grids_1d.x[0], grids_1d.y[0], grids_1d.z[0]]
        # rmax = [grids_1d.x[-1], grids_1d.y[-1], grids_1d.z[-1]]
        ## truncate if not PML
        # new_bounds = Box.bounds_intersection(simulation.simulation_bounds, (rmin, rmax))
        # new_sim_box = Box.from_bounds(*new_bounds)

        new_sim_box = simulation.geometry

        # kwargs can overwrite the pre-defined kwargs
        mode_sim = cls(
            **(
                dict(
                    center=new_sim_box.center,
                    size=new_sim_box.size,
                    medium=simulation.medium,
                    structures=simulation.structures,
                    symmetry=symmetry,
                    grid_spec=grid_spec,
                    boundary_spec=simulation.boundary_spec,
                    version=simulation.version,
                    subpixel=simulation.subpixel,
                    plane=plane,
                    mode_spec=mode_spec,
                    freqs=freqs,
                )
                | kwargs
            )
        )

        if reduced:
            mode_sim = mode_sim.reduced_simulation_copy

        return mode_sim

    def subsection(
        self,
        region: Box,
        grid_spec: Union[GridSpec, Literal["identical"]] = None,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry] = None,
        remove_outside_structures: bool = True,
        remove_outside_custom_mediums: bool = False,
        **kwargs,
    ) -> ModeSimulation:
        """Generate a simulation instance containing only the ``region``.

        Parameters
        ----------
        region : :class:.`Box`
            New simulation domain.
        grid_spec : :class:.`GridSpec` = None
            New grid specification. If ``None``, then it is inherited from the original
            simulation. If ``identical``, then the original grid is transferred directly as a
            :class:.`CustomGrid`. Note that in the latter case the region of the new simulation is
            snapped to the original grid lines.
        symmetry : Tuple[Literal[0, -1, 1], Literal[0, -1, 1], Literal[0, -1, 1]] = None
            New simulation symmetry. If ``None``, then it is inherited from the original
            simulation. Note that in this case the size and placement of new simulation domain
            must be commensurate with the original symmetry.
        remove_outside_structures : bool = True
            Remove structures outside of the new simulation domain.
        remove_outside_custom_mediums : bool = True
            Remove custom medium data outside of the new simulation domain.
        **kwargs
            Other arguments passed to new simulation instance.
        """

        new_sim = super().subsection(
            region=region,
            grid_spec=grid_spec,
            symmetry=symmetry,
            remove_outside_structures=remove_outside_structures,
            remove_outside_custom_mediums=remove_outside_custom_mediums,
            **kwargs,
        )

        return new_sim

    @cached_property
    def reduced_simulation_copy(self):
        """Strip objects not used by the mode solver from simulation object.
        This might significantly reduce upload time in the presence of custom mediums.
        """
        return super()._reduced_simulation_copy(plane=self.plane)

    def sim_with_source(
        self,
        sim: Simulation,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pd.NonNegativeInt = 0,
    ) -> Simulation:
        """Creates a copy of the provided :class:`.Simulation` with a
        :class:`.ModeSource` added corresponding to the :class:`.ModeSimulation` parameters.

        Parameters
        ----------
        sim: :class:`.Simulation`
            FDTD simulation to add the mode source to.
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode simulation.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSource` with specifications taken
            from the ModeSimulation instance and the method inputs.
        """

        mode_source = self.to_source(
            mode_index=mode_index, direction=direction, source_time=source_time
        )
        new_sources = list(sim.sources) + [mode_source]
        new_sim = sim.updated_copy(sources=new_sources)
        return new_sim

    def sim_with_monitor(
        self,
        sim: Simulation,
        freqs: List[float] = None,
        name: str = None,
    ) -> Simulation:
        """Creates a copy of the provided :class:`.Simulation` with a
        :class:`.ModeMonitor` added corresponding to the :class:`.ModeSimulation` parameters.

        Parameters
        ----------
        sim: :class:`.Simulation`
            FDTD simulation to add the mode source to.
        freqs : List[float] = None
            Frequencies to include in Monitor (Hz).
            If not specified, uses the frequencies from the mode solver.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeMonitor` with specifications taken
            from the ModeSolver instance and the method inputs.
        """

        mode_monitor = self.to_monitor(freqs=freqs, name=name)
        new_monitors = list(sim.monitors) + [mode_monitor]
        new_sim = sim.updated_copy(monitors=new_monitors)
        return new_sim

    def sim_with_mode_solver_monitor(
        self,
        sim: AbstractYeeGridSimulation,
        name: str,
    ) -> AbstractYeeGridSimulation:
        """Creates a copy of the provided :class:`.Simulation` with a
        :class:`.ModeSolverMonitor` added corresponding to the :class:`.ModeSimulation` parameters.

        Parameters
        ----------
        sim: :class:`.AbstractYeeGridSimulation`
            FDTD simulation to add the mode source to.
        name : str
            Name of the monitor.

        Returns
        -------
        :class:`.AbstractYeeGridSimulation`
            Copy of the simulation with a :class:`.ModeSolverMonitor` with specifications taken
            from the ModeSimulation instance and ``name``.
        """
        mode_solver_monitor = self.to_mode_solver_monitor(name=name)
        new_monitors = list(sim.monitors) + [mode_solver_monitor]
        new_sim = sim.updated_copy(monitors=new_monitors)
        return new_sim

    @cached_property
    def _solver_grid(self) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and also with
        a small correction for symmetries. We don't do the snapping yet because 0-sized cells are
        currently confusing to the subpixel averaging. The final data coordinates along the
        plane normal dimension and dimensions where the simulation domain is 2D will be correctly
        set after the solve."""

        return self._get_solver_grid(
            plane=self.plane, keep_additional_layers=False, truncate_symmetry=True
        )

    @cached_property
    def _num_cells_freqs_modes(self) -> Tuple[int, int, int]:
        """Get the number of spatial points, number of freqs, and number of modes requested."""
        num_cells = np.prod(self._solver_grid.num_cells)
        num_modes = self.mode_spec.num_modes
        num_freqs = len(self.freqs)
        return num_cells, num_freqs, num_modes

    def _freqs_for_group_index(self) -> FreqArray:
        """Get frequencies used to compute group index."""
        f_step = self.mode_spec.group_index_step
        fractional_steps = (1 - f_step, 1, 1 + f_step)
        return np.outer(self.freqs, fractional_steps).flatten()

    def _remove_freqs_for_group_index(self) -> FreqArray:
        """Remove frequencies used to compute group index.

        Returns
        -------
        FreqArray
            Filtered frequency array with only original values.
        """
        return np.array(self.freqs[1 : len(self.freqs) : 3])

    @cached_property
    def grid_snapped(self) -> Grid:
        """The solver grid snapped to the plane normal and to simulation 0-sized dims if any."""
        grid_snapped = self._solver_grid.snap_to_box_zero_dim(self.plane)
        return self._snap_zero_dim(grid_snapped)

    def _get_epsilon(self, freq: float) -> ArrayComplex4D:
        """Compute the epsilon tensor in the plane. Order of components is xx, xy, xz, yx, etc."""
        eps_keys = ["Ex", "Exy", "Exz", "Eyx", "Ey", "Eyz", "Ezx", "Ezy", "Ez"]
        eps_tensor = [self.epsilon_on_grid(self._solver_grid, key, freq) for key in eps_keys]
        return np.stack(eps_tensor, axis=0)

    def _tensorial_material_profile_modal_plane_tranform(
        self, mat_data: ArrayComplex4D
    ) -> ArrayComplex4D:
        """For tensorial material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        # get rid of normal axis
        mat_tensor = np.take(mat_data, indices=[0], axis=1 + self.normal_axis)
        mat_tensor = np.squeeze(mat_tensor, axis=1 + self.normal_axis)

        # convert to into 3-by-3 representation for easier axis swap
        flat_shape = np.shape(mat_tensor)  # 9 components flat
        tensor_shape = [3, 3] + list(flat_shape[1:])  # 3-by-3 matrix
        mat_tensor = mat_tensor.reshape(tensor_shape)

        # swap axes to plane coordinates (normal_axis goes to z)
        if self.normal_axis == 0:
            # swap x and y
            mat_tensor[[0, 1], :, ...] = mat_tensor[[1, 0], :, ...]
            mat_tensor[:, [0, 1], ...] = mat_tensor[:, [1, 0], ...]
        if self.normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            mat_tensor[[1, 2], :, ...] = mat_tensor[[2, 1], :, ...]
            mat_tensor[:, [1, 2], ...] = mat_tensor[:, [2, 1], ...]

        # back to "flat" representation
        mat_tensor = mat_tensor.reshape(flat_shape)

        # construct to feed to mode solver
        return mat_tensor

    def _diagonal_material_profile_modal_plane_tranform(
        self, mat_data: ArrayComplex4D
    ) -> ArrayComplex3D:
        """For diagonal material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        # get rid of normal axis
        mat_tensor = np.take(mat_data, indices=[0], axis=1 + self.normal_axis)
        mat_tensor = np.squeeze(mat_tensor, axis=1 + self.normal_axis)

        # swap axes to plane coordinates (normal_axis goes to z)
        if self.normal_axis == 0:
            # swap x and y
            mat_tensor[[0, 1], :, ...] = mat_tensor[[1, 0], :, ...]
        if self.normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            mat_tensor[[1, 2], :, ...] = mat_tensor[[2, 1], :, ...]

        # construct to feed to mode solver
        return mat_tensor

    def _solver_eps(self, freq: float) -> ArrayComplex4D:
        """Diagonal permittivity in the shape needed by solver, with normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        eps_tensor = self._get_epsilon(freq)
        # tranformation
        return self._tensorial_material_profile_modal_plane_tranform(eps_tensor)

    @property
    def _is_tensorial(self) -> bool:
        """Whether the mode computation should be fully tensorial. This is either due to fully
        anisotropic media, or due to an angled waveguide, in which case the transformed eps and mu
        become tensorial. A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to invoke the tensorial solver, so
        the actual behavior may differ from what's predicted by this property."""
        return abs(self.mode_spec.angle_theta) > 0 or self._has_fully_anisotropic_media

    @cached_property
    def _intersecting_media(self) -> List:
        """List of media (including simulation background) intersecting the mode plane."""
        total_structures = [self.scene.background_structure]
        total_structures += list(self.structures)
        return self.scene.intersecting_media(self.plane, total_structures)

    @cached_property
    def _has_fully_anisotropic_media(self) -> bool:
        """Check if there are any fully anisotropic media in the plane of the mode."""
        if np.any([isinstance(mat, FullyAnisotropicMedium) for mat in self.scene.mediums]):
            for int_mat in self._intersecting_media:
                if isinstance(int_mat, FullyAnisotropicMedium):
                    return True
        return False

    @cached_property
    def _has_complex_eps(self) -> bool:
        """Check if there are media with a complex-valued epsilon in the plane of the mode.
        A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to use real or complex fields, so
        the actual behavior may differ from what's predicted by this property."""
        check_freqs = np.unique([np.amin(self.freqs), np.amax(self.freqs), np.mean(self.freqs)])
        for int_mat in self._intersecting_media:
            for freq in check_freqs:
                max_imag_eps = np.amax(np.abs(np.imag(int_mat.eps_model(freq))))
                if not isclose(max_imag_eps, 0):
                    return False
        return True

    def _center_and_lims(self) -> Tuple[List, List, List, List]:
        """Get the mode plane center and limits."""

        n_axis, t_axes = self.plane.pop_axis([0, 1, 2], self.normal_axis)
        a_center = [None, None, None]
        a_center[n_axis] = self.plane.center[n_axis]

        _, (h_min_s, v_min_s) = Box.pop_axis(self.bounds[0], axis=n_axis)
        _, (h_max_s, v_max_s) = Box.pop_axis(self.bounds[1], axis=n_axis)

        h_min = a_center[n_axis] - self.plane.size[t_axes[0]] / 2
        h_max = a_center[n_axis] + self.plane.size[t_axes[0]] / 2
        v_min = a_center[n_axis] - self.plane.size[t_axes[1]] / 2
        v_max = a_center[n_axis] + self.plane.size[t_axes[1]] / 2

        h_lim = [
            h_min if abs(h_min) < abs(h_min_s) else h_min_s,
            h_max if abs(h_max) < abs(h_max_s) else h_max_s,
        ]
        v_lim = [
            v_min if abs(v_min) < abs(v_min_s) else v_min_s,
            v_max if abs(v_max) < abs(v_max_s) else v_max_s,
        ]

        return a_center, h_lim, v_lim, t_axes
