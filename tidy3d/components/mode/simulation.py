"""Defines mode solver simulation class."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...constants import C_0
from ...exceptions import SetupError, ValidationError
from ...log import log
from ..base import cached_property
from ..boundary import Boundary, BoundarySpec
from ..geometry.base import Box
from ..grid.grid import Grid
from ..grid.grid_spec import GridSpec
from ..mode_spec import ModeSpec
from ..monitor import ModeMonitor, ModeSolverMonitor, PermittivityMonitor
from ..simulation import AbstractYeeGridSimulation, Simulation, validate_boundaries_for_zero_dims
from ..source.field import ModeSource
from ..types import (
    TYPE_TAG_STR,
    Ax,
    Direction,
    EMField,
    FreqArray,
)
from ..validators import validate_mode_plane_radius
from .mode_solver import ModeSolver

ModeSimulationMonitorType = PermittivityMonitor

# dummy run time for conversion to FDTD sim
# should be very small -- otherwise, generating tmesh will fail or take a long time
RUN_TIME = 1e-30

MODE_PLANE_TYPE = Union[Box, ModeSource, ModeMonitor, ModeSolverMonitor]


# attributes shared between ModeSimulation class and ModeSolver class
MODE_SIM_MODE_SOLVER_SHARED_ATTRS = [
    "plane",
    "mode_spec",
    "freqs",
    "direction",
    "colocate",
    "fields",
]
# attributes shared between ModeSimulation class and AbstractYeeGridSimulation
# excludes things like monitors and sources which are not present in a ModeSimulation
MODE_SIM_YEE_SIM_SHARED_ATTRS = [
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


class ModeSimulation(AbstractYeeGridSimulation):
    """
    Simulation class for solving electromagnetic eigenmodes in a 2D plane with
    translational invariance in the third dimension. If the field ``plane`` is
    specified, the domain for mode solving is the intersection between the ``plane``
    and the simulation geometry. If the simulation geometry is 2D (has zero size
    in one dimension) and the ``plane`` is ``None``, then the domain for mode solving
    is the entire 2D geometry.

    The ``symmetry`` field can be used to enforce reflection symmetry across planes
    through the ``center`` of the simulation. Each component of the ``symmetry`` field
    is only used if the ``center`` of the ``plane`` and the simulation geometry
    coincide in that component. Symmetry normal to the mode solving domain has no
    effect; the field ``filter_pol`` in :class:`.ModeSpec` can be used here instead.

    Example
    -------
    >>> from tidy3d import C_0, ModeSpec, BoundarySpec, Boundary
    >>> lambda0 = 1550e-9
    >>> freq0 = C_0 / lambda0
    >>> freqs = [freq0]
    >>> sim_size = lambda0, lambda0, 0
    >>> mode_spec = ModeSpec(num_modes=4)
    >>> boundary_spec = BoundarySpec(
    ...     x=Boundary.pec(),
    ...     y=Boundary.pec(),
    ...     z=Boundary.periodic()
    ... )
    >>> sim = ModeSimulation(
    ...     size=sim_size,
    ...     freqs=freqs,
    ...     mode_spec=mode_spec,
    ...     boundary_spec=boundary_spec
    ... )

    See Also
    --------

    :class:`ModeSource`:
        Injects current source to excite modal profile on finite extent plane.

    **Notebooks:**
        * `Waveguide Y junction <../../notebooks/YJunction.html>`_
        * `Photonic crystal waveguide polarization filter <../../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_

    **Lectures:**
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_
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

    fields: Tuple[EMField, ...] = pd.Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor. Note that some "
        "methods like ``flux``, ``dot`` require all tangential field components, while others "
        "like ``mode_area`` require all E-field components.",
    )

    boundary_spec: BoundarySpec = pd.Field(
        BoundarySpec(),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides. This behavior is for "
        "consistency with FDTD simulations; however, please note that the mode solver "
        "terminates the mode plane with PEC boundary. The 'ModeSpec' can be used to "
        "apply PML layers in the mode solver.",
    )

    monitors: Tuple[ModeSimulationMonitorType, ...] = pd.Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )

    sources: Tuple[()] = pd.Field(
        (),
        title="Sources",
        description="Sources in the simulation. Note: sources are not supported in mode "
        "simulations.",
    )

    grid_spec: GridSpec = pd.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )

    plane: MODE_PLANE_TYPE = pd.Field(
        None,
        title="Plane",
        description="Cross-sectional plane in which the mode will be computed. "
        "If provided, the computational domain will be the intersection between "
        "the provided ``plane`` and the simulation geometry. "
        "If ``None``, the simulation must be 2D, and the plane will be the entire "
        "simulation geometry.",
        discriminator=TYPE_TAG_STR,
    )

    @pd.validator("plane", always=True)
    def is_plane(cls, val, values):
        """Raise validation error if not planar."""
        if val is None:
            sim_center = values.get("center")
            sim_size = values.get("size")
            val = Box(size=sim_size, center=sim_center)
            if val.size.count(0.0) != 1:
                raise ValidationError(
                    "If the 'ModeSimulation' geometry is not planar, "
                    "then 'plane' must be specified."
                )
            return val
        if val.size.count(0.0) != 1:
            raise ValidationError(f"'ModeSimulation.plane' must be planar, given 'size={val}'")
        return val

    @pd.validator("plane", always=True)
    def plane_in_sim_bounds(cls, val, values):
        """Check that the plane is at least partially inside the simulation bounds."""
        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        if not sim_box.intersects(val):
            raise SetupError("'ModeSimulation.plane' must intersect 'ModeSimulation.geometry.")
        return val

    @pd.validator("boundary_spec", always=True)
    def boundaries_for_zero_dims(cls, val, values):
        """Replace with periodic boundary along zero-size dimensions."""
        boundaries = [val.x, val.y, val.z]
        size = values.get("size")

        for dim, size_dim in enumerate(size):
            if size_dim == 0:
                boundaries[dim] = Boundary.periodic()

        return BoundarySpec(x=boundaries[0], y=boundaries[1], z=boundaries[2])

    def _post_init_validators(self) -> None:
        """Call validators taking `self` that get run after init."""
        validate_mode_plane_radius(
            mode_spec=self.mode_spec, plane=self.plane, msg_prefix="'ModeSimulation'"
        )
        _ = self._mode_solver
        _ = self.grid

    @pd.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Handle the case where grid_spec is auto and wavelength is not provided."""
        # this is handled instead post-init to ensure freqs is defined
        return val

    @cached_property
    def _mode_solver(self) -> ModeSolver:
        """Convert the :class:`.ModeSimulation` to a :class:`.ModeSolver`."""
        kwargs = {key: getattr(self, key) for key in MODE_SIM_MODE_SOLVER_SHARED_ATTRS}
        return ModeSolver(simulation=self._as_fdtd_sim, **kwargs)

    def run_local(self):
        """Run locally."""
        from .data.sim_data import ModeSimulationData

        modes_raw = self._mode_solver.data_raw
        return ModeSimulationData(simulation=self, modes_raw=modes_raw)

    @cached_property
    def grid(self) -> Grid:
        """Grid spatial locations and information as defined by `grid_spec`.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """
        return self._as_fdtd_sim.grid

    @cached_property
    def _as_fdtd_sim(self) -> Simulation:
        """Convert :class:`.ModeSimulation` to :class:`.Simulation`.
        This should only be used to obtain the same material properties
        for mode solving or related purposes; the sources and monitors of the
        resulting simulation are not meaningful."""

        grid_spec = self.grid_spec
        if grid_spec.auto_grid_used and grid_spec.wavelength is None:
            min_wvl = C_0 / np.max(self.freqs)
            log.info(
                f"Auto meshing using wavelength {min_wvl:1.4f} defined from "
                "largest of 'ModeSimulation.freqs'."
            )
            grid_spec = grid_spec.updated_copy(wavelength=min_wvl)

        kwargs = {key: getattr(self, key) for key in MODE_SIM_YEE_SIM_SHARED_ATTRS}
        return Simulation(
            **kwargs,
            run_time=RUN_TIME,
            grid_spec=grid_spec,
            monitors=[],
        )

    @classmethod
    def from_simulation(
        cls,
        simulation: AbstractYeeGridSimulation,
        wavelength: Optional[pd.PositiveFloat] = None,
        **kwargs,
    ) -> ModeSimulation:
        """Creates :class:`.ModeSimulation` from a :class:`.AbstractYeeGridSimulation`.

        Parameters
        ----------
        simulation: :class:`.AbstractYeeGridSimulation`
            Starting simulation defining structures, grid, etc.
        wavelength: Optional[pd.PositiveFloat]
            Wavelength used for automatic grid generation. Required if auto grid
            is used in ``grid_spec``.
        **kwargs
            Other arguments passed to new mode simulation instance.

        Returns
        -------
        :class:`.ModeSimulation`
            Mode simulation reproducing the material and grid properties
            of the original simulation.
        """
        if isinstance(simulation, ModeSimulation):
            return simulation.updated_copy(**kwargs)

        grid_spec = simulation.grid_spec
        if grid_spec.auto_grid_used:
            if wavelength is None:
                if grid_spec.wavelength is None:
                    raise ValidationError(
                        "Automatic grid generation requires specifying "
                        "'wavelength' or 'grid_spec.wavelength'."
                    )
            else:
                if grid_spec.wavelength is not None and grid_spec.wavelength != wavelength:
                    log.warning("Replacing 'grid_spec.wavelength' with provided 'wavelength'.")
                grid_spec = grid_spec.updated_copy(wavelength=wavelength)

        sim_kwargs = {key: getattr(simulation, key) for key in MODE_SIM_YEE_SIM_SHARED_ATTRS}

        mode_sim = cls(
            grid_spec=grid_spec,
            **sim_kwargs,
            **kwargs,
        )

        return mode_sim

    @cached_property
    def reduced_simulation_copy(self) -> ModeSimulation:
        """Strip objects not used by the mode solver from simulation object.
        This might significantly reduce upload time in the presence of custom mediums.
        """
        reduced_mode_solver = self._mode_solver.reduced_simulation_copy
        return self.from_mode_solver(reduced_mode_solver)

    @classmethod
    def from_mode_solver(
        cls, mode_solver: ModeSolver, wavelength: Optional[pd.PositiveFloat] = None
    ) -> ModeSimulation:
        """Creates :class:`.ModeSimulation` from a :class:`.ModeSolver`.

        Parameters
        ----------
        simulation: :class:`.AbstractYeeGridSimulation`
            Starting simulation defining structures, grid, etc.
        wavelength: Optional[pd.PositiveFloat]
            Wavelength used for automatic grid generation. Required if auto grid
            is used in ``grid_spec``.

        Returns
        -------
        :class:`.ModeSimulation`
            Mode simulation reproducing the material and grid properties
            of the original simulation.
        """

        kwargs = {key: getattr(mode_solver, key) for key in MODE_SIM_MODE_SOLVER_SHARED_ATTRS}
        mode_sim = cls.from_simulation(
            simulation=mode_solver.simulation, wavelength=wavelength, **kwargs
        )
        return mode_sim

    def plot_mode_plane(
        self,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot the mode plane simulation's components.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Materials <../../notebooks/VizSimulation.html#Plotting-Materials>`_

        """
        return self._mode_solver.plot(ax=ax, **patch_kwargs)

    def plot_eps_mode_plane(
        self,
        freq: float = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane simulation's components.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """
        return self._mode_solver.plot_eps(freq=freq, alpha=alpha, ax=ax)

    def plot_structures_eps_mode_plane(
        self,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane simulation's components.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """
        return self._mode_solver.plot_structures_eps(
            freq=freq, alpha=alpha, cbar=cbar, reverse=reverse, ax=ax
        )

    def plot_grid_mode_plane(
        self,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the mode plane cell boundaries as lines.

        Parameters
        ----------
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
        return self._mode_solver.plot_grid(ax=ax)

    def plot_pml_mode_plane(
        self,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane absorbing boundaries.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self._mode_solver.plot_pml(ax=ax)

    def validate_pre_upload(self, source_required: bool = False):
        self._mode_solver.validate_pre_upload(source_required=source_required)

    _boundaries_for_zero_dims = validate_boundaries_for_zero_dims()
