"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from __future__ import annotations

from functools import wraps
from typing import Dict, List, Tuple, Union

import pydantic.v1 as pydantic
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ...components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ...components.data.data_array import (
    FreqModeDataArray,
    ModeIndexDataArray,
)
from ...components.data.monitor_data import ModeSolverData
from ...components.data.sim_data import SimulationData
from ...components.eme.data.sim_data import EMESimulationData
from ...components.eme.simulation import EMESimulation
from ...components.geometry.base import Box
from ...components.grid.grid import Grid
from ...components.mode.data.sim_data import ModeSimulationData
from ...components.mode.run_local import ModeSimulationSolver
from ...components.mode.simulation import ModeSimulation
from ...components.mode_spec import ModeSpec
from ...components.monitor import ModeMonitor, ModeSolverMonitor
from ...components.simulation import Simulation
from ...components.source import ModeSource, SourceTime
from ...components.types import (
    TYPE_TAG_STR,
    ArrayComplex3D,
    ArrayComplex4D,
    ArrayFloat1D,
    Ax,
    Axis,
    Direction,
    EpsSpecType,
    FreqArray,
    Literal,
    PlotScale,
    Symmetry,
)
from ...components.validators import validate_freqs_min, validate_freqs_not_empty
from ...components.viz import plot_params_pml
from ...exceptions import SetupError, ValidationError

FIELD = Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
MODE_MONITOR_NAME = "<<<MODE_SOLVER_MONITOR>>>"

MODE_SIMULATION_TYPE = Union[Simulation, EMESimulation, ModeSimulation]
MODE_SIMULATION_DATA_TYPE = Union[SimulationData, EMESimulationData, ModeSimulationData]
MODE_PLANE_TYPE = Union[Box, ModeSource, ModeMonitor, ModeSolverMonitor]


def require_fdtd_simulation(fn):
    """Decorate a function to check that ``simulation`` is an FDTD ``Simulation``."""

    @wraps(fn)
    def _fn(self, **kwargs):
        """New decorated function."""
        if not isinstance(self.simulation, Simulation):
            raise SetupError(
                f"The function '{fn.__name__}' is only supported "
                "for 'simulation' of type FDTD 'Simulation'."
            )
        return fn(self, **kwargs)

    return _fn


class ModeSolver(Tidy3dBaseModel):
    """
    Interface for solving electromagnetic eigenmodes in a 2D plane with translational
    invariance in the third dimension.

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

    simulation: MODE_SIMULATION_TYPE = pydantic.Field(
        ...,
        title="Simulation",
        description="Simulation, EMESimulation, or ModeSimulation defining all structures and "
        "mediums.",
        discriminator="type",
    )

    plane: MODE_PLANE_TYPE = pydantic.Field(
        ...,
        title="Plane",
        description="Cross-sectional plane in which the mode will be computed.",
        discriminator=TYPE_TAG_STR,
    )

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode specification",
        description="Container with specifications about the modes to be solved for.",
    )

    freqs: FreqArray = pydantic.Field(
        ..., title="Frequencies", description="A list of frequencies at which to solve."
    )

    direction: Direction = pydantic.Field(
        "+",
        title="Propagation direction",
        description="Direction of waveguide mode propagation along the axis defined by its normal "
        "dimension.",
    )

    colocate: bool = pydantic.Field(
        True,
        title="Colocate fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default is ``True``.",
    )

    @pydantic.validator("plane", always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    @pydantic.validator("plane", always=True)
    @skip_if_fields_missing(["simulation"])
    def plane_in_sim_bounds(cls, val, values):
        """Check that the plane is at least partially inside the simulation bounds."""
        sim_center = values.get("simulation").center
        sim_size = values.get("simulation").size
        sim_box = Box(size=sim_size, center=sim_center)

        if not sim_box.intersects(val):
            raise SetupError("'ModeSolver.plane' must intersect 'ModeSolver.simulation'.")
        return val

    @property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.reduced_mode_sim.normal_axis

    @property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        return self.reduced_mode_sim.solver_symmetry

    def _get_solver_grid(
        self, keep_additional_layers: bool = False, truncate_symmetry: bool = True
    ) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and optionally
        corrected for symmetries.

        Parameters
        ----------
        keep_additional_layers : bool = False
            Do not discard layers of cells in front and behind the main layer of cells. Together they
            represent the region where custom medium data is needed for proper subpixel.
        truncate_symmetry : bool = True
            Truncate to symmetry quadrant if symmetry present.

        Returns
        -------
        :class:.`Grid`
            The resulting grid.
        """
        return self.reduced_mode_sim._get_solver_grid(
            plane=self.plane,
            keep_additional_layers=keep_additional_layers,
            truncate_symmetry=truncate_symmetry,
        )

    @property
    def _solver_grid(self) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and also with
        a small correction for symmetries. We don't do the snapping yet because 0-sized cells are
        currently confusing to the subpixel averaging. The final data coordinates along the
        plane normal dimension and dimensions where the simulation domain is 2D will be correctly
        set after the solve."""

        return self.reduced_mode_sim._solver_grid

    @property
    def _num_cells_freqs_modes(self) -> Tuple[int, int, int]:
        """Get the number of spatial points, number of freqs, and number of modes requested."""
        return self.reduced_mode_sim._num_cells_freqs_modes

    def solve(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        return self.data

    def _freqs_for_group_index(self) -> FreqArray:
        """Get frequencies used to compute group index."""
        return self.reduced_mode_sim._freqs_for_group_index()

    def _remove_freqs_for_group_index(self) -> FreqArray:
        """Remove frequencies used to compute group index.

        Returns
        -------
        FreqArray
            Filtered frequency array with only original values.
        """
        return self.reduced_mode_sim._remove_freqs_for_group_index()

    def _get_data_with_group_index(self) -> ModeSolverData:
        """:class:`.ModeSolverData` with fields, effective and group indices on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective and group indices, and mode
            fields.
        """

        return self.mode_sim_solver._get_data_with_group_index()

    @property
    def grid_snapped(self) -> Grid:
        """The solver grid snapped to the plane normal and to simulation 0-sized dims if any."""
        return self.reduced_mode_sim.grid_snapped

    @cached_property
    def data_raw(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        return self.mode_sim_solver.data_raw

    def _data_on_yee_grid(self) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        return self.mode_sim_solver._data_on_yee_grid()

    def _data_on_yee_grid_relative(self, basis: ModeSolverData) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        return self.mode_sim_solver._data_on_yee_grid_relative(basis=basis)

    def _colocate_data(self, mode_solver_data: ModeSolverData) -> ModeSolverData:
        """Colocate data to Yee grid boundaries."""
        return self.mode_sim_solver._colocate_data(mode_solver_data=mode_solver_data)

    def _normalize_modes(self, mode_solver_data: ModeSolverData):
        """Normalize modes. Note: this modifies ``mode_solver_data`` in-place."""
        return self.mode_sim_solver._normalize_modes(mode_solver_data=mode_solver_data)

    def _filter_polarization(self, mode_solver_data: ModeSolverData):
        """Filter polarization. Note: this modifies ``mode_solver_data`` in-place."""
        return self.mode_sim_solver._filter_polarization(mode_solver_data=mode_solver_data)

    @cached_property
    def data(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        mode_solver_data = self.data_raw
        return mode_solver_data.symmetry_expanded_copy

    @property
    def sim_data(self) -> MODE_SIMULATION_DATA_TYPE:
        """:class:`.SimulationData` object containing the :class:`.ModeSolverData` for this object.

        Returns
        -------
        SimulationData
            :class:`.SimulationData` object containing the effective index and mode fields.
        """
        if isinstance(self.simulation, ModeSimulation):
            return ModeSimulationData(
                simulation=self.simulation, data_raw=self.data_raw, remote=False
            )
        monitor_data = self.data
        new_monitors = list(self.simulation.monitors) + [monitor_data.monitor]
        new_simulation = self.simulation.copy(update=dict(monitors=new_monitors))
        if isinstance(new_simulation, Simulation):
            return SimulationData(simulation=new_simulation, data=(monitor_data,))
        elif isinstance(new_simulation, EMESimulation):
            return EMESimulationData(
                simulation=new_simulation, data=(monitor_data,), smatrix=None, port_modes=None
            )
        else:
            raise SetupError(
                "The 'simulation' provided does not correspond to any known "
                "'AbstractSimulationData' type."
            )

    def _get_epsilon(self, freq: float) -> ArrayComplex4D:
        """Compute the epsilon tensor in the plane. Order of components is xx, xy, xz, yx, etc."""
        return self.reduced_mode_sim._get_epsilon(freq=freq)

    def _tensorial_material_profile_modal_plane_tranform(
        self, mat_data: ArrayComplex4D
    ) -> ArrayComplex4D:
        """For tensorial material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        return self.reduced_mode_sim._tensorial_material_profile_modal_plane_transform(
            mat_data=mat_data
        )

    def _diagonal_material_profile_modal_plane_tranform(
        self, mat_data: ArrayComplex4D
    ) -> ArrayComplex3D:
        """For diagonal material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        return self.reduced_mode_sim._diagonal_material_profile_modal_plane_transform(
            mat_data=mat_data
        )

    def _solver_eps(self, freq: float) -> ArrayComplex4D:
        """Diagonal permittivity in the shape needed by solver, with normal axis rotated to z."""

        return self.reduced_mode_sim._solver_eps(freq=freq)

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""
        return self.reduced_mode_sim._solve_all_freqs(coords=coords, symmetry=symmetry)

    def _solve_all_freqs_relative(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
        basis_fields: List[Dict[str, ArrayComplex4D]],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""
        return self.reduced_mode_sim._solve_all_freqs_relative(
            coords=coords, symmetry=symmetry, basis_fields=basis_fields
        )

    def _postprocess_solver_fields(self, solver_fields):
        """Postprocess `solver_fields` from `compute_modes` to proper coordinate"""
        return self.reduced_mode_sim._postprocess_solver_fields(solver_fields=solver_fields)

    def _solve_single_freq(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.

        The fields are rotated from propagation coordinates back to global coordinates.
        """
        return self.reduced_mode_sim._solve_single_freq(freq=freq, coords=coords, symmetry=symmetry)

    def _rotate_field_coords_inverse(self, field: FIELD) -> FIELD:
        """Move the propagation axis to the z axis in the array."""
        return self.reduced_mode_sim._rotate_field_coords_inverse(field=field)

    def _postprocess_solver_fields_inverse(self, fields):
        """Convert ``fields`` to ``solver_fields``. Doesn't change gauge."""
        return self.reduced_mode_sim._postprocess_solver_fields_inverse(fields=fields)

    def _solve_single_freq_relative(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
        basis_fields: Dict[str, ArrayComplex4D],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.
        Modes are computed as linear combinations of ``basis_fields``.
        """
        return self.reduced_mode_sim._solve_single_freq_relative(
            freq=freq, coords=coords, symmetry=symmetry, basis_fields=basis_fields
        )

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        return self.reduced_mode_sim._rotate_field_coords(field=field)

    def _process_fields(
        self, mode_fields: ArrayComplex4D, mode_index: pydantic.NonNegativeInt
    ) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes and set gauge."""
        return self.reduced_mode_sim._process_fields(mode_fields=mode_fields, mode_index=mode_index)

    def _field_decay_warning(self, field_data: ModeSolverData):
        """Warn if any of the modes do not decay at the edges."""
        self.reduced_mode_sim._field_decay_warning(field_data=field_data)

    @staticmethod
    def _grid_correction(
        simulation: MODE_SIMULATION_TYPE,
        plane: Box,
        mode_spec: ModeSpec,
        n_complex: ModeIndexDataArray,
        direction: Direction,
    ) -> [FreqModeDataArray, FreqModeDataArray]:
        """Correct the fields due to propagation on the grid.

        Return a copy of the :class:`.ModeSolverData` with the fields renormalized to account
        for propagation on a finite grid along the propagation direction. The fields are assumed to
        have ``E exp(1j k r)`` dependence on the finite grid and are then resampled using linear
        interpolation to the exact position of the mode plane. This is needed to correctly compute
        overlap with fields that come from a :class:`.FieldMonitor` placed in the same grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Numerical grid on which the modes are assumed to propagate.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of the data with renormalized fields.
        """

        return ModeSimulationSolver._grid_correction(
            simulation=simulation,
            plane=plane,
            mode_spec=mode_spec,
            n_complex=n_complex,
            direction=direction,
        )

    @property
    def _is_tensorial(self) -> bool:
        """Whether the mode computation should be fully tensorial. This is either due to fully
        anisotropic media, or due to an angled waveguide, in which case the transformed eps and mu
        become tensorial. A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to invoke the tensorial solver, so
        the actual behavior may differ from what's predicted by this property."""
        return self.reduced_mode_sim._is_tensorial

    @property
    def _intersecting_media(self) -> List:
        """List of media (including simulation background) intersecting the mode plane."""
        return self.reduced_mode_sim._intersecting_media

    @property
    def _has_fully_anisotropic_media(self) -> bool:
        """Check if there are any fully anisotropic media in the plane of the mode."""
        return self.reduced_mode_sim._has_fully_anisotropic_media

    @property
    def _has_complex_eps(self) -> bool:
        """Check if there are media with a complex-valued epsilon in the plane of the mode.
        A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to use real or complex fields, so
        the actual behavior may differ from what's predicted by this property."""
        return self.reduced_mode_sim._has_complex_eps

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`ModeSolver` instance plus additional
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
            Mode source with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        return self.reduced_mode_sim.to_source(
            source_time=source_time, direction=direction, mode_index=mode_index
        )

    def to_monitor(self, freqs: List[float] = None, name: str = None) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`ModeSolver` instance plus additional
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

        return self.reduced_mode_sim.to_monitor(freqs=freqs, name=name)

    def to_mode_solver_monitor(self, name: str, colocate: bool = None) -> ModeSolverMonitor:
        """Creates :class:`ModeSolverMonitor` from a :class:`ModeSolver` instance.

        Parameters
        ----------
        name : str
            Name of the monitor.
        colocate : bool
            Whether to colocate fields or compute on the Yee grid. If not provided, the value
            set in the :class:`ModeSolver` instance is used.

        Returns
        -------
        :class:`.ModeSolverMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and ``name``.
        """

        return self.reduced_mode_sim.to_mode_solver_monitor(name=name, colocate=colocate)

    @require_fdtd_simulation
    def sim_with_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> Simulation:
        """Creates :class:`Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a ModeSource added corresponding to
        the ModeSolver parameters.

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
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSource` with specifications taken
            from the ModeSolver instance and the method inputs.
        """
        return self.reduced_mode_sim.sim_with_source(
            sim=self.simulation, source_time=source_time, direction=direction, mode_index=mode_index
        )

    @require_fdtd_simulation
    def sim_with_monitor(
        self,
        freqs: List[float] = None,
        name: str = None,
    ) -> Simulation:
        """Creates :class:`.Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a mode monitor added corresponding to
        the ModeSolver parameters.

        Parameters
        ----------
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
        return self.reduced_mode_sim.sim_with_monitor(sim=self.simulation, freqs=freqs, name=name)

    def sim_with_mode_solver_monitor(
        self,
        name: str,
    ) -> MODE_SIMULATION_TYPE:
        """Creates :class:`AbstractYeeGridSimulation` from a :class:`ModeSolver`. Creates a
        copy of the ModeSolver's original simulation with a mode solver monitor
        added corresponding to the ModeSolver parameters.

        Parameters
        ----------
        name : str
            Name of the monitor.

        Returns
        -------
        :class:`.AbstractYeeGridSimulation`
            Copy of the simulation with a :class:`.ModeSolverMonitor` with specifications taken
            from the ModeSolver instance and ``name``.
        """
        return self.reduced_mode_sim.sim_with_mode_solver_monitor(sim=self.simulation, name=name)

    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlaid.

        Parameters
        ----------
        field_name : str
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        # TODO: check consistency with ModeSimulation

        sim_data = self.sim_data
        return sim_data.plot_field(
            field_monitor_name=MODE_MONITOR_NAME,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )

    def plot(
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
        # TODO: check consistency with ModeSimulation
        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims()

        return self.simulation.plot(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            hlim=h_lim,
            vlim=v_lim,
            source_alpha=0,
            monitor_alpha=0,
            lumped_element_alpha=0,
            ax=ax,
            **patch_kwargs,
        )

    def plot_eps(
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
        # TODO: check consistency with ModeSimulation

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims()

        # Plot at central mode frequency if freq is not provided.
        f = freq if freq is not None else self.freqs[len(self.freqs) // 2]

        return self.simulation.plot_eps(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            freq=f,
            alpha=alpha,
            hlim=h_lim,
            vlim=v_lim,
            source_alpha=0,
            monitor_alpha=0,
            lumped_element_alpha=0,
            ax=ax,
        )

    def plot_structures_eps(
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
        # TODO: check consistency with ModeSimulation

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims()

        # Plot at central mode frequency if freq is not provided.
        f = freq if freq is not None else self.freqs[len(self.freqs) // 2]

        return self.simulation.plot_structures_eps(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            freq=f,
            alpha=alpha,
            cbar=cbar,
            reverse=reverse,
            hlim=h_lim,
            vlim=v_lim,
            ax=ax,
        )

    def plot_grid(
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
        # TODO: check consistency with ModeSimulation

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims()

        return self.simulation.plot_grid(
            x=a_center[0], y=a_center[1], z=a_center[2], hlim=h_lim, vlim=v_lim, ax=ax, **kwargs
        )

    def plot_pml(
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
        # TODO: check consistency with ModeSimulation

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, t_axes = self._center_and_lims()

        # Plot the mode plane is ax=None.
        if not ax:
            ax = self.simulation.plot(
                x=a_center[0],
                y=a_center[1],
                z=a_center[2],
                hlim=h_lim,
                vlim=v_lim,
                source_alpha=0,
                monitor_alpha=0,
                ax=ax,
            )

        # Mode plane grid.
        plane_grid = self.grid_snapped.centers.to_list
        coord_0 = plane_grid[t_axes[0]][1:-1]
        coord_1 = plane_grid[t_axes[1]][1:-1]

        # Number of PML layers in ModeSpec.
        num_pml_0 = self.mode_spec.num_pml[0]
        num_pml_1 = self.mode_spec.num_pml[1]

        # Calculate PML thickness.
        pml_thick_0_plus = 0
        pml_thick_0_minus = 0
        if num_pml_0 > 0:
            pml_thick_0_plus = coord_0[-1] - coord_0[-num_pml_0 - 1]
            pml_thick_0_minus = coord_0[num_pml_0] - coord_0[0]
            if self.solver_symmetry[0] != 0:
                pml_thick_0_minus = pml_thick_0_plus

        pml_thick_1_plus = 0
        pml_thick_1_minus = 0
        if num_pml_1 > 0:
            pml_thick_1_plus = coord_1[-1] - coord_1[-num_pml_1 - 1]
            pml_thick_1_minus = coord_1[num_pml_1] - coord_1[0]
            if self.solver_symmetry[1] != 0:
                pml_thick_1_minus = pml_thick_1_plus

        # Mode Plane width and height
        mp_w = h_lim[1] - h_lim[0]
        mp_h = v_lim[1] - v_lim[0]

        # Plot the absorbing layers.
        if num_pml_0 > 0 or num_pml_1 > 0:
            pml_rect = []
            if pml_thick_0_minus > 0:
                pml_rect.append(Rectangle((h_lim[0], v_lim[0]), pml_thick_0_minus, mp_h))
            if pml_thick_0_plus > 0:
                pml_rect.append(
                    Rectangle((h_lim[1] - pml_thick_0_plus, v_lim[0]), pml_thick_0_plus, mp_h)
                )
            if pml_thick_1_minus > 0:
                pml_rect.append(Rectangle((h_lim[0], v_lim[0]), mp_w, pml_thick_1_minus))
            if pml_thick_1_plus > 0:
                pml_rect.append(
                    Rectangle((h_lim[0], v_lim[1] - pml_thick_1_plus), mp_w, pml_thick_1_plus)
                )

            pc = PatchCollection(
                pml_rect,
                alpha=plot_params_pml.alpha,
                facecolor=plot_params_pml.facecolor,
                edgecolor=plot_params_pml.edgecolor,
                hatch=plot_params_pml.hatch,
                zorder=plot_params_pml.zorder,
            )
            ax.add_collection(pc)

        return ax

    def _center_and_lims(self) -> Tuple[List, List, List, List]:
        """Get the mode plane center and limits."""
        return self.reduced_mode_sim._center_and_lims()

    def _validate_modes_size(self):
        """Make sure that the total size of the modes fields is not too large."""
        return self.reduced_mode_sim._validate_modes_size()

    def validate_pre_upload(self, source_required: bool = True):
        self.reduced_mode_sim.validate_pre_upload(source_required=source_required)

    @property
    def reduced_simulation_copy(self):
        """Strip objects not used by the mode solver from simulation object.
        This might significantly reduce upload time in the presence of custom mediums.
        """

        if isinstance(self.simulation, EMESimulation):
            raise ValidationError(
                "The method 'reduced_simulation_copy' cannot be "
                "applied to EME simulations. If this is an issue, consider "
                "using 'reduced_mode_sim' to obtain a reduced version of the "
                "corresponding 'ModeSimulation'."
            )
            # return self.to_fdtd_mode_solver().reduced_simulation_copy

        new_sim = self.simulation._reduced_simulation_copy(plane=self.plane)
        return self.updated_copy(simulation=new_sim)

    def to_fdtd_mode_solver(self) -> ModeSolver:
        """Construct a new :class:`.ModeSolver` by converting ``simulation``
        from a :class:`.EMESimulation` or :class:`.ModeSimulation` to an FDTD :class:`.Simulation`.
        Only used as a workaround until these are natively supported in the
        :class:`.ModeSolver` webapi."""
        if not isinstance(self.simulation, (EMESimulation, ModeSimulation)):
            raise ValidationError(
                "The method 'to_fdtd_mode_solver' is only needed "
                "when the 'simulation' is an 'EMESimulation'."
            )
        fdtd_sim = self.simulation._to_fdtd_sim()
        return self.updated_copy(simulation=fdtd_sim)

    @classmethod
    def from_mode_sim(cls, sim: ModeSimulation) -> ModeSolver:
        """Construct a :class:`.ModeSolver` from a :class:`.ModeSimulation."""
        return cls(
            simulation=sim,
            plane=sim.geometry,
            mode_spec=sim.mode_spec,
            freqs=sim.freqs,
            direction=sim.direction,
            colocate=sim.colocate,
        )

    def to_mode_sim(self, reduced: bool = False, **kwargs) -> ModeSimulation:
        """Construct a :class`.ModeSimulation` from the :class:`.ModeSolver`.

        Parameters
        ----------
        reduced: bool
            Whether to strip objects not used by the mode solver from the simulation object.
            This might significantly reduce upload time in the presence of custom mediums.
        **kwargs
            Other arguments passed to new mode simulation instance.

        Returns
        -------
        :class:`.ModeSimulation`
            Mode simulation based off of the mode solver.
        """
        return ModeSimulation.from_simulation(
            simulation=self.simulation,
            plane=self.plane,
            mode_spec=self.mode_spec,
            freqs=self.freqs,
            direction=self.direction,
            colocate=self.colocate,
            reduced=reduced,
            **kwargs,
        )

    @cached_property
    def reduced_mode_sim(self) -> ModeSimulation:
        """Construct a :class`.ModeSimulation` from the :class:`.ModeSolver`.
        The mode simulation is reduced to contain only objects used by the mode solver.
        """
        return self.to_mode_sim(reduced=True)

    @cached_property
    def mode_sim_solver(self) -> ModeSimulationSolver:
        """:class:`.ModeSimulationSolver` used for mode solving."""
        return ModeSimulationSolver(simulation=self.reduced_mode_sim)
