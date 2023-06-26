"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
import pydantic
import xarray as xr

from ...log import log
from ...components.base import Tidy3dBaseModel, cached_property
from ...components.geometry import Box
from ...components.simulation import Simulation
from ...components.grid.grid import Grid
from ...components.mode import ModeSpec
from ...components.monitor import ModeSolverMonitor, ModeMonitor
from ...components.source import ModeSource, SourceTime
from ...components.types import Direction, FreqArray, Ax, Literal, Axis, Symmetry, PlotScale
from ...components.types import ArrayComplex3D, ArrayComplex4D, ArrayFloat1D, EpsSpecType
from ...components.data.data_array import ModeIndexDataArray, ScalarModeFieldDataArray
from ...components.data.data_array import FreqModeDataArray
from ...components.data.sim_data import SimulationData
from ...components.data.monitor_data import ModeSolverData
from ...exceptions import ValidationError
from ...constants import C_0
from .solver import compute_modes


FIELD = Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
MODE_MONITOR_NAME = "<<<MODE_SOLVER_MONITOR>>>"

# Warning for field intensity at edges over total field intensity larger than this value
FIELD_DECAY_CUTOFF = 1e-2


class ModeSolver(Tidy3dBaseModel):
    """Interface for solving electromagnetic eigenmodes in a 2D plane with translational
    invariance in the third dimension.
    """

    simulation: Simulation = pydantic.Field(
        ..., title="Simulation", description="Simulation defining all structures and mediums."
    )

    plane: Box = pydantic.Field(
        ..., title="Plane", description="Cross-sectional plane in which the mode will be computed."
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

    @pydantic.validator("plane", always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    @pydantic.validator("freqs", always=True)
    def freqs_not_empty(cls, val):
        """Raise validation error if ``freqs`` is an empty Tuple."""
        if len(val) == 0:
            raise ValidationError("ModeSolver 'freqs' must be a non-empty tuple.")
        return val

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.plane.size.index(0.0)

    @cached_property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        mode_symmetry = list(self.simulation.symmetry)
        for dim in range(3):
            if self.simulation.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_sym = self.plane.pop_axis(mode_symmetry, axis=self.normal_axis)
        return solver_sym

    @cached_property
    def _solver_grid(self) -> Grid:
        """Grid for the mode solver, including extension in the normal direction, which is needed
        to get epsilon from the simulation. The mode fields coordinate along the normal direction
        will be reset to the exact plane position after the solve."""
        plane_sym = self.simulation.min_sym_box(self.plane)
        boundaries = self.simulation.discretize(plane_sym, extend=True).boundaries.to_list
        # Do not extend if simulation has a single pixel along a dimension
        for dim, num_cells in enumerate(self.simulation.grid.num_cells):
            if num_cells <= 1:
                boundaries[dim] = self.simulation.grid.boundaries.to_list[dim]
        # Remove extension on the min side if symmetry present
        bounds_norm, bounds_plane = plane_sym.pop_axis(boundaries, self.normal_axis)
        bounds_plane = list(bounds_plane)
        for dim, sym in enumerate(self.solver_symmetry):
            if sym != 0:
                bounds_plane[dim] = bounds_plane[dim][1:]
        boundaries = plane_sym.unpop_axis(bounds_norm, bounds_plane, axis=self.normal_axis)
        return Grid(boundaries=dict(zip("xyz", boundaries)))

    def solve(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        log.warning(
            "Use the remote mode solver with subpixel averaging for better accuracy through "
            "'tidy3d.plugins.mode.web.run(...)'.",
            log_once=True,
        )
        return self.data

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

    def _get_data_with_group_index(self) -> ModeSolverData:
        """:class:`.ModeSolverData` with fields, effective and group indices on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective and group indices, and mode
            fields.
        """

        # create a copy with the required frequencies for numerical differentiation
        mode_spec = self.mode_spec.copy(update={"group_index_step": False})
        mode_solver = self.copy(
            update={"freqs": self._freqs_for_group_index(), "mode_spec": mode_spec}
        )

        # pylint:disable=protected-access
        return mode_solver.data_raw._group_index_post_process(self.mode_spec.group_index_step)

    # pylint:disable=too-many-locals
    @cached_property
    def data_raw(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """

        if self.mode_spec.group_index_step > 0:
            return self._get_data_with_group_index()

        _, _solver_coords = self.plane.pop_axis(
            self._solver_grid.boundaries.to_list, axis=self.normal_axis
        )

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = self._solve_all_freqs(
            coords=_solver_coords, symmetry=self.solver_symmetry
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(self.freqs),
                mode_index=np.arange(self.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct and add all the data for the fields
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):

            xyz_coords = self._solver_grid[field_name].to_list
            # Snap to plane center along normal direction
            xyz_coords[self.normal_axis] = [self.plane.center[self.normal_axis]]
            # Snap to simulation center if simulation is 0D along a tangential dimension
            _, plane_axes = self.plane.pop_axis([0, 1, 2], axis=self.normal_axis)
            for plane_axis in plane_axes:
                if len(xyz_coords[plane_axis]) == 1:
                    xyz_coords[plane_axis] = [self.simulation.center[plane_axis]]

            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(self.freqs),
                    mode_index=np.arange(self.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = self._grid_correction(
            simulation=self.simulation,
            plane=self.plane,
            mode_spec=self.mode_spec,
            n_complex=index_data,
            direction=self.direction,
        )

        # make mode solver data
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=self.simulation.discretize(self.plane, extend=True),
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )
        self._field_decay_warning(mode_solver_data.symmetry_expanded_copy)

        # normalize modes
        scaling = np.sqrt(np.abs(mode_solver_data.flux))
        mode_solver_data = mode_solver_data.copy(
            update={
                key: field / scaling for key, field in mode_solver_data.field_components.items()
            }
        )

        # sort modes if requested
        if self.mode_spec.track_freq and len(self.freqs) > 1:
            mode_solver_data = mode_solver_data.overlap_sort(self.mode_spec.track_freq)

        return mode_solver_data

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

    @cached_property
    def sim_data(self) -> SimulationData:
        """:class:`.SimulationData` object containing the :class:`.ModeSolverData` for this object.

        Returns
        -------
        SimulationData
            :class:`.SimulationData` object containing the effective index and mode fields.
        """
        monitor_data = self.data
        new_monitors = list(self.simulation.monitors) + [monitor_data.monitor]
        new_simulation = self.simulation.copy(update=dict(monitors=new_monitors))
        return SimulationData(simulation=new_simulation, data=(monitor_data,))

    def _get_epsilon(self, freq: float) -> ArrayComplex4D:
        """Compute the epsilon tensor in the plane. Order of components is xx, xy, xz, yx, etc."""
        eps_keys = ["Ex", "Exy", "Exz", "Eyx", "Ey", "Eyz", "Ezx", "Ezy", "Ez"]
        eps_tensor = [
            self.simulation.epsilon_on_grid(self._solver_grid, key, freq) for key in eps_keys
        ]
        return np.stack(eps_tensor, axis=0)

    def _solver_eps(self, freq: float) -> ArrayComplex4D:
        """Diagonal permittivity in the shape needed by solver, with normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        eps_tensor = self._get_epsilon(freq)

        # get rid of normal axis
        eps_tensor = np.take(eps_tensor, indices=[0], axis=1 + self.normal_axis)
        eps_tensor = np.squeeze(eps_tensor, axis=1 + self.normal_axis)

        # convert to into 3-by-3 representation for easier axis swap
        flat_shape = np.shape(eps_tensor)  # 9 components flat
        tensor_shape = [3, 3] + list(flat_shape[1:])  # 3-by-3 matrix
        eps_tensor = eps_tensor.reshape(tensor_shape)

        # swap axes to plane coordinates (normal_axis goes to z)
        if self.normal_axis == 0:
            # swap x and y
            eps_tensor[[0, 1], :, ...] = eps_tensor[[1, 0], :, ...]
            eps_tensor[:, [0, 1], ...] = eps_tensor[:, [1, 0], ...]
        if self.normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            eps_tensor[[1, 2], :, ...] = eps_tensor[[2, 1], :, ...]
            eps_tensor[:, [1, 2], ...] = eps_tensor[:, [2, 1], ...]

        # back to "flat" representation
        eps_tensor = eps_tensor.reshape(flat_shape)

        # construct eps to feed to mode solver
        return eps_tensor

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq in self.freqs:
            # pylint: disable=unbalanced-tuple-unpacking
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)

        return n_complex, fields, eps_spec

    # pylint:disable=too-many-locals
    def _solve_single_freq(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.

        The fields are rotated from propagation coordinates back to global coordinates.
        """

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
            direction=self.direction,
        )

        fields = {key: [] for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        for mode_index in range(self.mode_spec.num_modes):
            # Get E and H fields at the current mode_index
            ((Ex, Ey, Ez), (Hx, Hy, Hz)) = self._process_fields(solver_fields, mode_index)

            # Note: back in original coordinates
            fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
            for field_name, field in fields_mode.items():
                fields[field_name].append(field)

        for field_name, field in fields.items():
            fields[field_name] = np.stack(field, axis=-1)

        return n_complex, fields, eps_spec

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.normal_axis)
        return np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=self.normal_axis), axis=0)

    def _process_fields(
        self, mode_fields: ArrayComplex4D, mode_index: pydantic.NonNegativeInt
    ) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes, set gauge, and check decay at boundaries."""

        # Separate E and H fields (in solver coordinates)
        E, H = mode_fields[..., mode_index]

        # Set gauge to highest-amplitude in-plane E being real and positive
        ind_max = np.argmax(np.abs(E[:2]))
        phi = np.angle(E[:2].ravel()[ind_max])
        E *= np.exp(-1j * phi)
        H *= np.exp(-1j * phi)

        # Rotate back to original coordinates
        (Ex, Ey, Ez) = self._rotate_field_coords(E)
        (Hx, Hy, Hz) = self._rotate_field_coords(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def _field_decay_warning(self, field_data: ModeSolverData):
        """Warn if any of the modes do not decay at the edges."""
        _, plane_dims = self.plane.pop_axis(["x", "y", "z"], axis=self.normal_axis)
        field_sizes = field_data.Ex.sizes
        for freq_index in range(field_sizes["f"]):
            for mode_index in range(field_sizes["mode_index"]):
                e_edge, e_norm = 0, 0
                # Sum up the total field intensity
                for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                    e_norm += np.sum(np.abs(E[{"f": freq_index, "mode_index": mode_index}]) ** 2)
                # Sum up the field intensity at the edges
                if field_sizes[plane_dims[0]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[0]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                if field_sizes[plane_dims[1]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[1]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                # Warn if needed
                if e_edge / e_norm > FIELD_DECAY_CUTOFF:
                    log.warning(
                        f"Mode field at frequency index {freq_index}, mode index {mode_index} does "
                        "not decay at the plane boundaries."
                    )

    @staticmethod
    def _grid_correction(
        simulation: Simulation,
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
        normal_axis = plane.size.index(0.0)
        normal_pos = plane.center[normal_axis]
        normal_dim = "xyz"[normal_axis]

        # Primal and dual grid along the normal direction,
        # i.e. locations of the tangential E-field and H-field components, respectively
        grid = simulation.grid
        normal_primal = grid.boundaries.to_list[normal_axis]
        normal_primal = xr.DataArray(normal_primal, coords={normal_dim: normal_primal})
        normal_dual = grid.centers.to_list[normal_axis]
        normal_dual = xr.DataArray(normal_dual, coords={normal_dim: normal_dual})

        # Propagation phase at the primal and dual locations. The k-vector is along the propagation
        # direction, so angle_theta has to be taken into account. The distance along the propagation
        # direction is the distance along the normal direction over cosine(theta).
        cos_theta = np.cos(mode_spec.angle_theta)
        k_vec = 2 * np.pi * n_complex * n_complex.f / C_0 / cos_theta
        if direction == "-":
            k_vec *= -1
        phase_primal = np.exp(1j * k_vec * (normal_primal - normal_pos))
        phase_dual = np.exp(1j * k_vec * (normal_dual - normal_pos))

        # Fields are modified by a linear interpolation to the exact monitor position
        if normal_primal.size > 1:
            phase_primal = phase_primal.interp(**{normal_dim: normal_pos})
        else:
            phase_primal = phase_primal.squeeze(dim=normal_dim)
        if normal_dual.size > 1:
            phase_dual = phase_dual.interp(**{normal_dim: normal_pos})
        else:
            phase_dual = phase_dual.squeeze(dim=normal_dim)

        return FreqModeDataArray(phase_primal), FreqModeDataArray(phase_dual)

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.ModeSource`
            Mode source with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        return ModeSource(
            center=self.plane.center,
            size=self.plane.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=direction,
        )

    def to_monitor(self, freqs: List[float], name: str) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.ModeMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        return ModeMonitor(
            center=self.plane.center,
            size=self.plane.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=name,
        )

    def to_mode_solver_monitor(self, name: str) -> ModeSolverMonitor:
        """Creates :class:`ModeSolverMonitor` from a :class:`ModeSolver` instance.

        Parameters
        ----------
        name : str
            Name of the monitor.

        Returns
        -------
        :class:`.ModeSolverMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and ``name``.
        """
        return ModeSolverMonitor(
            size=self.plane.size,
            center=self.plane.center,
            mode_spec=self.mode_spec,
            freqs=self.freqs,
            direction=self.direction,
            name=name,
        )

    # pylint:disable=too-many-arguments
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
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlayed.

        Parameters
        ----------
        field_name : str
            Name of `field` component to plot (eg. `'Ex'`).
            Also accepts `'E'` and `'H'` to plot the vector magnitudes of the electric and
            magnetic fields, and `'S'` for the Poynting vector.
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

        sim_data = self.sim_data
        sim_data.plot_field(
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
