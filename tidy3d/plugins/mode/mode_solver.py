"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from typing import List, Tuple, Dict
import logging

import numpy as np
import pydantic

from ...components.base import Tidy3dBaseModel, cached_property
from ...components import Box
from ...components import Simulation
from ...components import ModeSpec
from ...components.monitor import ModeSolverMonitor, ModeMonitor
from ...components.source import ModeSource, SourceTime
from ...components.types import Direction, ArrayLike, FreqArray, Ax, Literal, Axis
from ...components.data.data_array import ModeIndexDataArray, ScalarModeFieldDataArray
from ...components.data.sim_data import SimulationData
from ...components.data.monitor_data import ModeSolverData
from ...components.boundary import Symmetry
from ...log import ValidationError
from .solver import compute_modes

FIELD = Tuple[ArrayLike[complex, 3], ArrayLike[complex, 3], ArrayLike[complex, 3]]
MODE_MONITOR_NAME = "mode"

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
    def _plane_sym(self) -> Box:
        """Potentially smaller plane if symmetries present in the simulation."""
        return self.simulation.min_sym_box(self.plane)

    @cached_property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        mode_symmetry = list(self.simulation.symmetry)
        for dim in range(3):
            if self.simulation.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_sym = self.plane.pop_axis(mode_symmetry, axis=self.normal_axis)
        return solver_sym

    def solve(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        # note: calling sim_data like this expands the symmetries under the hood
        # ModeSolver.data will instead return the data without symmetry expansion
        return self.sim_data[MODE_MONITOR_NAME]

    @cached_property
    def data(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """

        plane_grid_sym = self.simulation.discretize(self._plane_sym)
        _, _solver_coords = self.plane.pop_axis(
            plane_grid_sym.boundaries.to_list, axis=self.normal_axis
        )

        # Compute and store the modes at all frequencies
        n_complex, fields = self._solve_all_freqs(
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

            xyz_coords = plane_grid_sym[field_name].to_list
            xyz_coords[self.normal_axis] = [self.plane.center[self.normal_axis]]
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

        # make mode solver data
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        mode_solver_data = ModeSolverData(monitor=mode_solver_monitor, **data_dict)
        self._field_decay_warning(mode_solver_data)
        return mode_solver_data

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
        return SimulationData(
            simulation=new_simulation, monitor_data={MODE_MONITOR_NAME: monitor_data}
        )

    def _get_epsilon(self, plane: Box, freq: float) -> ArrayLike[complex, 4]:
        """Compute the diagonal components of the epsilon tensor in the plane."""

        eps_xx = self.simulation.epsilon(plane, "Ex", freq)
        eps_yy = self.simulation.epsilon(plane, "Ey", freq)
        eps_zz = self.simulation.epsilon(plane, "Ez", freq)

        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def _solver_eps(self, freq: float) -> ArrayLike[complex, 4]:
        """Get the diagonal permittivity in the shape needed to be supplied to the sovler, with the
        normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        eps_xx, eps_yy, eps_zz = self._get_epsilon(self._plane_sym, freq)

        # get rid of normal axis
        eps_xx = np.squeeze(eps_xx, axis=self.normal_axis)
        eps_yy = np.squeeze(eps_yy, axis=self.normal_axis)
        eps_zz = np.squeeze(eps_zz, axis=self.normal_axis)

        # swap axes to plane coordinates (normal_axis goes to z)
        eps_sim_ax = (eps_xx, eps_yy, eps_zz)
        eps_zz, (eps_xx, eps_yy) = self.plane.pop_axis(eps_sim_ax, axis=self.normal_axis)

        # construct eps to feed to mode solver
        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayLike[float, 1], ArrayLike[float, 1]],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayLike[complex, 4]]]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        for freq in self.freqs:
            n_freq, fields_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)

        return n_complex, fields

    def _solve_single_freq(  # pylint:disable=too-many-locals
        self,
        freq: float,
        coords: Tuple[ArrayLike[float, 1], ArrayLike[float, 1]],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayLike[complex, 4]]]:
        """Call the mode solver at a single frequency.
        The fields are rotated from propagation coordinates back to global coordinates.
        """

        solver_fields, n_complex = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
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

        return n_complex, fields

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.normal_axis)
        return np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=self.normal_axis), axis=0)

    def _process_fields(
        self, mode_fields: ArrayLike[complex, 4], mode_index: pydantic.NonNegativeInt
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
                    logging.warning(
                        f"Mode field at frequency index {freq_index}, mode index {mode_index} does "
                        "not decay at the plane boundaries."
                    )

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`.ModeSolver` instance plus additional
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
        """Creates :class:`ModeMonitor` from a :class:`.ModeSolver` instance plus additional
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
        """Creates :class:`ModeSolverMonitor` from a :class:`.ModeSolver` instance.

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
            name=name,
        )

    def plot_field(  # pylint:disable=too-many-arguments
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
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
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'int'`` to plot intensity.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
            If ``field_name == 'int'``, this has no effect.
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
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )
