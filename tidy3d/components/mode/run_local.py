"""Local run functionality for ModeSimulation, without subpixel averaging."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ...constants import C_0
from ...exceptions import ValidationError
from ...log import log
from ..base import Tidy3dBaseModel
from ..data.data_array import FreqModeDataArray, ModeIndexDataArray, ScalarModeFieldDataArray
from ..data.monitor_data import ModeSolverData
from ..geometry.base import Box
from ..mode_spec import ModeSpec
from ..simulation import AbstractYeeGridSimulation
from ..types import (
    ArrayComplex3D,
    ArrayComplex4D,
    ArrayFloat1D,
    Direction,
    EpsSpecType,
    Symmetry,
)
from .data.sim_data import ModeSimulationData
from .simulation import MODE_MONITOR_NAME, ModeSimulation

# Warning for field intensity at edges over total field intensity larger than this value
FIELD_DECAY_CUTOFF = 1e-2

FIELD = Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]

# Importing the local solver may not work if e.g. scipy is not installed
IMPORT_ERROR_MSG = """Could not import local solver, 'ModeSolver' objects can still be constructed
but will have to be run through the server.
"""
try:
    from .solver import compute_modes

    LOCAL_SOLVER_IMPORTED = True
except ImportError:
    log.warning(IMPORT_ERROR_MSG)
    LOCAL_SOLVER_IMPORTED = False


class ModeSimulationSolver(Tidy3dBaseModel):
    """Local mode solver for :class:`.ModeSimulation`.
    Does not use subpixel averaging."""

    simulation: ModeSimulation = pd.Field(
        ..., title="Simulation", description=":class:`.ModeSimulation` to solve."
    )

    @property
    def data_raw(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """

        if self.simulation.mode_spec.group_index_step > 0:
            return self._get_data_with_group_index()

        # Compute data on the Yee grid
        mode_solver_data = self._data_on_yee_grid()

        # Colocate to grid boundaries if requested
        if self.simulation.colocate:
            mode_solver_data = self._colocate_data(mode_solver_data=mode_solver_data)

        # normalize modes
        self._normalize_modes(mode_solver_data=mode_solver_data)

        # filter polarization if requested
        if self.simulation.mode_spec.filter_pol is not None:
            self._filter_polarization(mode_solver_data=mode_solver_data)

        # sort modes if requested
        if self.simulation.mode_spec.track_freq and len(self.simulation.freqs) > 1:
            mode_solver_data = mode_solver_data.overlap_sort(self.simulation.mode_spec.track_freq)

        self._field_decay_warning(mode_solver_data.symmetry_expanded)

        return mode_solver_data

    def _data_on_yee_grid(self) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        _, _solver_coords = self.simulation.plane.pop_axis(
            self.simulation._solver_grid.boundaries.to_list, axis=self.simulation.normal_axis
        )

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = self._solve_all_freqs(
            coords=_solver_coords, symmetry=self.simulation.solver_symmetry
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(self.simulation.freqs),
                mode_index=np.arange(self.simulation.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct the field data on Yee grid
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            xyz_coords = self.simulation.grid_snapped[field_name].to_list
            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(self.simulation.freqs),
                    mode_index=np.arange(self.simulation.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = self._grid_correction(
            simulation=self.simulation,
            plane=self.simulation.plane,
            mode_spec=self.simulation.mode_spec,
            n_complex=index_data,
            direction=self.simulation.direction,
        )

        # make mode solver data on the Yee grid
        mode_solver_monitor = self.simulation.to_mode_solver_monitor(
            name=MODE_MONITOR_NAME, colocate=False
        )
        grid_expanded = self.simulation.discretize_monitor(mode_solver_monitor)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )

        return mode_solver_data

    def _data_on_yee_grid_relative(self, basis: ModeSolverData) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        if basis.monitor.colocate:
            raise ValidationError("Relative mode solver 'basis' must have 'colocate=False'.")
        _, _solver_coords = self.simulation.plane.pop_axis(
            self.simulation._solver_grid.boundaries.to_list, axis=self.simulation.normal_axis
        )

        basis_fields = []
        for freq_ind in range(len(basis.n_complex.f)):
            basis_fields_freq = {}
            for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                basis_fields_freq[field_name] = (
                    basis.field_components[field_name].isel(f=freq_ind).to_numpy()
                )
            basis_fields.append(basis_fields_freq)

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = self._solve_all_freqs_relative(
            coords=_solver_coords,
            symmetry=self.simulation.solver_symmetry,
            basis_fields=basis_fields,
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(self.simulation.freqs),
                mode_index=np.arange(self.simulation.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct the field data on Yee grid
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            xyz_coords = self.simulation.grid_snapped[field_name].to_list
            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(self.simulation.freqs),
                    mode_index=np.arange(self.simulation.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = self._grid_correction(
            simulation=self.simulation,
            plane=self.simulation.plane,
            mode_spec=self.simulation.mode_spec,
            n_complex=index_data,
            direction=self.simulation.direction,
        )

        # make mode solver data on the Yee grid
        mode_solver_monitor = self.simulation.to_mode_solver_monitor(
            name=MODE_MONITOR_NAME, colocate=False
        )
        grid_expanded = self.simulation.discretize_monitor(mode_solver_monitor)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )

        return mode_solver_data

    def _colocate_data(self, mode_solver_data: ModeSolverData) -> ModeSolverData:
        """Colocate data to Yee grid boundaries."""

        # Get colocation coordinates in the solver plane
        _, plane_dims = self.simulation.plane.pop_axis("xyz", self.simulation.normal_axis)
        colocate_coords = {}
        for dim, sym in zip(plane_dims, self.simulation.solver_symmetry):
            coords = self.simulation.grid_snapped.boundaries.to_dict[dim]
            if len(coords) > 2:
                if sym == 0:
                    colocate_coords[dim] = coords[1:-1]
                else:
                    colocate_coords[dim] = coords[:-1]

        # Colocate input data to new coordinates
        data_dict_colocated = {}
        for key, field in mode_solver_data.symmetry_expanded.field_components.items():
            data_dict_colocated[key] = field.interp(**colocate_coords).astype(field.dtype)

        # Update data
        mode_solver_monitor = self.simulation.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        grid_expanded = self.simulation.discretize_monitor(mode_solver_monitor)
        data_dict_colocated.update({"monitor": mode_solver_monitor, "grid_expanded": grid_expanded})
        mode_solver_data = mode_solver_data._updated(update=data_dict_colocated)

        return mode_solver_data

    def _normalize_modes(self, mode_solver_data: ModeSolverData):
        """Normalize modes. Note: this modifies ``mode_solver_data`` in-place."""
        scaling = np.sqrt(np.abs(mode_solver_data.flux))
        for field in mode_solver_data.field_components.values():
            field /= scaling

    def _filter_polarization(self, mode_solver_data: ModeSolverData):
        """Filter polarization. Note: this modifies ``mode_solver_data`` in-place."""
        pol_frac = mode_solver_data.pol_fraction
        for ifreq in range(len(self.simulation.freqs)):
            te_frac = pol_frac.te.isel(f=ifreq)
            if self.simulation.mode_spec.filter_pol == "te":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac >= 0.5)[0],
                        np.where(te_frac < 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            elif self.simulation.mode_spec.filter_pol == "tm":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac <= 0.5)[0],
                        np.where(te_frac > 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            for data in list(mode_solver_data.field_components.values()) + [
                mode_solver_data.n_complex,
                mode_solver_data.grid_primal_correction,
                mode_solver_data.grid_dual_correction,
            ]:
                data.values[..., ifreq, :] = data.values[..., ifreq, sort_inds]

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq in self.simulation.freqs:
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)
        return n_complex, fields, eps_spec

    def _solve_all_freqs_relative(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
        basis_fields: List[Dict[str, ArrayComplex4D]],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq, basis_fields_freq in zip(self.simulation.freqs, basis_fields):
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq_relative(
                freq=freq, coords=coords, symmetry=symmetry, basis_fields=basis_fields_freq
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)

        return n_complex, fields, eps_spec

    def _postprocess_solver_fields(self, solver_fields):
        """Postprocess `solver_fields` from `compute_modes` to proper coordinate"""
        fields = {key: [] for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        for mode_index in range(self.simulation.mode_spec.num_modes):
            # Get E and H fields at the current mode_index
            ((Ex, Ey, Ez), (Hx, Hy, Hz)) = self._process_fields(solver_fields, mode_index)

            # Note: back in original coordinates
            fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
            for field_name, field in fields_mode.items():
                fields[field_name].append(field)

        for field_name, field in fields.items():
            fields[field_name] = np.stack(field, axis=-1)
        return fields

    def _solve_single_freq(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.

        The fields are rotated from propagation coordinates back to global coordinates.
        """

        if not LOCAL_SOLVER_IMPORTED:
            raise ImportError(IMPORT_ERROR_MSG)

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self.simulation._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.simulation.mode_spec,
            symmetry=symmetry,
            direction=self.simulation.direction,
        )

        fields = self._postprocess_solver_fields(solver_fields)
        return n_complex, fields, eps_spec

    def _rotate_field_coords_inverse(self, field: FIELD) -> FIELD:
        """Move the propagation axis to the z axis in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=1 + self.simulation.normal_axis, destination=3)
        f_n, f_ts = self.simulation.plane.pop_axis(
            (f_x, f_y, f_z), axis=self.simulation.normal_axis
        )
        return np.stack(self.simulation.plane.unpop_axis(f_n, f_ts, axis=2), axis=0)

    def _postprocess_solver_fields_inverse(self, fields):
        """Convert ``fields`` to ``solver_fields``. Doesn't change gauge."""
        E = [fields[key] for key in ("Ex", "Ey", "Ez")]
        H = [fields[key] for key in ("Hx", "Hy", "Hz")]

        (Ex, Ey, Ez) = self._rotate_field_coords_inverse(E)
        (Hx, Hy, Hz) = self._rotate_field_coords_inverse(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.simulation.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        solver_fields = np.stack((Ex, Ey, Ez, Hx, Hy, Hz), axis=0)
        return solver_fields

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

        if not LOCAL_SOLVER_IMPORTED:
            raise ImportError(IMPORT_ERROR_MSG)

        solver_basis_fields = self._postprocess_solver_fields_inverse(basis_fields)

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self.simulation._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.simulation.mode_spec,
            symmetry=symmetry,
            direction=self.simulation.direction,
            solver_basis_fields=solver_basis_fields,
        )

        fields = self._postprocess_solver_fields(solver_fields)
        return n_complex, fields, eps_spec

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.simulation.normal_axis)
        return np.stack(
            self.simulation.plane.unpop_axis(f_z, (f_x, f_y), axis=self.simulation.normal_axis),
            axis=0,
        )

    def _process_fields(
        self, mode_fields: ArrayComplex4D, mode_index: pd.NonNegativeInt
    ) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes and set gauge."""

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
        if self.simulation.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def _field_decay_warning(self, field_data: ModeSolverData):
        """Warn if any of the modes do not decay at the edges."""

        _, plane_dims = self.simulation.plane.pop_axis(
            ["x", "y", "z"], axis=self.simulation.normal_axis
        )
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
        simulation: AbstractYeeGridSimulation,
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

    def _get_data_with_group_index(self) -> ModeSolverData:
        """:class:`.ModeSolverData` with fields, effective and group indices on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective and group indices, and mode
            fields.
        """

        # create a copy with the required frequencies for numerical differentiation
        mode_spec = self.simulation.mode_spec.copy(update={"group_index_step": False})
        mode_sim = self.simulation.copy(
            update={"freqs": self.simulation._freqs_for_group_index(), "mode_spec": mode_spec}
        )
        mode_sim_solver = self.updated_copy(simulation=mode_sim)
        return mode_sim_solver.data_raw._group_index_post_process(
            self.simulation.mode_spec.group_index_step
        )

    @property
    def reduced_simulation_copy(self) -> ModeSimulationSolver:
        """Reduced simulation copy, used for more efficient mode solving."""
        return self.updated_copy(simulation=self.simulation.reduced_simulation_copy)

    def solve(self) -> ModeSimulationData:
        """Run the :class:`ModeSimulation` locally. Local runs may be less accurate than remote
        runs, which can use subpixel averaging.

        Parameters
        ----------
        sim: ModeSimulation
            :class:`.ModeSimulation` object to run..
        Returns
        -------
        ModeSimulationData
            :class:`.ModeSimulationData` obtained from the local run of the :class:`.ModeSimulation`..
        """
        log.warning(
            "Use the remote mode solver with subpixel averaging for better accuracy through "
            "'tidy3d.plugins.mode.web.run(...)'.",
            log_once=True,
        )

        data_raw = self.reduced_simulation_copy.data_raw
        return ModeSimulationData(simulation=self.simulation, data_raw=data_raw, remote=False)


def run_local(sim: ModeSimulation) -> ModeSimulationData:
    """Run the :class:`ModeSimulation` locally. Local runs may be less accurate than remote
    runs, which can use subpixel averaging.

    Parameters
    ----------
    sim: ModeSimulation
        :class:`.ModeSimulation` object to run..
    Returns
    -------
    ModeSimulationData
        :class:`.ModeSimulationData` obtained from the local run of the :class:`.ModeSimulation`..
    """

    return ModeSimulationSolver(simulation=sim).solve()
