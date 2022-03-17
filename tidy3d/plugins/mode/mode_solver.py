"""Turn Mode Specifications into Mode profiles 
"""

from typing import List, Tuple
import logging

import numpy as np
import pydantic

from ...components.base import Tidy3dBaseModel
from ...components import Box
from ...components import Simulation
from ...components import ModeSpec
from ...components import ModeMonitor
from ...components.source import ModeSource, SourceTime
from ...components.types import Direction, Array
from ...components.data import ModeIndexData, ModeFieldData, ScalarModeFieldData, ModeSolverData
from ...log import ValidationError

from .solver import compute_modes

FIELD = Tuple[Array[complex], Array[complex], Array[complex]]

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

    @pydantic.validator("plane", allow_reuse=True, always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    @property
    def normal_axis(self):
        """Axis normal to the mode plane."""
        return self.plane.size.index(0.0)

    @property
    def plane_sym(self):
        """Potentially smaller plane if symmetries present in the simulation."""
        return self.simulation.min_sym_box(self.plane)

    def solve(self, mode_spec: ModeSpec, freqs: List[float]) -> ModeSolverData:
        """Solves for modal profile and effective index of ``Mode`` object.

        Parameters
        ----------
        mode_spec : ModeSpec
            ``ModeSpec`` object containing specifications of the mode solver.
        freqs : List[float]
            List of frequencies to solve at (Hz).

        Returns
        -------
        ModeSolverData
            ``ModeSolverData`` object containing the effective index and mode fields for all modes.
        """

        normal_axis = self.normal_axis

        # get the in-plane grid coordinates on which eps and the mode fields live
        plane_grid = self.simulation.discretize(self.plane)

        # restrict to a smaller plane if symmetries present in the simulation
        plane_grid_sym = self.simulation.discretize(self.plane_sym)

        # Coords and symmetry arguments to the solver (restricted to in-plane)
        _, solver_coords = self.plane.pop_axis(plane_grid_sym.boundaries.to_list, axis=normal_axis)
        mode_symmetry = list(self.simulation.symmetry)
        for dim in range(3):
            if self.simulation.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_symmetry = self.plane.pop_axis(mode_symmetry, axis=normal_axis)

        # Compute and store the modes at all frequencies
        fields = {"Ex": [], "Ey": [], "Ez": [], "Hx": [], "Hy": [], "Hz": []}
        n_complex = []
        for ifreq, freq in enumerate(freqs):
            # Compute the modes
            mode_fields, n_comp = compute_modes(
                eps_cross=self.solver_eps(freq),
                coords=solver_coords,
                freq=freq,
                mode_spec=mode_spec,
                symmetry=solver_symmetry,
            )
            n_complex.append(n_comp)

            fields_freq = {"Ex": [], "Ey": [], "Ez": [], "Hx": [], "Hy": [], "Hz": []}
            for mode_index in range(mode_spec.num_modes):
                # Get E and H fields at the current mode_index
                ((Ex, Ey, Ez), (Hx, Hy, Hz)) = self.process_fields(mode_fields, ifreq, mode_index)

                # note: back in original coordinates
                fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
                for field_name, field in fields_mode.items():
                    fields_freq[field_name].append(np.copy(field))

            for field_name, field in fields_freq.items():
                fields[field_name].append(np.stack(field, axis=-1))

        # Generate the dictionary of ScalarModeFieldData for every field
        data_dict = {}
        for field_name, field in fields.items():
            xyz_coords = plane_grid_sym[field_name].to_list
            xyz_coords[normal_axis] = [self.plane.center[normal_axis]]
            data_dict[field_name] = ScalarModeFieldData(
                x=xyz_coords[0],
                y=xyz_coords[1],
                z=xyz_coords[2],
                f=freqs,
                mode_index=np.arange(mode_spec.num_modes),
                values=np.stack(field, axis=-2),
            )

        field_data = ModeFieldData(data_dict=data_dict).apply_syms(
            plane_grid, self.simulation.center, self.simulation.symmetry
        )
        index_data = ModeIndexData(
            f=freqs,
            mode_index=np.arange(mode_spec.num_modes),
            values=np.stack(n_complex, axis=0),
        )
        mode_info = ModeSolverData(data_dict={"fields": field_data, "n_complex": index_data})

        return mode_info

    def get_epsilon(self, plane: Box, freq: float) -> Array[complex]:
        """Compute the diagonal components of the epsilon tensor in the plane."""

        eps_xx = self.simulation.epsilon(plane, "Ex", freq)
        eps_yy = self.simulation.epsilon(plane, "Ey", freq)
        eps_zz = self.simulation.epsilon(plane, "Ez", freq)

        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def solver_eps(self, freq: float) -> Array[complex]:
        """Get the diagonal permittivity as supplied to the sovler, with the nomral axis rotated to
        z."""

        # Get diagonal epsilon components in the plane
        (eps_xx, eps_yy, eps_zz) = self.get_epsilon(self.plane_sym, freq)

        # get rid of normal axis
        eps_xx = np.squeeze(eps_xx, axis=self.normal_axis)
        eps_yy = np.squeeze(eps_yy, axis=self.normal_axis)
        eps_zz = np.squeeze(eps_zz, axis=self.normal_axis)

        # swap axes to plane coordinates (normal_axis goes to z)
        eps_sim_ax = (eps_xx, eps_yy, eps_zz)
        eps_zz, (eps_xx, eps_yy) = self.plane.pop_axis(eps_sim_ax, axis=self.normal_axis)

        # construct eps to feed to mode solver
        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def rotate_field_coords(self, field):
        """move the propagation axis=z to the proper order in the array"""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.normal_axis)
        f_rot = np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=self.normal_axis), axis=0)
        return f_rot

    def process_fields(
        self, mode_fields: Array[complex], freq_index: int, mode_index: int
    ) -> Tuple[FIELD, FIELD]:
        """transform solver fields to simulation axes, set gauge, and check decay at boundaries."""

        # Separate E and H fields (in solver coordinates)
        E, H = mode_fields[..., mode_index]

        # Warn if not decayed at edges
        e_edge = np.sum(np.abs(E[:, 0, :]) ** 2 + np.abs(E[:, -1, :]) ** 2)
        e_edge += np.sum(np.abs(E[:, :, 0]) ** 2 + np.abs(E[:, :, -1]) ** 2)
        e_norm = np.sum(np.abs(E) ** 2)

        if e_edge / e_norm > FIELD_DECAY_CUTOFF:
            logging.warning(
                f"Mode field at frequency index {freq_index}, mode index {mode_index} does not "
                "decay at the plane boundaries."
            )

        # Set gauge to highest-amplitude in-plane E being real and positive
        ind_max = np.argmax(np.abs(E[:2]))
        phi = np.angle(E[:2].ravel()[ind_max])
        E *= np.exp(-1j * phi)
        H *= np.exp(-1j * phi)

        # Rotate back to original coordinates
        (Ex, Ey, Ez) = self.rotate_field_coords(E)
        (Hx, Hy, Hz) = self.rotate_field_coords(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def to_source(
        self,
        mode_spec: ModeSpec,
        source_time: SourceTime,
        direction: Direction,
        mode_index: int = 0,
    ) -> ModeSource:
        """Creates ``ModeSource`` from a Mode + additional specifications.

        Parameters
        ----------
        mode_spec : ModeSpec
            :class:`ModeSpec` object containing specifications of mode.
        source_time: SourceTime
            Specification of the source time-dependence.
        direction : Direction
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        ModeSource
            Modal source containing specification in ``mode``.
        """

        center = self.plane.center
        size = self.plane.size
        return ModeSource(
            center=center,
            size=size,
            source_time=source_time,
            mode_spec=mode_spec,
            mode_index=mode_index,
            direction=direction,
        )

    def to_monitor(self, mode_spec: ModeSpec, freqs: List[float], name: str) -> ModeMonitor:
        """Creates ``ModeMonitor`` from a Mode + additional specifications.

        Parameters
        ----------
        mode_spec : ModeSpec
            :class:`ModeSpec` object containing specifications of mode.
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
        name : str
            Required name of monitor.
        Returns
        -------
        ModeMonitor
            Monitor that measures modes specified by ``mode_spec`` on ``plane`` at ``freqs``.
        """
        center = self.plane.center
        size = self.plane.size
        return ModeMonitor(center=center, size=size, freqs=freqs, mode_spec=mode_spec, name=name)
