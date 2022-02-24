"""Turn Mode Specifications into Mode profiles 
"""

from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...components.base import Tidy3dBaseModel
from ...components import Box
from ...components import Simulation
from ...components import ModeSpec
from ...components import ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components.types import Direction
from ...components.data import ScalarFieldData, FieldData
from ...log import SetupError

from .solver import compute_modes


"""
Stage:                Simulation     Mode Specs    Outputs       Viz           Export
                      ----------  +  ---------- -> ----------- -> ---------- -> ----------
Method:                        __init__()          .solve()       .plot()       .to_file()

td Objects:     Simulation     Mode    ->    FieldData   ->  image     -> ModeSource
                Plane           ^                             |           ModeMonitor
                Frequency       |_____________________________|
                                      iterative design


simulation = td.Simulation(...)        # define PML, gridsize, structures
plane = td.Box(...)                    # define plane that we solve modes on
freqs = td.FreqSampler(freqs=[1, 2])   # frequencies we care about
ms = ModeSolver(simulation, plane, freqs)

mode_spec = td.Mode(num_modes=3)       # solve for a number of modes to find the one we want
modes = ms.solve(mode_spec=mode_spec)  # list of ModeInfo objects for each mode
mode_index = 1                         # initial guess for the mode index
modes[mode_index].field_data.plot()    # inspect fields, do they look ok?

mon = ms.export_monitor(mode_spec=mode_spec)   # if we're happy with results, return td.ModeMonitor
src = ms.export_src(mode_spec=mode_spec,       # or as a td.ModeSource
    mode_index=mode_index,
    src_time=...)   

src.to_file('data/my_source.json')                 # this source /monitor can be saved to file
src = ModeSource.from_file('data/my_source.json')  # and loaded in our script
"""


class ModeInfo(Tidy3dBaseModel):
    """stores information about a (solved) mode.
    Attributes
    ----------
    field_data: FieldData
        Contains information about the fields of the modal profile.
    mode_spec: ModeSpec
        Specifications of the mode.
    n_eff: float
        Real part of the effective refractive index of mode.
    k_eff: float
        Imaginary part of the effective refractive index of mode.
    """

    field_data: FieldData
    mode_spec: ModeSpec
    mode_index: int
    n_eff: float
    k_eff: float


class ModeSolver:
    """Interface for creating ``Mode`` objects."""

    def __init__(self, simulation: Simulation, plane: Box, freq: float):
        """Create a ``ModeSolver`` instance.

        Parameters
        ----------
        simulation : Simulation
            ``Simulation`` the ``Mode`` will be inserted into.
        plane : Box
            Plane where the mode will be computed in ``Simulation``.
        freq : float
            Frequency of mode (Hz).
        """

        self.simulation = simulation
        self.plane = plane
        self.freq = freq

        assert 0.0 in plane.size, "plane must have at least one axis with size=0"

    def solve(self, mode_spec: ModeSpec) -> List[ModeInfo]:
        """Solves for modal profile and effective index of ``Mode`` object.

        Parameters
        ----------
        mode_spec : ModeSpec
            ``ModeSpec`` object containing specifications of the mode solver.

        Returns
        -------
        List[ModeInfo]
            A list of ``ModeInfo`` objects for each mode.
        """

        normal_axis = self.plane.size.index(0.0)

        # get the in-plane grid coordinates on which eps and the mode fields live
        plane_grid = self.simulation.discretize(self.plane)

        # restrict to a smaller plane if symmetries present in the simulation
        plane_sym = self.simulation.min_sym_box(self.plane)
        plane_grid_sym = self.simulation.discretize(plane_sym)

        # Coords and symmetry arguments to the solver (restricted to in-plane)
        _, solver_coords = self.plane.pop_axis(plane_grid_sym.boundaries.to_list, axis=normal_axis)
        mode_symmetry = list(self.simulation.symmetry)
        for dim in range(3):
            if self.simulation.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_symmetry = self.plane.pop_axis(mode_symmetry, axis=normal_axis)

        # Get diagonal epsilon components in the plane
        (eps_xx, eps_yy, eps_zz) = self.get_epsilon(plane_sym)

        # get rid of normal axis
        eps_xx = np.squeeze(eps_xx, axis=normal_axis)
        eps_yy = np.squeeze(eps_yy, axis=normal_axis)
        eps_zz = np.squeeze(eps_zz, axis=normal_axis)

        # swap axes to waveguide coordinates (propagating in z)
        eps_wg_zz, (eps_wg_xx, eps_wg_yy) = self.plane.pop_axis(
            (eps_xx, eps_yy, eps_zz), axis=normal_axis
        )

        # construct eps_cross section to feed to mode solver
        eps_cross = np.stack((eps_wg_xx, eps_wg_yy, eps_wg_zz))

        # Compute the modes
        mode_fields, n_eff_complex = compute_modes(
            eps_cross=eps_cross,
            coords=solver_coords,
            freq=self.freq,
            mode_spec=mode_spec,
            symmetry=solver_symmetry,
        )

        def rotate_field_coords(field):
            """move the propagation axis=z to the proper order in the array"""
            f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + normal_axis)
            f_rot = np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=normal_axis), axis=0)
            return f_rot

        modes = []
        for mode_index in range(mode_spec.num_modes):

            # Get E and H fields at the current mode_index
            E, H = mode_fields[..., mode_index]

            # Set gauge to highest-amplitude in-plane E being real and positive
            ind_max = np.argmax(np.abs(E[:2]))
            phi = np.angle(E[:2].ravel()[ind_max])
            E *= np.exp(-1j * phi)
            H *= np.exp(-1j * phi)

            # Rotate back to original coordinates
            (Ex, Ey, Ez) = rotate_field_coords(E)
            (Hx, Hy, Hz) = rotate_field_coords(H)

            # apply -1 to H fields if a reflection was involved in the rotation
            if normal_axis == 1:
                Hx *= -1
                Hy *= -1
                Hz *= -1

            # note: from this point on, back in original coordinates
            fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

            # note: re-discretizing, need to make consistent.
            data_dict = {}
            for field_name, field in fields.items():
                xyz_coords = plane_grid_sym[field_name].to_list
                xyz_coords[normal_axis] = [self.plane.center[normal_axis]]
                data_dict[field_name] = ScalarFieldData(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=np.array([self.freq]),
                    values=field[..., None],
                )

            field_data = FieldData(data_dict=data_dict).apply_syms(
                plane_grid, self.simulation.center, self.simulation.symmetry
            )
            mode_info = ModeInfo(
                field_data=field_data,
                mode_spec=mode_spec,
                mode_index=mode_index,
                n_eff=n_eff_complex[mode_index].real,
                k_eff=n_eff_complex[mode_index].imag,
            )

            modes.append(mode_info)

        return modes

    def get_epsilon(self, plane):
        """Compute the diagonal components of the epsilon tensor in the plane."""

        eps_xx = self.simulation.epsilon(plane, "Ex", self.freq)
        eps_yy = self.simulation.epsilon(plane, "Ey", self.freq)
        eps_zz = self.simulation.epsilon(plane, "Ez", self.freq)

        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def to_source(
        self, mode_spec: ModeSpec, fwidth: float, direction: Direction, mode_index: int = 0
    ) -> ModeSource:
        """Creates ``ModeSource`` from a Mode + additional specifications.

        Parameters
        ----------
        mode_spec : ModeSpec
            :class:`ModeSpec` object containing specifications of mode.
        fwidth : float
            Standard deviation of ``GaussianPulse`` of source (Hz).
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
        source_time = GaussianPulse(freq0=self.freq, fwidth=fwidth)
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
