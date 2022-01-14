"""Turn Mode Specifications into Mode profiles 
"""

from typing import List
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...components import Box
from ...components import Simulation
from ...components import Mode
from ...components import ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components.types import Direction
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

mode = td.Mode(mode_index=2)           # our initial guess for the mode
field = ms.solve(mode=mode)            # td.FieldData storing E, H
field.plot()                           # inspect fields, do they look ok?

mon = ms.export_monitor(mode=mode)             # if we're happy with results, return td.ModeMonitor
src = ms.export_src(mode=mode, src_time=...)   # or as a td.ModeSource

src.to_file('data/my_source.json')             # this source /monitor can be saved to file
src = ModeSource.from_file('data/my_source.json')  # and loaded in our script
"""


@dataclass
class ModeInfo:
    """stores information about a (solved) mode.
    Attributes
    ----------
    field_data: xr.Dataset
        Contains information about the fields of the modal profile.
    mode: Mode
        Specifications of the mode.
    n_eff: float
        Real part of the effective refractive index of mode.
    k_eff: float
        Imaginary part of the effective refractive index of mode.
    """

    field_data: xr.Dataset
    mode: Mode
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

    def solve(self, mode: Mode) -> ModeInfo:
        """Solves for modal profile and effective index of ``Mode`` object.

        Parameters
        ----------
        mode : Mode
            ``Mode`` object containing specifications of mode.

        Returns
        -------
        ModeInfo
            Object containing mode profile and effective index data.
        """

        normal_axis = self.plane.size.index(0.0)

        # note discretizing, need to make consistent
        eps_xx = self.simulation.epsilon(self.plane, "Ex", self.freq)
        eps_yy = self.simulation.epsilon(self.plane, "Ey", self.freq)
        eps_zz = self.simulation.epsilon(self.plane, "Ez", self.freq)

        # make numpy array and get rid of normal axis
        eps_xx = np.squeeze(eps_xx.values, axis=normal_axis)
        eps_yy = np.squeeze(eps_yy.values, axis=normal_axis)
        eps_zz = np.squeeze(eps_zz.values, axis=normal_axis)

        # swap axes to waveguide coordinates (propagating in z)
        eps_wg_zz, (eps_wg_xx, eps_wg_yy) = self.plane.pop_axis(
            (eps_xx, eps_yy, eps_zz), axis=normal_axis
        )

        # note: from this point on, in waveguide coordinates (propagating in z)

        # construct eps_cross section to feed to mode solver
        eps_cross = np.stack((eps_wg_xx, eps_wg_yy, eps_wg_zz))

        Nx, Ny = eps_cross.shape[1:]
        if mode.symmetries[0] != 0:
            eps_cross = np.stack(tuple(e[Nx // 2, :] for e in eps_cross))
        if mode.symmetries[1] != 0:
            eps_cross = np.stack(tuple(e[:, Ny // 2] for e in eps_cross))

        num_modes = mode.num_modes if mode.num_modes else mode.mode_index + 1
        if num_modes <= mode.mode_index:

            raise SetupError(
                f"Mode.mode_index = {mode.mode_index} "
                f"is out of bounds for the number of modes given: Mode.num_modes={mode.um_modes}."
            )

        # note, internally discretizing, need to make consistent.
        field, n_eff_complex = compute_modes(
            eps_cross=eps_cross,
            freq=self.freq,
            grid_size=self.simulation.grid_size,
            pml_layers=mode.num_pml,
            num_modes=num_modes,
            target_neff=mode.target_neff,
            symmetries=mode.symmetries,
            coords=None,
        )

        # Get fields at the Mode.mode_index
        field_values = field[..., mode.mode_index]
        E, H = field_values

        # Handle symmetries
        if mode.symmetries[0] != 0:
            E_half = E[:, 1:, ...]
            H_half = H[:, 1:, ...]
            E = np.concatenate((+E_half[:, ::-1, ...], E_half), axis=1)
            H = np.concatenate((-H_half[:, ::-1, ...], H_half), axis=1)
        if mode.symmetries[1] != 0:
            E_half = E[:, :, 1:, ...]
            H_half = H[:, :, 1:, ...]
            E = np.concatenate((+E_half[:, :, ::-1, ...], E_half), axis=2)
            H = np.concatenate((-H_half[:, :, ::-1, ...], H_half), axis=2)
        Ex, Ey, Ez = E[..., None]
        Hx, Hy, Hz = H[..., None]

        # add in the normal coordinate for each of the fields
        def rotate_field_coords(field_array):
            """move the propagation axis=z to the proper order in the array"""
            return np.moveaxis(field_array, source=2, destination=normal_axis)

        Ex = rotate_field_coords(Ex)
        Ey = rotate_field_coords(Ey)
        Ez = rotate_field_coords(Ez)
        Hx = rotate_field_coords(Hx)
        Hy = rotate_field_coords(Hy)
        Hz = rotate_field_coords(Hz)

        # return the fields and coordinates in the original coordinate system
        Ex, Ey, Ez = self.simulation.unpop_axis(Ez, (Ex, Ey), axis=normal_axis)
        Hx, Hy, Hz = self.simulation.unpop_axis(Hz, (Hx, Hy), axis=normal_axis)

        # apply -1 to H fields if needed, due to how they transform under reflections
        if normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        # note: from this point on, back in original coordinates

        fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

        # note: re-discretizing, need to make consistent.
        data_dict = {}
        for field_name, field in fields.items():
            plane_grid = self.simulation.discretize(self.plane)
            plane_coords = plane_grid[field_name]
            coords = {
                "x": plane_coords.x,
                "y": plane_coords.y,
                "z": plane_coords.z,
                "f": np.array([self.freq]),
            }
            data_dict[field_name] = xr.DataArray(field, coords=coords)

        n_eff_complex = n_eff_complex[mode.mode_index]

        mode_info = ModeInfo(
            field_data=xr.Dataset(data_dict),
            mode=mode,
            n_eff=n_eff_complex.real,
            k_eff=n_eff_complex.imag,
        )

        return mode_info

    def make_source(self, mode: Mode, fwidth: float, direction: Direction) -> ModeSource:
        """Creates ``ModeSource`` from a Mode + additional specifications.

        Parameters
        ----------
        mode : Mode
            ``Mode`` object containing specifications of mode.
        fwidth : float
            Standard deviation of ``GaussianPulse`` of source (Hz).
        direction : Direction
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.

        Returns
        -------
        ModeSource
            Modal source containing specification in ``mode``.
        """
        center = self.plane.center
        size = self.plane.size
        source_time = GaussianPulse(freq0=self.freq, fwidth=fwidth)
        return ModeSource(
            center=center, size=size, source_time=source_time, mode=mode, direction=direction
        )

    def make_monitor(self, mode: Mode, freqs: List[float], name: str) -> ModeMonitor:
        """Creates ``ModeMonitor`` from a Mode + additional specifications.

        Parameters
        ----------
        mode : Mode
            ``Mode`` object containing specifications of mode.
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
        name : str
            Required name of monitor.
        Returns
        -------
        ModeMonitor
            Monitor that measures ``Mode`` on ``plane`` at ``freqs``.
        """
        center = self.plane.center
        size = self.plane.size
        return ModeMonitor(center=center, size=size, freqs=freqs, modes=[mode], name=name)
