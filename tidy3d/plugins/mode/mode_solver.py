""" Turn Mode Specifications into Mode profiles """

import numpy as np
from pydantic import BaseModel

from ...components import Box
from ...components import Simulation
from ...components import Mode
from ...components import FieldData
from ...components import FreqSampler, ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components import eps_complex_to_nk
from ...components.validators import assert_plane

from .solver import compute_modes


"""
Stage:                Simulation     Mode Specs    Outputs       Viz           Export
                      ----------  +  ---------- -> ----------- -> ---------- -> ----------
Method:                        __init__()          .solve()       .plot()       .export()

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

src.export('data/my_source.json')             # this source /monitor can be saved to file
src = ModeSource.load('data/my_source.json')  # and loaded in our script
"""


class ModeInfo(BaseModel):
    """stores information about a (solved) mode"""

    field_data: FieldData
    mode: Mode
    n_eff: float
    k_eff: float


class ModeSolver:
    """inferface for finding mode specification for ModeSource and ModeMonitor objects"""

    def __init__(self, simulation: Simulation, plane: Box, freq: float):
        """makes a mode solver object"""

        self.simulation = simulation
        self.plane = plane
        self.freq = freq

        assert 0.0 in plane.size, "plane must have at least one axis with size=0"

    def solve(self, mode: Mode) -> ModeInfo:
        """gets information about the mode specification from mode solver"""

        eps_cross = np.squeeze(self.simulation.epsilon(self.plane, self.freq))
        target_neff = np.mean(eps_complex_to_nk(eps_cross))
        plane_indices = [index for index in range(3) if self.plane.size[index] > 0.0]
        symmetries = [self.simulation.symmetry[i] for i in plane_indices]
        pml_layers = [self.simulation.pml_layers[i].num_layers for i in plane_indices]

        field, n_eff_complex = compute_modes(
            eps_cross=eps_cross,
            freq=self.freq,
            grid_size=self.simulation.grid_size,
            pml_layers=pml_layers,
            num_modes=mode.mode_index + 1,
            target_neff=target_neff,
            symmetries=symmetries,
            coords=None,
        )

        # field.shape = (2, 3, Nx, Ny, 1, Nmodes)
        n_eff_complex = n_eff_complex[mode.mode_index]
        field_values = field[..., mode.mode_index]
        Nx = field_values.shape[2]
        Ny = field_values.shape[3]
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.plane._get_bounds()
        xs = np.linspace(xmin, xmax, Nx)
        ys = np.linspace(ymin, ymax, Ny)
        zs = np.linspace(zmin, zmax, 1)

        field_data = FieldData(
            monitor_name="mode_solver",
            sampler_label="f",
            sampler_values=[self.freq],
            values=field_values[..., None],
            x=xs,
            y=ys,
            z=zs,
        )

        return ModeInfo(
            field_data=field_data,
            mode=mode,
            n_eff=n_eff_complex.real,
            k_eff=n_eff_complex.imag,
        )

    def make_source(self, mode: Mode, fwidth: float) -> ModeSource:
        """creates ModeMonitor from a Mode"""
        center = self.plane.center
        size = self.plane.size
        source_time = GaussianPulse(freq0=self.freq, fwidth=fwidth)
        return ModeSource(center=center, size=size, source_time=source_time, modes=[mode])

    def make_monitor(self, mode: Mode) -> ModeMonitor:
        """creates ModeMonitor from a Mode"""
        center = self.plane.center
        size = self.plane.size
        sampler = FreqSampler(freqs=[self.freq])
        return ModeMonitor(center=center, size=size, sampler=sampler, modes=[mode])
