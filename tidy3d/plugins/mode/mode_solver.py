""" Turn Mode Specifications into Mode profiles """

from typing import List

import numpy as np
from pydantic import BaseModel

from ...components import Box
from ...components import Simulation
from ...components import Mode
from ...components import FieldData
from ...components import FreqSampler, ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components import eps_complex_to_nk
from ...components.types import Direction

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

        # note discretizing, need to make consistent
        eps_cross = self.simulation.epsilon(self.plane, self.freq)

        # note, internally discretizing, need to make consistent.
        field, n_eff_complex = compute_modes(
            eps_cross=np.squeeze(eps_cross),
            freq=self.freq,
            grid_size=self.simulation.grid_size,
            pml_layers=mode.num_pml,
            num_modes=mode.mode_index + 1,
            target_neff=mode.target_neff,
            symmetries=(0, 0),  # note: mode symmetries not handled yet.
            coords=None,
        )

        # field.shape = (2, 3, Nx, Ny, 1, Nmodes)
        field_values = field[..., mode.mode_index]

        # note: re-discretizing, need to make consistent.
        (_, _, Nx, Ny, _) = field_values.shape
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.plane.get_bounds()
        xs = np.linspace(xmin, xmax, Nx)
        ys = np.linspace(ymin, ymax, Ny)
        zs = np.linspace(zmin, zmax, 1)

        n_eff_complex = n_eff_complex[mode.mode_index]

        field_data = FieldData(
            monitor_name="mode_solver_output",
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

    def make_source(self, mode: Mode, fwidth: float, direction: Direction) -> ModeSource:
        """creates ModeSource from a Mode + additional specs"""
        center = self.plane.center
        size = self.plane.size
        source_time = GaussianPulse(freq0=self.freq, fwidth=fwidth)
        return ModeSource(
            center=center, size=size, source_time=source_time, mode=mode, direction=direction
        )

    def make_monitor(self, mode: Mode, freqs: List[float]) -> ModeMonitor:
        """creates ModeMonitor from a Mode + additional specs"""
        center = self.plane.center
        size = self.plane.size
        sampler = FreqSampler(freqs=freqs)
        return ModeMonitor(center=center, size=size, sampler=sampler, modes=[mode])
