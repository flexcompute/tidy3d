""" Turn Mode Specifications into Mode profiles """

import numpy as np

from ...components import Box
from ...components import Simulation
from ...components import Mode
from ...components import FreqSampler, ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components import eps_complex_to_nk

from .solver import compute_modes, ModeInfo


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


class ModeSolver:
    """inferface for finding mode specification for ModeSource and ModeMonitor objects"""

    def __init__(self, simulation: Simulation, plane: Box, freq: float):
        """makes a mode solver object"""

        self.simulation = simulation
        self.plane = plane
        self.freq = freq

    def solve(self, mode: Mode) -> ModeInfo:
        """gets information about the mode specification from mode solver"""

        # to do
        eps_cross = self.simulation.epsilon(self.plane)
        pml_layers = [p.num_layers for p in self.simulation.pml_layers]
        target_neff = np.mean(eps_complex_to_nk(eps_cross))
        plane_indices = [index for index in range(3) if self.plane.size[index] > 0.0]
        symmetries = [self.simulation.symmetries[i] for i in plane_indices]

        return compute_modes(
            eps_cross=eps_cross,
            freq=self.freq,
            grid_size=self.simulation.grid_size,
            pml_layers=pml_layers,
            num_modes=mode.mode_index,
            target_neff=target_neff,
            symmetries=symmetries,
            coords=None,
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
