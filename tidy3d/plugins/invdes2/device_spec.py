from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as np

import tidy3d as td
import tidy3d.web as web

from .design_region import DesignRegion
from .metric import Metric


@dataclass
class DeviceSpec:
    simulation: td.Simulation
    design_regions: list[DesignRegion]
    metrics: list[Metric]
    name: str

    def get_simulation(self, params: list[np.ndarray]) -> td.Simulation:
        structures = list(self.simulation.structures)
        for param, design_region in zip(params, self.design_regions):
            structure = design_region.to_structure(param)
            structures.append(structure)
        return self.simulation.updated_copy(structures=structures)

    def run_simulation(self, simulation: td.Simulation) -> web.SimulationData:
        return web.run(simulation, task_name=self.name)

    def get_metric(self, sim_data: web.SimulationData) -> float:
        """Get metric from running the simulation"""
        value = 0.0
        for metric in self.metrics:
            value = value + metric.evaluate(sim_data)
        return value

    def get_objective(self, params: list[np.ndarray]) -> float:
        sim = self.get_simulation(params)
        sim_data = self.run_simulation(sim)
        return self.get_metric(sim_data)
