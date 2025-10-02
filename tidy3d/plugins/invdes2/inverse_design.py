from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as np

import tidy3d as td
import tidy3d.web as web

from .device_spec import DeviceSpec
from .optimizer_spec import OptimizerSpec


@dataclass
class InverseDesign:
    optimizer_spec: OptimizerSpec
    device_specs: list[DeviceSpec]

    # TODO: ensure that all device specs have unique names

    def get_simulations(self, params: list[list[np.ndarray]]) -> dict[str, td.Simulation]:
        simulations = {}
        for param, device_spec in zip(params, self.device_specs):
            simulation = device_spec.get_simulation(param)
            simulations[device_spec.name] = simulation
        return simulations

    def run_simulations(self, sims: dict[str, td.Simulation]) -> web.BatchData:
        return web.run_async(sims)

    def get_metric(self, batch_data: web.BatchData) -> float:
        value = 0.0
        for device_spec in self.device_specs:
            sim_data = batch_data[device_spec.name]
            value = value + device_spec.get_metric(sim_data)
        return value

    def get_objective(self, params: list[list[np.ndarray]]) -> float:
        sims = self.get_simulations(params)
        batch_data = self.run_simulations(sims)
        return self.get_metric(batch_data)
