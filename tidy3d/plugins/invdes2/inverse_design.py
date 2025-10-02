"""Inverse design orchestration utilities.

Coordinates multiple device scenarios to evaluate a scalar objective by
constructing simulations from parameter vectors, submitting them to Tidy3D Web
in batch, and aggregating metric values across devices. Compatible with
autograd-enabled Tidy3D Web.
"""

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

    def __post_init__(self) -> None:
        """Validate that all `DeviceSpec` names are unique."""
        names = [device_spec.name for device_spec in self.device_specs]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                "All DeviceSpec names must be unique; duplicates found: " + ", ".join(duplicates)
            )

    def get_simulations(self, params: list[list[np.ndarray]]) -> dict[str, td.Simulation]:
        """Build simulations for each device from parameter vectors."""
        simulations = {}
        for param, device_spec in zip(params, self.device_specs):
            simulation = device_spec.get_simulation(param)
            simulations[device_spec.name] = simulation
        return simulations

    def run_simulations(self, sims: dict[str, td.Simulation]) -> web.BatchData:
        """Execute multiple simulations concurrently via Tidy3D Web."""
        return web.run_async(sims)

    def get_metric(self, batch_data: web.BatchData) -> float:
        """Aggregate metrics across all devices."""
        value = 0.0
        for device_spec in self.device_specs:
            sim_data = batch_data[device_spec.name]
            value = value + device_spec.get_metric(sim_data)
        return value

    def get_objective(self, params: list[list[np.ndarray]]) -> float:
        """Evaluate the scalar objective across all devices given parameters."""
        sims = self.get_simulations(params)
        batch_data = self.run_simulations(sims)
        return self.get_metric(batch_data)
