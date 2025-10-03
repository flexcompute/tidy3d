from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as np

import tidy3d as td
import tidy3d.web as web

from .design_region import DesignRegionType
from .metric import MetricType


@dataclass
class DeviceSpec:
    """Specification of a single device scenario for inverse design.

    Attributes
    ----------
    simulation:
        Base `td.Simulation` onto which parameterized structures are appended.
    design_regions:
        Ordered list of `DesignRegion` instances. Each must consume a
        corresponding parameter array in `get_simulation`.
    metrics:
        List of `Metric` instances whose weighted sum forms the device's
        objective contribution.
    name:
        Unique identifier for this device, used as the `task_name` and as the
        key in batch results.
    """

    simulation: td.Simulation
    design_regions: list[DesignRegionType]
    metrics: list[MetricType]
    name: str

    def get_simulation(self, params: list[np.ndarray]) -> td.Simulation:
        """Construct a simulation by appending parameterized structures.

        Parameters
        ----------
        params:
            List of arrays matching `design_regions` order; each array is passed
            to the corresponding region's `to_structure` to produce a structure.

        Returns
        -------
        td.Simulation
            A new simulation with appended structures.
        """
        structures = list(self.simulation.structures)
        for param, design_region in zip(params, self.design_regions):
            structure = design_region.to_structure(param)
            structures.append(structure)
        return self.simulation.updated_copy(structures=structures)

    def run_simulation(self, simulation: td.Simulation) -> web.SimulationData:
        """Run the simulation via Tidy3D Web and return results."""
        return web.run(simulation, task_name=self.name)

    def get_metric(self, sim_data: web.SimulationData) -> float:
        """Compute the weighted sum of metrics for this device.

        Parameters
        ----------
        sim_data:
            Simulation results to be consumed by each metric.

        Returns
        -------
        float
            Weighted sum of metric values.
        """
        value = 0.0
        for metric in self.metrics:
            mnt_data = sim_data[metric.monitor_name]
            value = value + metric.weight * metric.evaluate(mnt_data)
        return value

    def get_objective(self, params: list[np.ndarray]) -> float:
        """Build, run, and score this device for the given parameters."""
        sim = self.get_simulation(params)
        sim_data = self.run_simulation(sim)
        return self.get_metric(sim_data)

    @property
    def parameter_shape(self) -> list[int]:
        """Return the shape of the parameters for each design region."""
        return [design_region.parameter_shape for design_region in self.design_regions]

    def ones(self, **kwargs) -> list[np.ndarray]:
        """Return a list of arrays of ones with the shape of the parameters for each design region."""
        return [design_region.ones(**kwargs) for design_region in self.design_regions]
