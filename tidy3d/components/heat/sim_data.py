"""Defines heat simulation data class"""
from __future__ import annotations

import pydantic as pd

from .simulation import HeatSimulation

from ..data.data_array import SpatialDataArray
from ..base import Tidy3dBaseModel
from ..scene import Scene

from ...constants import KELVIN


class HeatSimulationData(Tidy3dBaseModel):
    """Stores results of a heat simulation.

    Example
    -------
    >>> from tidy3d import Medium, SolidSpec, FluidSpec, UniformHeatGrid, Scene, SpatialDataArray
    >>> from tidy3d import Structure, Box, UniformHeatGrid, UniformHeatSource, HeatBoundarySpec
    >>> from tidy3d import StructureBoundary, TemperatureBC
    >>> import numpy as np
    >>> scene = Scene(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(
    ...                 permittivity=2.0, heat_spec=SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=Medium(permittivity=3.0, heat_spec=FluidSpec()),
    ... )
    >>> heat_sim = HeatSimulation(
    ...     scene=scene,
    ...     grid_spec=UniformHeatGrid(dl=0.1),
    ...     heat_sources=[UniformHeatSource(rate=1, structures=["box"])],
    ...     boundary_specs=[
    ...         HeatBoundarySpec(
    ...             placement=StructureBoundary(structure="box"),
    ...             condition=TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     heat_domain=Box(size=(2, 2, 2)),
    ... )
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> coords = dict(x=x, y=y, z=z)
    >>> temperature_data = SpatialDataArray((1+1j) * np.random.random((2,3,4)), coords=coords)
    >>> heat_sim_data = HeatSimulationData(
    ...     heat_simulation=heat_sim, temperature_data=temperature_data,
    ... )
    """

    heat_simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="``HeatSimulation`` object describing the problem setup.",
    )

    temperature_data: SpatialDataArray = pd.Field(
        title="Temperature Field",
        description="Temperature field obtained from heat simulation.",
        units=KELVIN,
    )

    def perturbed_mediums_scene(self) -> Scene:
        """Apply heat data to the original Tidy3D simulation (replaces appropriate media with CustomMedia). """

        return self.heat_simulation.scene.perturbed_mediums_copy(temperature=self.temperature_data)

