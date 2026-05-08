"""Defines heat simulation class
NOTE: Keeping this class for backward compatibility only"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.log import log


class HeatSimulation(HeatChargeSimulation):
    """
    Contains all information about heat simulation.

    Example
    -------
    >>> import tidy3d as td
    >>> heat_sim = td.HeatSimulation( # doctest: +SKIP
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         td.Structure(
    ...             geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=td.Medium(
    ...                 permittivity=2.0, heat_spec=td.SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=td.Medium(permittivity=3.0, heat_spec=td.FluidSpec()),
    ...     grid_spec=td.UniformUnstructuredGrid(
    ...         dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2
    ...     ),
    ...     sources=[td.HeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         td.HeatChargeBoundarySpec(
    ...             placement=td.StructureBoundary(structure="box"),
    ...             condition=td.TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[td.TemperatureMonitor(size=(1, 2, 3), name="sample", unstructured=True)],
    ... )
    """

    @model_validator(mode="before")
    @classmethod
    def issue_warning_deprecated(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Issue warning for 'HeatSimulations'."""
        log.warning(
            "Setting up deprecated 'HeatSimulation'. "
            "Consider defining 'HeatChargeSimulation' instead."
        )
        return data
