"""Data structures for post-processing modal component simulations to calculate S-matrices."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.data_array import DataArray
from tidy3d.components.data.index import SimulationDataMap
from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler


class AbstractComponentModelerData(ABC, Tidy3dBaseModel):
    """A data container for the results of a :class:`.AbstractComponentModeler` run.

    Notes
    -----
        This class stores the original modeler and the simulation data obtained
        from running the simulations it defines. It also provides a method to
        compute the S-matrix from the simulation data.
    """

    modeler: AbstractComponentModeler = pd.Field(
        ...,
        title="Component modeler",
        description="The original :class:`AbstractComponentModeler` object that defines the "
        "simulation setup and from which this data was generated.",
    )

    data: SimulationDataMap = pd.Field(
        ...,
        title="SimulationDataMap",
        description="A mapping from task names to :class:`.SimulationData` objects, "
        "containing the results of each simulation run.",
    )

    log: str = pd.Field(
        None,
        title="Modeler Post-process Log",
        description="A string containing the log information from the modeler post-processing run.",
    )

    @abstractmethod
    def smatrix(self) -> DataArray:
        """Computes and returns the scattering matrix (S-matrix)."""

    @pd.validator("data")
    def keys_match_modeler(cls, val, values):
        """
        Validates that the keys of the 'data' dictionary match the keys
        of the 'modeler.sim_dict' dictionary, irrespective of order.
        """
        modeler = values.get("modeler")

        # It's good practice to handle cases where 'modeler' might not be present
        if not modeler or not hasattr(modeler, "sim_dict"):
            return val

        # Use sets for an order-insensitive comparison
        modeler_keys = set(modeler.sim_dict.keys())
        data_keys = set(val.keys())

        if modeler_keys != data_keys:
            # Provide a more helpful error by showing the exact differences
            missing_keys = sorted(modeler_keys - data_keys)
            extra_keys = sorted(data_keys - modeler_keys)

            error_parts = []
            if missing_keys:
                error_parts.append(f"Data is missing keys: {missing_keys}")
            if extra_keys:
                error_parts.append(f"Data has extra keys: {extra_keys}")

            raise ValueError(f"Key mismatch between modeler and data. {'; '.join(error_parts)}")

        return val
