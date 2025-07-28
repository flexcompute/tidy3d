"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.plugins.smatrix.component_modelers.modal import ComponentModeler
from tidy3d.plugins.smatrix.data.data_array import ModalPortDataArray
from tidy3d.plugins.smatrix.ports.types import PortReferenceType


class PortSimulationData(Tidy3dBaseModel):
    ports: tuple[PortReferenceType, ...]
    data: tuple[SimulationData, ...]

    def __getitem__(self, port_name: str) -> SimulationData:
        """
        Allows retrieving simulation data by the port name.

        Args:
            port_name: The string name of the port to look up.

        Returns:
            The SimulationData object corresponding to the given port name.

        Raises:
            KeyError: If no port with the given name is found.
        """
        for i, port_i in enumerate(self.ports):
            if port_i == port_name:
                return self.data[i]
        raise KeyError(f"Port '{port_name}' not found.")


class ComponentModelerData(Tidy3dBaseModel):
    modeler: ComponentModeler = pd.Field(
        ...,
        title="ComponentModeler",
        description="The original :class:`ComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
    )

    data: PortSimulationData = pd.Field(
        ...,
        title="ComponentModeler",
        description="The original :class:`ComponentModeler` object that defines the simulation setup "
        "and from which this data was generated.",
    )

    @cached_property
    def smatrix(self) -> ModalPortDataArray:
        "Stores the computed S-matrix and reference impedances for the terminal ports"
        from tidy3d.plugins.smatrix.analysis.modal import modal_construct_smatrix

        modal_port_data_array = modal_construct_smatrix(modeler_data=self)
        return modal_port_data_array
