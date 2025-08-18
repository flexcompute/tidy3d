from __future__ import annotations

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.simulaton_types import SimulationDataType


class IndexSimulationData(Tidy3dBaseModel):
    index: tuple[str, ...]
    data: tuple[SimulationDataType, ...]

    def __getitem__(self, index: str) -> SimulationDataType:
        """
        Allows retrieving simulation data by the port name.

        Args:
            port_name: The string name of the port to look up.

        Returns:
            The SimulationData object corresponding to the given port name.

        Raises:
            KeyError: If no port with the given name is found.
        """
        for i, index_i in enumerate(self.index):
            if index_i == index:
                return self.data[i]
        raise KeyError(f"Index '{index}' not found.")
