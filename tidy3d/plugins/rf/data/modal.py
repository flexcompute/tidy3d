from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.data.sim_data import SimulationData


class PortSimulationData(pd.BaseModel):
    ports: tuple[str, ...]
    data: tuple[SimulationData, ...]

    def __getitem__(self, port_name: str) -> SimulationData:
        for i, port_i in enumerate(self.ports):
            if port_i == port_name:
                return self.data[i]
        raise KeyError(f"Port '{port_name}' not found.")
