from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tidy3d as td
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.parallel_adjoint_bases import ParallelAdjointBasis


@dataclass(frozen=True)
class ParallelAdjointState:
    """Typed cache for locally precomputed parallel-adjoint runtime state."""

    task_name: str
    num_sims: int
    basis_specs: list[ParallelAdjointBasis]
    basis_maps: dict[ParallelAdjointBasis, dict[str, AutogradFieldMap]]
    basis_task_map: dict[ParallelAdjointBasis, str]

    @property
    def has_parallel_state(self) -> bool:
        """Whether the state contains populated parallel-adjoint basis information."""
        return bool(self.basis_specs) or bool(self.basis_maps) or bool(self.basis_task_map)


@dataclass
class AutogradContext:
    """Typed runtime state container for autograd execution."""

    simulation_data_original: td.SimulationData | None = None
    simulation_data_forward: td.SimulationData | None = None
    forward_task_id: str | None = None
    parallel_adjoint_state: ParallelAdjointState | None = None
