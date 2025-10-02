from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerSpec:
    learning_rate: float
    num_steps: int
