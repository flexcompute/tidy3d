from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerSpec:
    """Hyperparameters describing the optimization loop to be used externally."""

    num_steps: int
    learning_rate: float
