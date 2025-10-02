from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerSpec:
    """Hyperparameters describing the optimization loop to be used externally."""

    learning_rate: float
    num_steps: int
