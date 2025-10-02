from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import autograd.numpy as np

import tidy3d as td


@dataclass
class DesignRegion:
    """Abstract parameterized geometry provider for inverse design.

    Implementations transform parameter arrays into concrete `td.Structure`
    instances that will be appended to a base simulation.
    """

    @abstractmethod
    def to_structure(self, params: np.ndarray) -> td.Structure:
        """Return a `td.Structure` built from the provided parameters."""
