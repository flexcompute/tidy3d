from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import autograd.numpy as np

import tidy3d as td


@dataclass
class DesignRegion:
    @abstractmethod
    def to_structure(self, params: np.ndarray) -> td.Structure:
        pass
