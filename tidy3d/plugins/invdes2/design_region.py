from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Union

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

    @property
    @abstractmethod
    def parameter_shape(self) -> int:
        """Return the (flattened) shape of the parameters for this design region."""


@dataclass
class TopologyDesignRegion(DesignRegion):
    """Design region as a pixellated permittivity grid."""

    size: tuple[float, float, float]
    center: tuple[float, float, float]
    eps_bounds: tuple[float, float]
    pixel_size: float

    @property
    def shape_3d(self) -> tuple[int, int, int]:
        """Return the shape of the parameters for this design region."""
        return tuple(int(np.ceil(size / self.pixel_size)) for size in self.size)

    @property
    def parameter_shape(self) -> int:
        """Return the shape of the parameters for this design region."""
        return int(np.prod(self.shape_3d))

    def to_structure(self, params: np.ndarray) -> td.Structure:
        """Return a `td.Structure` built from the provided parameters."""

        geometry = td.Box(center=self.center, size=self.size)

        # TODO: add transformations
        eps_data = params.reshape(self.shape_3d)

        return td.Structure.from_permittivity_array(
            geometry=geometry, eps_data=eps_data, eps_bounds=self.eps_bounds
        )


DesignRegionType = Union[TopologyDesignRegion]
