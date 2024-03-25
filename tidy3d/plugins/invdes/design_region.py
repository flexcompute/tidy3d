# container for everything defining the inverse design
import abc
import typing

import jax.numpy as jnp
import numpy as np
import jax

import tidy3d as td
from tidy3d.components.types import annotate_type, Symmetry
import tidy3d.plugins.adjoint as tda

from .transformation import (
    TransformationType,
)
from .penalty import PenaltyType, ErosionDilationPenalty, RadiusPenalty, Penalty


class DesignRegion(td.Box, abc.ABC):
    params_shape: typing.Tuple[int, int, int]
    symmetry: typing.Tuple[Symmetry, Symmetry, Symmetry] = (0, 0, 0)
    eps_bounds: typing.Tuple[float, float]
    transformations: typing.Tuple[annotate_type(TransformationType), ...] = ()
    penalties: typing.Tuple[annotate_type(PenaltyType), ...] = ()
    penalty_weights: typing.Tuple[float, ...] = None

    @property
    def geometry(self) -> td.Box:
        """Geometry for this design region."""
        return td.Box(center=self.center, size=self.size)

    def material_density(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a dataset."""
        for transformation in self.transformations:
            data = transformation.evaluate(data)
        return data

    @property
    def _num_penalties(self) -> int:
        """How many penalties present?"""
        return len(self.penalties)

    @property
    def _penalty_weights(self) -> jnp.ndarray:
        """Penalty weights as an array."""

        if not self._num_penalties:
            raise ValueError("Can't get penalty weights because `penalties` are not defined.")

        if self.penalty_weights is None:
            return jnp.ones(self._num_penalties)

        return jnp.array(self.penalty_weights)

    def penalty_value(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a dataset."""

        if not self.penalties:
            return 0.0

        # sum the penalty values scaled by their weights (optional)
        material_density = self.material_density(data)
        penalty_values = [penalty.evaluate(material_density) for penalty in self.penalties]
        penalties_weighted = self._penalty_weights * jnp.array(penalty_values)
        return jnp.sum(penalties_weighted)

    @abc.abstractmethod
    def make_structure(self) -> tda.JaxStructure:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``."""


class TopologyDesignRegion(DesignRegion):
    transformations: typing.Tuple[annotate_type(TransformationType), ...] = ()
    penalties: typing.Tuple[annotate_type(typing.Union[ErosionDilationPenalty, Penalty]), ...] = ()

    @property
    def step_sizes(self) -> typing.Tuple[float, float, float]:
        """Step sizes along x, y, z."""
        bounds = np.array(self.bounds)
        return tuple((bounds[1] - bounds[0]).tolist())

    @property
    def coords(self) -> typing.Dict[str, typing.List[float]]:
        """Coordinates for the custom medium corresponding to this design region."""

        rmin, rmax = self.bounds

        coords = dict()

        # for i, (coord_key, ptmin, ptmax, num_pts) in enumerate(zip("xyz", rmin, rmax, self.params_shape)):
        for (center, coord_key, ptmin, ptmax, num_pts) in zip(self.center, "xyz", rmin, rmax, self.params_shape):
            size = ptmax - ptmin
            if np.isinf(size):
                coord_vals = num_pts * [center]
            else:
                step_size = size / num_pts
                coord_vals = np.linspace(ptmin + step_size / 2, ptmax - step_size / 2, num_pts).tolist()
            coords[coord_key] = coord_vals

        coords["f"] = [td.C_0]  # TODO: is this a safe choice?
        return coords

    def eps_values(self, params: jnp.ndarray) -> jnp.ndarray:
        """Values for the custom medium permittivity."""
        material_density = self.material_density(params)
        eps_min, eps_max = self.eps_bounds
        arr_3d = eps_min + material_density * (eps_max - eps_min)
        arr_3d = jax.lax.stop_gradient(arr_3d)
        arr_3d = arr_3d.reshape(params.shape)
        return jnp.expand_dims(arr_3d, axis=-1)

    def make_structure(self, params: jnp.ndarray) -> tda.JaxStructureStaticGeometry:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``."""
        coords = self.coords
        eps_values = self.eps_values(params)
        data_array = tda.JaxDataArray(values=eps_values, coords=coords)
        field_components = {f"eps_{dim}{dim}": data_array for dim in "xyz"}
        eps_dataset = tda.JaxPermittivityDataset(**field_components)
        medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)
        return tda.JaxStructureStaticGeometry(geometry=self.geometry, medium=medium)


class ShapeDesignRegion(DesignRegion):
    transformations: typing.Literal[()] = ()
    penalties: typing.Tuple[annotate_type(typing.Union[RadiusPenalty, Penalty]), ...] = ()


class LevelSetDesignRegion(DesignRegion):
    """Implement later"""

    pass
