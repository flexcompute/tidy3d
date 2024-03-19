# define transformations applied to parameters from design region
import typing
import abc

import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint.utils.filter as adjoint_filters

# make these classes importable through `tidy3d.plugins.invdes`
CircularFilter = adjoint_filters.CircularFilter
ConicFilter = adjoint_filters.ConicFilter
BinaryProjector = adjoint_filters.BinaryProjector

# template for user to define transformations
class Transformation(td.components.base.Tidy3dBaseModel, abc.ABC):
    """Arbitrary transformation that a user can define."""

    @abc.abstractmethod
    def evaluate(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate some parameter data."""

# define the allowable transformation types for the `DesignRegion.transformations` field
TransformationType = typing.Union[CircularFilter, ConicFilter, BinaryProjector, Transformation]
