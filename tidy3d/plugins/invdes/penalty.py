# define penalties applied to parameters from design region
import typing
import abc

import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint.utils.penalty as adjoint_penalties

# make these classes importable through `tidy3d.plugins.invdes`
RadiusPenalty = adjoint_penalties.RadiusPenalty
ErosionDilationPenalty = adjoint_penalties.ErosionDilationPenalty


# template for user to define penalty
class Penalty(td.components.base.Tidy3dBaseModel, abc.ABC):
    """Arbitrary penalty that a user can define."""

    @abc.abstractmethod
    def evaluate(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the penalty on some material density data."""


# define the allowable penalties types for the `DesignRegion.transformations` field
PenaltyType = typing.Union[RadiusPenalty, ErosionDilationPenalty, Penalty]
