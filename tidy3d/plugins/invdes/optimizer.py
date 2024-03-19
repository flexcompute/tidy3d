# specification for running the optimizer

import typing

import optax

import jax.numpy as jnp
import pydantic.v1 as pd

import tidy3d as td


class Optimizer(td.components.base.Tidy3dBaseModel):
    """Specification for an optimization."""

    learning_rate: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Learning Rate",
        description="Step size for the gradient descent optimizer.",
    )

    num_steps: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Steps",
        description="Number of steps in the gradient descent optimizer.",
    )

    states: typing.List[typing.Any] = []

    # TODO: beta schedule
    # TODO: penalty schedule
