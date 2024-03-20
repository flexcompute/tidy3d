# specification for running the optimizer


import pydantic.v1 as pd
import optax

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

    # optional kwargs passed to ``optax.adam()``

    b1: float = pd.Field(
        0.9,
        title="Beta 1",
        description="Beta 1 parameter in the Adam optimization method.",
    )

    b2: float = pd.Field(
        0.999,
        title="Beta 2",
        description="Beta 2 parameter in the Adam optimization method.",
    )

    eps: float = pd.Field(
        1e-8,
        title="Epsilon",
        description="Epsilon parameter in the Adam optimization method.",
    )

    @property
    def optax_optimizer(self) -> optax.GradientTransformationExtraArgs:
        """The optimizer used by ``optax`` corresponding to this spec."""
        return optax.adam(
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
        )

    # TODO: beta schedule
    # TODO: penalty schedule
