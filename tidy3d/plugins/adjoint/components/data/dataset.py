"""Defines jax-compatible datasets."""
import pydantic as pd

from jax.tree_util import register_pytree_node_class

from .....components.data.dataset import PermittivityDataset
from .data_array import JaxDataArray
from ..base import JaxObject


@register_pytree_node_class
class JaxPermittivityDataset(PermittivityDataset, JaxObject):
    """A :class:`.PermittivityDataset` registered with jax."""

    eps_xx: JaxDataArray = pd.Field(
        ...,
        title="Epsilon xx",
        description="Spatial distribution of the xx-component of the relative permittivity.",
        jax_field=True,
    )
    eps_yy: JaxDataArray = pd.Field(
        ...,
        title="Epsilon yy",
        description="Spatial distribution of the yy-component of the relative permittivity.",
        jax_field=True,
    )
    eps_zz: JaxDataArray = pd.Field(
        ...,
        title="Epsilon zz",
        description="Spatial distribution of the zz-component of the relative permittivity.",
        jax_field=True,
    )
