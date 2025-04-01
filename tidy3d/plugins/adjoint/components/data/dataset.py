"""Defines jax-compatible datasets."""

from __future__ import annotations

from jax.tree_util import register_pytree_node_class
from pydantic import Field

from tidy3d.components.data.dataset import PermittivityDataset
from tidy3d.plugins.adjoint.components.base import JaxObject

from .data_array import JaxDataArray


@register_pytree_node_class
class JaxPermittivityDataset(PermittivityDataset, JaxObject):
    """A :class:`.PermittivityDataset` registered with jax."""

    _tidy3d_class = PermittivityDataset

    eps_xx: JaxDataArray = Field(
        title="Epsilon xx",
        description="Spatial distribution of the xx-component of the relative permittivity.",
        jax_field=True,
    )
    eps_yy: JaxDataArray = Field(
        title="Epsilon yy",
        description="Spatial distribution of the yy-component of the relative permittivity.",
        jax_field=True,
    )
    eps_zz: JaxDataArray = Field(
        title="Epsilon zz",
        description="Spatial distribution of the zz-component of the relative permittivity.",
        jax_field=True,
    )
