"""Defines a jax-compatible structure and its conversion to a gradient monitor."""
from __future__ import annotations

import pydantic as pd
import numpy as np
from jax.tree_util import register_pytree_node_class

from ....constants import C_0
from ....components.structure import Structure
from ....components.monitor import FieldMonitor
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.types import Bound

from .base import JaxObject
from .medium import JaxMediumType, JAX_MEDIUM_MAP
from .geometry import JaxGeometryType, JAX_GEOMETRY_MAP


@register_pytree_node_class
class JaxStructure(Structure, JaxObject):
    """A :class:`.Structure` registered with jax."""

    geometry: JaxGeometryType = pd.Field(
        ...,
        title="Geometry",
        description="Geometry of the structure, which is jax-compatible.",
        jax_field=True,
    )

    medium: JaxMediumType = pd.Field(
        ...,
        title="Medium",
        description="Medium of the structure, which is jax-compatible.",
        jax_field=True,
    )

    def to_structure(self) -> Structure:
        """Convert :class:`.JaxStructure` instance to :class:`.Structure`"""
        self_dict = self.dict(exclude={"type", "geometry", "medium"})
        self_dict["geometry"] = self.geometry.to_tidy3d()
        self_dict["medium"] = self.medium.to_medium()
        return Structure.parse_obj(self_dict)

    @classmethod
    def from_structure(cls, structure: Structure) -> JaxStructure:
        """Convert :class:`.Structure` to :class:`.JaxStructure`."""

        # get the appropriate jax types corresponding to the td.Structure fields
        jax_geometry_type = JAX_GEOMETRY_MAP[type(structure.geometry)]
        jax_medium_type = JAX_MEDIUM_MAP[type(structure.medium)]

        # load them into the JaxStructure dictionary and parse it into an instance
        struct_dict = structure.dict(exclude={"type", "geometry", "medium"})
        struct_dict["geometry"] = jax_geometry_type.from_tidy3d(structure.geometry)
        struct_dict["medium"] = jax_medium_type.from_tidy3d(structure.medium)

        return cls.parse_obj(struct_dict)

    # pylint:disable=too-many-arguments
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        eps_out: complex,
        num_proc: int = 1,
    ) -> JaxStructure:
        """Returns the gradient of the structure parameters given forward and adjoint field data."""

        # compute wavelength in material (to use for determining integration points)
        freq = float(grad_data_eps.eps_xx.f)
        wvl_free_space = C_0 / freq
        eps_in = self.medium.eps_model(frequency=freq)
        ref_ind = np.sqrt(np.max(np.real(eps_in)))
        wvl_mat = wvl_free_space / ref_ind

        geo_vjp = self.geometry.store_vjp(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            grad_data_eps=grad_data_eps,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            eps_out=eps_out,
            eps_in=eps_in,
            num_proc=num_proc,
        )

        medium_vjp = self.medium.store_vjp(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            inside_fn=self.geometry.inside,
        )

        return self.copy(update=dict(geometry=geo_vjp, medium=medium_vjp))

    def make_grad_monitors(self, freq: float, name: str) -> FieldMonitor:
        """Return gradient monitor associated with this object."""
        return self.geometry.make_grad_monitors(freq=freq, name=name)
