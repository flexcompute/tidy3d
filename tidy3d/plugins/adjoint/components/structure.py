"""Defines a jax-compatible structure and its conversion to a gradient monitor."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pydantic.v1 as pd
from jax.tree_util import register_pytree_node_class

from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.geometry.utils import GeometryType
from ....components.medium import MediumType
from ....components.monitor import FieldMonitor
from ....components.structure import Structure
from ....components.types import TYPE_TAG_STR, Bound
from ....constants import C_0
from .base import JaxObject
from .geometry import JAX_GEOMETRY_MAP, JaxBox, JaxGeometryType
from .medium import JAX_MEDIUM_MAP, JaxMediumType

GEO_MED_MAPPINGS = dict(geometry=JAX_GEOMETRY_MAP, medium=JAX_MEDIUM_MAP)


class AbstractJaxStructure(Structure, JaxObject):
    """A :class:`.Structure` registered with jax."""

    _tidy3d_class = Structure

    # which of "geometry" or "medium" is differentiable for this class
    _differentiable_fields = ()

    geometry: Union[JaxGeometryType, GeometryType]
    medium: Union[JaxMediumType, MediumType]

    @pd.validator("medium", always=True)
    def _check_2d_geometry(cls, val, values):
        """Override validator checking 2D geometry, which triggers unnecessarily for gradients."""
        return val

    def _validate_web_adjoint(self) -> None:
        """Run validators for this component, only if using ``tda.web.run()``."""
        if "geometry" in self._differentiable_fields:
            self.geometry._validate_web_adjoint()
        if "medium" in self._differentiable_fields:
            self.medium._validate_web_adjoint()

    @property
    def jax_fields(self):
        """The fields that are jax-traced for this class."""
        return dict(geometry=self.geometry, medium=self.medium)

    @property
    def exclude_fields(self):
        """Fields to exclude from the self dict."""
        return set(["type"] + list(self.jax_fields.keys()))

    def to_structure(self) -> Structure:
        """Convert :class:`.JaxStructure` instance to :class:`.Structure`"""
        self_dict = self.dict(exclude=self.exclude_fields)
        for key, component in self.jax_fields.items():
            if key in self._differentiable_fields:
                self_dict[key] = component.to_tidy3d()
            else:
                self_dict[key] = component
        return Structure.parse_obj(self_dict)

    @classmethod
    def from_structure(cls, structure: Structure) -> JaxStructure:
        """Convert :class:`.Structure` to :class:`.JaxStructure`."""

        struct_dict = structure.dict(exclude={"type"})

        jax_fields = dict(geometry=structure.geometry, medium=structure.medium)

        for key, component in jax_fields.items():
            if key in cls._differentiable_fields:
                type_map = GEO_MED_MAPPINGS[key]
                jax_type = type_map[type(component)]
                struct_dict[key] = jax_type.from_tidy3d(component)
            else:
                struct_dict[key] = component

        return cls.parse_obj(struct_dict)

    def make_grad_monitors(self, freqs: List[float], name: str) -> FieldMonitor:
        """Return gradient monitor associated with this object."""
        if "geometry" not in self._differentiable_fields:
            # make a fake JaxBox to be able to call .make_grad_monitors
            rmin, rmax = self.geometry.bounds
            geometry = JaxBox.from_bounds(rmin=rmin, rmax=rmax)
        else:
            geometry = self.geometry
        return geometry.make_grad_monitors(freqs=freqs, name=name)

    def _get_medium_params(
        self,
        grad_data_eps: PermittivityData,
    ) -> Dict[str, float]:
        """Compute params in the material of this structure."""
        freq_max = float(max(grad_data_eps.eps_xx.f))
        eps_in = self.medium.eps_model(frequency=freq_max)
        ref_ind = np.sqrt(np.max(np.real(eps_in)))
        ref_ind = max([1.0, abs(ref_ind)])
        wvl_free_space = C_0 / freq_max
        wvl_mat = wvl_free_space / ref_ind
        return dict(wvl_mat=wvl_mat, eps_in=eps_in)

    def geometry_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        eps_out: complex,
        num_proc: int = 1,
    ) -> JaxGeometryType:
        """Compute the VJP for the structure geometry."""

        medium_params = self._get_medium_params(grad_data_eps=grad_data_eps)

        return self.geometry.store_vjp(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            grad_data_eps=grad_data_eps,
            sim_bounds=sim_bounds,
            wvl_mat=medium_params["wvl_mat"],
            eps_out=eps_out,
            eps_in=medium_params["eps_in"],
            num_proc=num_proc,
        )

    def medium_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
    ) -> JaxMediumType:
        """Compute the VJP for the structure medium."""

        medium_params = self._get_medium_params(grad_data_eps=grad_data_eps)

        return self.medium.store_vjp(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            sim_bounds=sim_bounds,
            wvl_mat=medium_params["wvl_mat"],
            inside_fn=self.geometry.inside,
        )

    def store_vjp(
        self,
        # field_keys: List[Literal["medium", "geometry"]],
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        eps_out: complex,
        num_proc: int = 1,
    ) -> JaxStructure:
        """Returns the gradient of the structure parameters given forward and adjoint field data."""

        # return right away if field_keys are not present for some reason
        if not self._differentiable_fields:
            return self

        vjp_dict = {}

        # compute minimum wavelength in material (to use for determining integration points)
        if "geometry" in self._differentiable_fields:
            vjp_dict["geometry"] = self.geometry_vjp(
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                grad_data_eps=grad_data_eps,
                sim_bounds=sim_bounds,
                eps_out=eps_out,
                num_proc=num_proc,
            )

        if "medium" in self._differentiable_fields:
            vjp_dict["medium"] = self.medium_vjp(
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                grad_data_eps=grad_data_eps,
                sim_bounds=sim_bounds,
            )

        return self.updated_copy(**vjp_dict)


@register_pytree_node_class
class JaxStructure(AbstractJaxStructure, JaxObject):
    """A :class:`.Structure` registered with jax."""

    geometry: JaxGeometryType = pd.Field(
        ...,
        title="Geometry",
        description="Geometry of the structure, which is jax-compatible.",
        jax_field=True,
        discriminator=TYPE_TAG_STR,
    )

    medium: JaxMediumType = pd.Field(
        ...,
        title="Medium",
        description="Medium of the structure, which is jax-compatible.",
        jax_field=True,
        discriminator=TYPE_TAG_STR,
    )

    _differentiable_fields = ("medium", "geometry")


@register_pytree_node_class
class JaxStructureStaticMedium(AbstractJaxStructure, JaxObject):
    """A :class:`.Structure` registered with jax."""

    geometry: JaxGeometryType = pd.Field(
        ...,
        title="Geometry",
        description="Geometry of the structure, which is jax-compatible.",
        jax_field=True,
        discriminator=TYPE_TAG_STR,
    )

    medium: MediumType = pd.Field(
        ...,
        title="Medium",
        description="Regular ``tidy3d`` medium of the structure, non differentiable. "
        "Supports dispersive materials.",
        jax_field=False,
        discriminator=TYPE_TAG_STR,
    )

    _differentiable_fields = ("geometry",)


@register_pytree_node_class
class JaxStructureStaticGeometry(AbstractJaxStructure, JaxObject):
    """A :class:`.Structure` registered with jax."""

    geometry: GeometryType = pd.Field(
        ...,
        title="Geometry",
        description="Regular ``tidy3d`` geometry of the structure, non differentiable. "
        "Supports angled sidewalls and other complex geometries.",
        jax_field=False,
        discriminator=TYPE_TAG_STR,
    )

    medium: JaxMediumType = pd.Field(
        ...,
        title="Medium",
        description="Medium of the structure, which is jax-compatible.",
        jax_field=True,
        discriminator=TYPE_TAG_STR,
    )

    _differentiable_fields = ("medium",)


JaxStructureType = Union[JaxStructure, JaxStructureStaticMedium, JaxStructureStaticGeometry]
