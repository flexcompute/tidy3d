"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import Tuple, List

from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel


class JaxObject(Tidy3dBaseModel):
    """Abstract class that makes a :class:`.Tidy3dBaseModel` jax-compatible through inheritance."""

    """Shortcut to get names of all fields that have jax components."""

    @classmethod
    def get_jax_field_names(cls) -> List[str]:
        """Returns list of field names that have a ``jax_field_type``."""
        adjoint_fields = []
        for field_name, model_field in cls.__fields__.items():
            jax_field_type = model_field.field_info.extra.get("jax_field")
            if jax_field_type:
                adjoint_fields.append(field_name)
        return adjoint_fields

    """Methods needed for jax to register arbitary classes."""

    def tree_flatten(self) -> Tuple[list, dict]:
        """How to flatten a :class:`.JaxObject` instance into a pytree."""
        children = []
        aux_data = self.dict()
        for field_name in self.get_jax_field_names():
            field = getattr(self, field_name)
            sub_children, sub_aux_data = jax_tree_flatten(field)
            children.append(sub_children)
            aux_data[field_name] = sub_aux_data
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
        """How to unflatten a pytree into a :class:`.JaxObject` instance."""
        self_dict = aux_data.copy()
        for field_name, sub_children in zip(cls.get_jax_field_names(), children):
            sub_aux_data = aux_data[field_name]
            field = jax_tree_unflatten(sub_aux_data, sub_children)
            self_dict[field_name] = field

        return cls.parse_obj(self_dict)

    """Type conversion helpers."""

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: Tidy3dBaseModel) -> JaxObject:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type"})
        return cls.parse_obj(obj_dict)
