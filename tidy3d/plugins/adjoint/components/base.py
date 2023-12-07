"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import List, Any, Callable
import json

import pydantic.v1 as pd

import jax
from jax.tree_util import register_pytree_node_class

from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel
from .data.data_array import JaxDataArray, JAX_DATA_ARRAY_TAG


@register_pytree_node_class
class JaxObject(Tidy3dBaseModel):
    """Abstract class that makes a :class:`.Tidy3dBaseModel` jax-compatible through inheritance."""

    jax_info: dict  # stores traced arrays and other fields that jax needs to know
    _tidy3d_class = Tidy3dBaseModel  # corresponding class type in tidy3d.components
    _jax_fields = ()  # fields that contain JaxObject objects
    _jax_fields2 = ()
    _type_mappings = {}  # mapping from all possible tidy3d fields to the correponding jax fields

    def split(self) -> tuple:
        """split self into component without jax_info and jax_info."""
        return self.updated_copy(jax_info={}), self.jax_info

    def combine(self, jax_info):
        """Combine self with jax info to add it to self."""
        return self.updated_copy(jax_info=jax_info)

    def tree_flatten(self) -> tuple[list, dict]:
        """Split ``JaxObject`` into jax-traced children and auxiliary data."""
        leaves, treedef = jax_tree_flatten(self.jax_info)
        aux_data = dict(self_no_jax=self.updated_copy(jax_info={}), treedef=treedef)

        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
        """Create the ``JaxObject`` from the auxiliary data and children."""
        self_no_jax = aux_data["self_no_jax"]
        treedef = aux_data["treedef"]
        leaves = children
        jax_info = jax_tree_unflatten(treedef, leaves)
        return self_no_jax.updated_copy(jax_info=jax_info)

    @pd.root_validator(pre=True)
    def _handle_jax_kwargs(cls, values):
        """How to parse passed values."""

        def get_jax_info(val) -> dict:
            """grab the jax info from a value."""
            if isinstance(val, JaxObject):
                return val.jax_info
            elif isinstance(val, (list, tuple)):
                return [get_jax_info(v) for v in val]
            elif isinstance(val, dict):
                return val.get("jax_info") if "jax_info" in val else {}
            else:
                return val

        # parse out the jax info from the input kwargs
        if not values.get("jax_info"):
            jax_info = {}
            for key in cls._jax_fields2:
                val = values.get(key)
                jax_info[key] = get_jax_info(val)

            # save the jax_info in a separate kwarg
            values["jax_info"] = jax_info
        return values

    @pd.root_validator(pre=True)
    def _sanitize_jax_kwargs(cls, values):
        """How to parse passed values."""
        for key in cls._jax_fields2:
            try:
                values[key] = jax.lax.stop_gradient(values[key])
            except Exception:
                pass
        return values

    @property
    def excluded_dict(self) -> dict:
        """Self.dict() with exclude_fields excluded."""
        return self.dict(exclude=self.exclude_fields)

    @property
    def exclude_fields(self) -> set:
        """Fields to exclude from self.dict()."""
        return set(["type", "jax_info"] + list(self._jax_fields))

    @property
    def jax_fields(self) -> dict:
        """The fields that are jax-traced for this class."""
        return {key: getattr(self, key) for key in self._jax_fields}

    def to_tidy3d(self) -> Tidy3dBaseModel:
        """Convert a ``JaxObject`` to a tidy3d component (without ``jax_info``)."""
        self_dict = self.excluded_dict
        for key, val in self.jax_fields.items():
            if isinstance(val, JaxObject):
                self_dict[key] = val.to_tidy3d()
            elif isinstance(val, (list, tuple)):
                self_dict[key] = [sub_val.to_tidy3d for sub_val in val]
            else:
                raise ValueError
        return self._tidy3d_class.parse_obj(self_dict)

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: Tidy3dBaseModel) -> JaxObject:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type"})
        return cls.parse_obj(obj_dict)

    """ IO """

    def _json(self, *args, **kwargs) -> str:
        """Overwritten method to get the json string to store in the files."""

        json_string_og = super()._json(*args, **kwargs)
        json_dict = json.loads(json_string_og)

        def strip_data_array(sub_dict: dict) -> None:
            """Strip any elements of the dictionary with type "JaxDataArray", replace with tag."""

            for key, val in sub_dict.items():

                if isinstance(val, dict):
                    if "type" in val and val["type"] == "JaxDataArray":
                        sub_dict[key] = JAX_DATA_ARRAY_TAG
                    else:
                        strip_data_array(val)
                elif isinstance(val, (list, tuple)):
                    val_dict = dict(zip(range(len(val)), val))
                    strip_data_array(val_dict)
                    sub_dict[key] = list(val_dict.values())

        strip_data_array(json_dict)
        return json.dumps(json_dict)

    def to_hdf5(self, fname: str, custom_encoders: List[Callable] = None) -> None:
        """Exports :class:`JaxObject` instance to .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to save the :class:`JaxObject` to.
        custom_encoders : List[Callable]
            List of functions accepting (fname: str, group_path: str, value: Any) that take
            the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

        Example
        -------
        >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        def data_array_encoder(fname: str, group_path: str, value: Any) -> None:
            """Custom encoder to convert the JaxDataArray dict to an instance."""
            if isinstance(value, dict) and "type" in value and value["type"] == "JaxDataArray":
                data_array = JaxDataArray(values=value["values"], coords=value["coords"])
                data_array.to_hdf5(fname=fname, group_path=group_path)

        if custom_encoders is None:
            custom_encoders = []

        custom_encoders += [data_array_encoder]

        return super().to_hdf5(fname=fname, custom_encoders=custom_encoders)

    @classmethod
    def dict_from_hdf5(
        cls, fname: str, group_path: str = "", custom_decoders: List[Callable] = None
    ) -> dict:
        """Loads a dictionary containing the model contents from a .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to load the :class:`JaxObject` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.
        custom_decoders : List[Callable]
            List of functions accepting
            (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
            value in the model dict after a custom decoding.

        Returns
        -------
        dict
            Dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        def data_array_decoder(
            fname: str, group_path: str, model_dict: dict, key: str, value: Any
        ) -> None:
            """Custom decoder to grab JaxDataArray from file and save it in model_dict."""

            # write the path to the element of the json dict where the data_array should be
            if isinstance(value, str) and value == JAX_DATA_ARRAY_TAG:
                jax_data_array = JaxDataArray.from_hdf5(fname=fname, group_path=group_path)
                model_dict[key] = jax_data_array

        if custom_decoders is None:
            custom_decoders = []

        custom_decoders += [data_array_decoder]

        return super().dict_from_hdf5(
            fname=fname, group_path=group_path, custom_decoders=custom_decoders
        )
