"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import Tuple, List, Any, Callable
import json

import numpy as np
import pydantic.v1 as pd
import jax

from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel
# from .data.data_array import JaxDataArray, JAX_DATA_ARRAY_TAG


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
    
    jax_dict : dict = {}
    _jax_leafs = ()
    _jax_objs = ()
    _jax_obj_lists = ()
    
    def jax_dict_nested(self) -> dict:
        """Return nested dict of all jax fields contained in this object."""

        # note, all jax_leafs contained in here already
        jax_dict = self.jax_dict.copy()
        
        # iterate over jax_objs
        for fld_name in self._jax_objs:
            jax_obj = getattr(self, fld_name)
            jax_dict[fld_name] = jax_obj.jax_dict_nested()
        
        # iterate over jax_obj_lists
        for fld_name in self._jax_obj_lists:
            jax_dict[fld_name] = []
            jax_obj_list = getattr(self, fld_name)
            for jax_obj in jax_obj_list:
                jax_dict[fld_name].append(jax_obj.jax_dict_nested())
                
        return jax_dict
    
    def tree_flatten(self) -> tuple[list, dict]:
        """Split ``JaxObject`` into jax-traced children and auxiliary data."""
        children, treedef = jax_tree_flatten(self.jax_dict_nested())
        aux_data = dict(treedef=treedef, self_dict=self.dict())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
        """Create the ``JaxObject`` from the auxiliary data and children."""      
        treedef = aux_data["treedef"]
        self_dict = aux_data["self_dict"]
        jax_dict = jax_tree_unflatten(treedef, children)
        self_dict["jax_dict"] = jax_dict
        return cls.parse_obj(self_dict)
    
    @pd.root_validator(pre=True)
    def _store_jax_values(cls, values):
        
        def get_jax_dict(val):
            if isinstance(val, dict):
                return val.get("jax_dict") or {}
            elif isinstance(val, JaxObject):
                return val.jax_dict
            else:
                return val

        jax_dict = get_jax_dict(values)

        # store jax_dict of leaves
        for key in cls._jax_leafs:
            val = values[key]
            jax_dict[key] = get_jax_dict(val)

        # store jax_dict of jax_objs
        for key in cls._jax_objs:
            val = values[key]
            jax_dict[key] = get_jax_dict(val)            
        
        # store jax_dict of jax_obj_lists
        for key in cls._jax_obj_lists:
            val = values[key]
            jax_dict[key] = []
            for _val in val:
                jax_dict[key] = get_jax_dict(_val)
        
        values["jax_dict"] = jax_dict

        return values
    
    @pd.root_validator(pre=True)
    def _sanitize_jax_values(cls, values):

        for key in cls._jax_leafs:
            val = values[key]
            values[key] = jax.lax.stop_gradient(val)
        return values

    """Type conversion helpers."""

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: Tidy3dBaseModel) -> JaxObject:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type"})
        return cls.parse_obj(obj_dict)

    """ IO """

    # def _json(self, *args, **kwargs) -> str:
    #     """Overwritten method to get the json string to store in the files."""

    #     json_string_og = super()._json(*args, **kwargs)
    #     json_dict = json.loads(json_string_og)

    #     def strip_data_array(sub_dict: dict) -> None:
    #         """Strip any elements of the dictionary with type "JaxDataArray", replace with tag."""

    #         for key, val in sub_dict.items():

    #             if isinstance(val, dict):
    #                 if "type" in val and val["type"] == "JaxDataArray":
    #                     sub_dict[key] = JAX_DATA_ARRAY_TAG
    #                 else:
    #                     strip_data_array(val)
    #             elif isinstance(val, (list, tuple)):
    #                 val_dict = dict(zip(range(len(val)), val))
    #                 strip_data_array(val_dict)
    #                 sub_dict[key] = list(val_dict.values())

    #     strip_data_array(json_dict)
    #     return json.dumps(json_dict)

    # def to_hdf5(self, fname: str, custom_encoders: List[Callable] = None) -> None:
    #     """Exports :class:`JaxObject` instance to .hdf5 file.

    #     Parameters
    #     ----------
    #     fname : str
    #         Full path to the .hdf5 file to save the :class:`JaxObject` to.
    #     custom_encoders : List[Callable]
    #         List of functions accepting (fname: str, group_path: str, value: Any) that take
    #         the ``value`` supplied and write it to the hdf5 ``fname`` at ``group_path``.

    #     Example
    #     -------
    #     >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
    #     """

    #     def data_array_encoder(fname: str, group_path: str, value: Any) -> None:
    #         """Custom encoder to convert the JaxDataArray dict to an instance."""
    #         if isinstance(value, dict) and "type" in value and value["type"] == "JaxDataArray":
    #             data_array = JaxDataArray(values=value["values"], coords=value["coords"])
    #             data_array.to_hdf5(fname=fname, group_path=group_path)

    #     if custom_encoders is None:
    #         custom_encoders = []

    #     custom_encoders += [data_array_encoder]

    #     return super().to_hdf5(fname=fname, custom_encoders=custom_encoders)

    # @classmethod
    # def dict_from_hdf5(
    #     cls, fname: str, group_path: str = "", custom_decoders: List[Callable] = None
    # ) -> dict:
    #     """Loads a dictionary containing the model contents from a .hdf5 file.

    #     Parameters
    #     ----------
    #     fname : str
    #         Full path to the .hdf5 file to load the :class:`JaxObject` from.
    #     group_path : str, optional
    #         Path to a group inside the file to selectively load a sub-element of the model only.
    #     custom_decoders : List[Callable]
    #         List of functions accepting
    #         (fname: str, group_path: str, model_dict: dict, key: str, value: Any) that store the
    #         value in the model dict after a custom decoding.

    #     Returns
    #     -------
    #     dict
    #         Dictionary containing the model.

    #     Example
    #     -------
    #     >>> sim_dict = Simulation.dict_from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
    #     """

    #     def data_array_decoder(
    #         fname: str, group_path: str, model_dict: dict, key: str, value: Any
    #     ) -> None:
    #         """Custom decoder to grab JaxDataArray from file and save it in model_dict."""

    #         # write the path to the element of the json dict where the data_array should be
    #         if isinstance(value, str) and value == JAX_DATA_ARRAY_TAG:
    #             jax_data_array = JaxDataArray.from_hdf5(fname=fname, group_path=group_path)
    #             model_dict[key] = jax_data_array

    #     if custom_decoders is None:
    #         custom_decoders = []

    #     custom_decoders += [data_array_decoder]

    #     return super().dict_from_hdf5(
    #         fname=fname, group_path=group_path, custom_decoders=custom_decoders
        # )
