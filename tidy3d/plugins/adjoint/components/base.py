"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import List, Any, Callable
import json

import pydantic.v1 as pd

import jax
from jax.tree_util import register_pytree_node_class

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

    def tree_flatten(self) -> tuple[list, dict]:
        """Split ``JaxObject`` into jax-traced children and auxiliary data."""
        aux_data = self.dict(exclude={"jax_info"})
        # values = tuple(self.jax_info.values())
        # keys = tuple(self.jax_info.keys())
        # aux_data["jax_keys"] = keys
        return self.jax_info, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
        """Create the ``JaxObject`` from the auxiliary data and children."""
        # keys = aux_data.pop("jax_keys")
        # jax_info = dict(zip(keys, children))
        return cls(**aux_data, jax_info=children)

    @pd.root_validator(pre=True)
    def _handle_jax_kwargs(cls, values):

        jax_info = {}
        _kwargs = {}

        def handle_jax_field(val):
            
            if isinstance(val, JaxObj):
                return val, val.jax_info
            elif isinstance(val, (list, tuple)):
                vals = []
                jax_infos = []
                for item in val:
                    subval, sub_jax_info = handle_jax_field(item)
                    vals.append(subval)
                    jax_infos.append(sub_jax_info)
                return vals, jax_infos
            else:
                return jax.lax.stop_gradient(val), val
    
        for key, val in values.items():
            if key in cls._jax_fields:
                val1, val2 = handle_jax_field(val)
                _kwargs[key] = val1
                jax_info[key] = val2
            else:
                _kwargs[key] = val
        _kwargs["jax_info"] = jax_info
        return _kwargs
        
    # @pd.root_validator(pre=True)
    # def _handle_jax_kwargs(cls, values):
    #     """Pre-process the init kwargs and sort jax-traced types into ``jax_info``."""

    #     jax_info = {}
    #     _kwargs = {}

    #     for key, val in values.items():

    #         # value can be traced by jax
    #         if key in cls._jax_fields2:

    #             # pass the untracked version to the regular tidy3d fields
    #             _val = jax.lax.stop_gradient(val)
    #             _kwargs[key] = _val

    #             # store the tracked value in the jax_info field
    #             jax_info[key] = val

    #         # value can't be traced by jax
    #         else:

    #             # handle like a regular kwarg
    #             _kwargs[key] = val

    #     # include the jax_info in the set of kwargs
    #     _kwargs["jax_info"] = jax_info
    #     return _kwargs

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
        # for key in cls._jax_fields:
        #     obj_type = type(tidy3d_obj)
        #     sub_type = cls._type_mappings[obj_type]
        #     obj_dict[key] = sub_type.from_tidy3d(obj_dict[key])
        return cls.parse_obj(obj_dict)

    # ALTERNATIVE TO _jax_fields
    # @classmethod
    # def get_jax_field_names(cls) -> List[str]:
    #     """Returns list of field names that have a ``jax_field_type``."""
    #     adjoint_fields = []
    #     for field_name, model_field in cls.__fields__.items():
    #         jax_field_type = model_field.field_info.extra.get("jax_field")
    #         if jax_field_type:
    #             adjoint_fields.append(field_name)
    #     return adjoint_fields

    #     """Methods needed for jax to register arbitary classes."""

    #     def tree_flatten(self) -> Tuple[list, dict]:
    #         """How to flatten a :class:`.JaxObject` instance into a pytree."""
    #         children = []
    #         aux_data = self.dict()
    #         for field_name in self.get_jax_field_names():
    #             field = getattr(self, field_name)
    #             sub_children, sub_aux_data = jax_tree_flatten(field)
    #             children.append(sub_children)
    #             aux_data[field_name] = sub_aux_data

    #         def fix_polyslab(geo_dict: dict) -> None:
    #             """Recursively Fix a dictionary possibly containing a polyslab geometry."""
    #             if geo_dict["type"] == "PolySlab":
    #                 vertices = geo_dict["vertices"]
    #                 geo_dict["vertices"] = vertices.tolist()
    #             elif geo_dict["type"] == "GeometryGroup":
    #                 for sub_geo_dict in geo_dict["geometries"]:
    #                     fix_polyslab(sub_geo_dict)
    #             elif geo_dict["type"] == "ClipOperation":
    #                 fix_polyslab(geo_dict["geometry_a"])
    #                 fix_polyslab(geo_dict["geometry_b"])

    #         def fix_monitor(mnt_dict: dict) -> None:
    #             """Fix a frequency containing monitor."""
    #             if "freqs" in mnt_dict:
    #                 freqs = mnt_dict["freqs"]
    #                 if isinstance(freqs, np.ndarray):
    #                     mnt_dict["freqs"] = freqs.tolist()

    #         # fixes bug with jax handling 2D numpy array in polyslab vertices
    #         if aux_data.get("type", "") == "JaxSimulation":
    #             structures = aux_data["structures"]
    #             for _i, structure in enumerate(structures):
    #                 geometry = structure["geometry"]
    #                 fix_polyslab(geometry)
    #             for monitor in aux_data["monitors"]:
    #                 fix_monitor(monitor)
    #             for monitor in aux_data["output_monitors"]:
    #                 fix_monitor(monitor)

    #         return children, aux_data

    #     @classmethod
    #     def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
    #         """How to unflatten a pytree into a :class:`.JaxObject` instance."""
    #         self_dict = aux_data.copy()
    #         for field_name, sub_children in zip(cls.get_jax_field_names(), children):
    #             sub_aux_data = aux_data[field_name]
    #             field = jax_tree_unflatten(sub_aux_data, sub_children)
    #             self_dict[field_name] = field

    #         return cls.parse_obj(self_dict)

    #     """Type conversion helpers."""

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
