"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import Tuple, List, Any, Callable
import json

import numpy as np
import jax
import pydantic.v1 as pd

from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel
from .data.data_array import JaxDataArray, JAX_DATA_ARRAY_TAG


class JaxObject(Tidy3dBaseModel):
    """Abstract class that makes a :class:`.Tidy3dBaseModel` jax-compatible through inheritance."""

    """Shortcut to get names of fields with certain properties."""

    @classmethod
    def _get_fields(cls, field_key: str) -> List[str]:
        """Get all fields where ``field_key=True`` in the ``pydantic.Field``."""
        fields = []
        for field_name, model_field in cls.__fields__.items():
            field_value = model_field.field_info.extra.get(field_key)
            if field_value:
                fields.append(field_name)
        return fields

    @classmethod
    def get_jax_field_names(cls) -> List[str]:
        """Returns list of field names where ``jax_field=True``."""
        return cls._get_fields("jax_field")

    @classmethod
    def get_jax_leaf_names(cls) -> List[str]:
        """Returns list of field names where ``jax_leaf=True``."""
        return cls._get_fields("jax_leaf")

    """Methods needed for jax to register arbitary classes."""

    def tree_flatten(self) -> Tuple[list, dict]:
        """How to flatten a :class:`.JaxObject` instance into a pytree."""
        children = []
        aux_data = self.dict()

        for field_name in self.get_jax_field_names():
            # if field_name == "vertices_jax":
            #     import pdb; pdb.set_trace()
            field = getattr(self, field_name)
            sub_children, sub_aux_data = jax_tree_flatten(field)
            children.append(sub_children)
            aux_data[field_name] = sub_aux_data

        def fix_numpy(value: Any) -> Any:
            """Recursively convert any numpy array in the value to nested list."""
            if isinstance(value, (tuple, list)):
                return [fix_numpy(val) for val in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: fix_numpy(val) for key, val in value.items()}
            else:
                return value

        aux_data = fix_numpy(aux_data)

        # def fix_polyslab(geo_dict: dict) -> None:
        #     """Recursively Fix a dictionary possibly containing a polyslab geometry."""
        #     if geo_dict["type"] == "PolySlab":
        #         vertices = geo_dict["vertices"]
        #         geo_dict["vertices"] = vertices.tolist()
        #         # vertices_jax = geo_dict["vertices_jax"]
        #         # geo_dict["vertices_jax"] = vertices_jax.tolist()
        #     elif geo_dict["type"] == "GeometryGroup":
        #         for sub_geo_dict in geo_dict["geometries"]:
        #             fix_polyslab(sub_geo_dict)
        #     elif geo_dict["type"] == "ClipOperation":
        #         fix_polyslab(geo_dict["geometry_a"])
        #         fix_polyslab(geo_dict["geometry_b"])

        # def fix_monitor(mnt_dict: dict) -> None:
        #     """Fix a frequency containing monitor."""
        #     if "freqs" in mnt_dict:
        #         freqs = mnt_dict["freqs"]
        #         if isinstance(freqs, np.ndarray):
        #             mnt_dict["freqs"] = freqs.tolist()

        # for key, val in aux_data.items():
        #     if isinstance(val, np.ndarray):
        #         aux_data[key] = val.tolist()

        # # fixes bug with jax handling 2D numpy array in polyslab vertices
        # if aux_data.get("type", "") == "JaxSimulation":
        #     structures = aux_data["structures"]
        #     for _i, structure in enumerate(structures):
        #         geometry = structure["geometry"]
        #         fix_polyslab(geometry)
        #     for monitor in aux_data["monitors"]:
        #         fix_monitor(monitor)
        #     for monitor in aux_data["output_monitors"]:
        #         fix_monitor(monitor)

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

    """Accounting with jax and regular fields."""

    @pd.root_validator(pre=True)
    def handle_jax_kwargs(cls, values: dict) -> dict:
        """Pass jax inputs to the jax fields and pass untraced values to the regular fields."""

        for jax_name in cls.get_jax_leaf_names():
            # if a value was passed to the object for the regular field
            orig_name = cls.get_orig_field(jax_name)
            val = values.get(orig_name)

            if val is not None:
                # add the sanitized (no trace) version to the regular field
                values[orig_name] = jax.lax.stop_gradient(val)

                # if the jax name was not specified directly, use the original traced value
                if jax_name not in values:
                    values[jax_name] = val

        return values

    @pd.root_validator(pre=True)
    def handle_array_jax_leafs(cls, values: dict) -> dict:
        """Handle jax_leafs that are numpy arrays."""
        for jax_name in cls.get_jax_leaf_names():
            val = values.get(jax_name)
            if isinstance(val, np.ndarray):
                values[jax_name] = val.tolist()
        return values

    @staticmethod
    def get_jax_field(orig_field: str) -> str:
        """Get the 'jax' field name from the original field name."""
        return orig_field + "_jax"

    @staticmethod
    def get_orig_field(jax_field: str) -> str:
        """Get the 'jax' field name from the original field name."""
        split = jax_field.split("_")
        if len(split) != 2:
            raise ValueError(f"Can't get original field from jax field {jax_field}.")
        return split[0]

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
