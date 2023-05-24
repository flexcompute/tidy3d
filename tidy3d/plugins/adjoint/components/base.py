"""Base model for Tidy3D components that are compatible with jax."""
from __future__ import annotations

from typing import Tuple, List
import json

import h5py
import xarray as xr

from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel, JSON_TAG, DATA_ARRAY_MAP, cached_property
from .data.data_array import JaxDataArray, JAX_DATA_ARRAY_TAG


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

    """ IO """

    @cached_property
    def _json_string(self) -> str:
        """Overwritten method to get the json string to store in the files."""

        json_string_og = super()._json_string
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

    def to_hdf5(self, fname: str) -> None:
        """Exports :class:`JaxObject` instance to .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to save the :class:`JaxObject` to.

        Example
        -------
        >>> simulation.to_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        with h5py.File(fname, "w") as f_handle:

            f_handle[JSON_TAG] = self._json_string

            def add_data_to_file(data_dict: dict, group_path: str = "") -> None:
                """For every DataArray item in dictionary, write path of hdf5 group as value."""

                for key, value in data_dict.items():

                    # append the key to the path
                    subpath = f"{group_path}/{key}"

                    if (
                        isinstance(value, dict)
                        and "type" in value
                        and value["type"] == "JaxDataArray"
                    ):
                        value = JaxDataArray(values=value["values"], coords=value["coords"])

                    # write the path to the element of the json dict where the data_array should be
                    if isinstance(value, (xr.DataArray, JaxDataArray)):
                        value.to_hdf5(fname=f_handle, group_path=subpath)

                    # if a tuple, assign each element a unique key
                    if isinstance(value, (list, tuple)):
                        value_dict = self.tuple_to_dict(tuple_values=value)
                        add_data_to_file(data_dict=value_dict, group_path=subpath)

                    # if a dict, recurse
                    elif isinstance(value, dict):
                        add_data_to_file(data_dict=value, group_path=subpath)

            add_data_to_file(data_dict=self.dict())

    @classmethod
    def dict_from_hdf5(cls, fname: str, group_path: str = "") -> dict:
        """Loads a dictionary containing the model contents from a .hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to the .hdf5 file to load the :class:`JaxObject` from.
        group_path : str, optional
            Path to a group inside the file to selectively load a sub-element of the model only.

        Returns
        -------
        dict
            Dictionary containing the model.

        Example
        -------
        >>> sim_dict = Simulation.dict_from_hdf5(fname='folder/sim.hdf5') # doctest: +SKIP
        """

        def load_data_from_file(model_dict: dict, group_path: str = "") -> None:
            """For every DataArray item in dictionary, load path of hdf5 group as value."""

            for key, value in model_dict.items():

                subpath = f"{group_path}/{key}"

                # write the path to the element of the json dict where the data_array should be
                if isinstance(value, str) and value == JAX_DATA_ARRAY_TAG:

                    jax_data_array = JaxDataArray.from_hdf5(fname=fname, group_path=subpath)
                    model_dict[key] = jax_data_array
                    continue

                if isinstance(value, str) and value in DATA_ARRAY_MAP:
                    data_array_type = DATA_ARRAY_MAP[value]
                    model_dict[key] = data_array_type.from_hdf5(fname=fname, group_path=subpath)
                    continue

                # if a list, assign each element a unique key, recurse
                if isinstance(value, (list, tuple)):
                    value_dict = cls.tuple_to_dict(tuple_values=value)
                    load_data_from_file(model_dict=value_dict, group_path=subpath)

                # if a dict, recurse
                elif isinstance(value, dict):
                    load_data_from_file(model_dict=value, group_path=subpath)

        with h5py.File(fname, "r") as f_handle:
            json_string = f_handle[JSON_TAG][()]
            model_dict = json.loads(json_string)

        group_path = cls._construct_group_path(group_path)
        model_dict = cls.get_sub_model(group_path=group_path, model_dict=model_dict)
        load_data_from_file(model_dict=model_dict, group_path=group_path)
        return model_dict
