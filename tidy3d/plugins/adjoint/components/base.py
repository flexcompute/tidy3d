"""Base model for Tidy3D components that are compatible with jax."""

from __future__ import annotations

import json
from typing import Any, Callable, List, Tuple

import jax
import numpy as np
import pydantic.v1 as pd
from jax.tree_util import tree_flatten as jax_tree_flatten
from jax.tree_util import tree_unflatten as jax_tree_unflatten

from ....components.base import Tidy3dBaseModel
from .data.data_array import JAX_DATA_ARRAY_TAG, JaxDataArray

# end of the error message when a ``_validate_web_adjoint`` exception is raised
WEB_ADJOINT_MESSAGE = (
    "You can still run this simulation through "
    "'tidy3d.plugins.adjoint.web.run_local' or 'tidy3d.plugins.adjoint.web.run_local' "
    ", which are similar to 'run' / 'run_async', but "
    "perform the gradient postprocessing calculation locally after the simulation runs. "
    "Note that the postprocessing time can become "
    "quite long (several minutes or more) if these restrictions are exceeded. "
    "Furthermore, the local versions of 'adjoint' require downloading field data "
    "inside of the 'input_structures', which can greatly increase the size of data "
    "needing to be downloaded."
)


class JaxObject(Tidy3dBaseModel):
    """Abstract class that makes a :class:`.Tidy3dBaseModel` jax-compatible through inheritance."""

    _tidy3d_class = Tidy3dBaseModel

    """Shortcut to get names of fields with certain properties."""

    @classmethod
    def _get_field_names(cls, field_key: str) -> List[str]:
        """Get all fields where ``field_key`` defined in the ``pydantic.Field``."""
        fields = []
        for field_name, model_field in cls.__fields__.items():
            field_value = model_field.field_info.extra.get(field_key)
            if field_value:
                fields.append(field_name)
        return fields

    @classmethod
    def get_jax_field_names(cls) -> List[str]:
        """Returns list of field names where ``jax_field=True``."""
        return cls._get_field_names("jax_field")

    @classmethod
    def get_jax_leaf_names(cls) -> List[str]:
        """Returns list of field names where ``stores_jax_for`` defined."""
        return cls._get_field_names("stores_jax_for")

    @classmethod
    def get_jax_field_names_all(cls) -> List[str]:
        """Returns list of field names where ``jax_field=True`` or ``stores_jax_for`` defined."""
        jax_field_names = cls.get_jax_field_names()
        jax_leaf_names = cls.get_jax_leaf_names()
        return list(set(jax_field_names + jax_leaf_names))

    @property
    def jax_fields(self) -> dict:
        """Get dictionary of ``jax`` fields."""

        # TODO: don't use getattr, define this dictionary better
        jax_field_names = self.get_jax_field_names()
        return {key: getattr(self, key) for key in jax_field_names}

    def _validate_web_adjoint(self) -> None:
        """Run validators for this component, only if using ``tda.web.run()``."""
        pass

    """Methods needed for jax to register arbitrary classes."""

    def tree_flatten(self) -> Tuple[list, dict]:
        """How to flatten a :class:`.JaxObject` instance into a ``pytree``."""
        children = []
        aux_data = self.dict()

        for field_name in self.get_jax_field_names_all():
            field = getattr(self, field_name)
            sub_children, sub_aux_data = jax_tree_flatten(field)
            children.append(sub_children)
            aux_data[field_name] = sub_aux_data

        def fix_numpy(value: Any) -> Any:
            """Recursively convert any ``numpy`` array in the value to nested list."""
            if isinstance(value, (tuple, list)):
                return [fix_numpy(val) for val in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: fix_numpy(val) for key, val in value.items()}
            else:
                return value

        aux_data = fix_numpy(aux_data)

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> JaxObject:
        """How to unflatten a ``pytree`` into a :class:`.JaxObject` instance."""
        self_dict = aux_data.copy()
        for field_name, sub_children in zip(cls.get_jax_field_names_all(), children):
            sub_aux_data = aux_data[field_name]
            field = jax_tree_unflatten(sub_aux_data, sub_children)
            self_dict[field_name] = field

        return cls.parse_obj(self_dict)

    """Type conversion helpers."""

    def to_tidy3d(self: JaxObject) -> Tidy3dBaseModel:
        """Convert :class:`.JaxObject` instance to :class:`.Tidy3dBaseModel` instance."""

        self_dict = self.dict(exclude=self.exclude_fields_leafs_only)

        for key in self.get_jax_field_names():
            sub_field = self.jax_fields[key]

            # TODO: simplify this logic
            if isinstance(sub_field, (tuple, list)):
                self_dict[key] = [x.to_tidy3d() for x in sub_field]
            else:
                self_dict[key] = sub_field.to_tidy3d()
            # end TODO

        return self._tidy3d_class.parse_obj(self_dict)

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: Tidy3dBaseModel) -> JaxObject:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type"})

        for key in cls.get_jax_field_names():
            sub_field_type = cls.__fields__[key].type_
            tidy3d_sub_field = getattr(tidy3d_obj, key)

            # TODO: simplify this logic
            if isinstance(tidy3d_sub_field, (tuple, list)):
                obj_dict[key] = [sub_field_type.from_tidy3d(x) for x in tidy3d_sub_field]
            else:
                obj_dict[key] = sub_field_type.from_tidy3d(tidy3d_sub_field)
            # end TODO

        return cls.parse_obj(obj_dict)

    @property
    def exclude_fields_leafs_only(self) -> set:
        """Fields to exclude from ``self.dict()``, ``"type"`` and all ``jax`` leafs."""
        return set(["type"] + self.get_jax_leaf_names())

    """Accounting with jax and regular fields."""

    @pd.root_validator(pre=True)
    def handle_jax_kwargs(cls, values: dict) -> dict:
        """Pass jax inputs to the jax fields and pass untraced values to the regular fields."""

        # for all jax-traced fields
        for jax_name in cls.get_jax_leaf_names():
            # if a value was passed to the object for the regular field
            orig_name = cls.__fields__[jax_name].field_info.extra.get("stores_jax_for")
            val = values.get(orig_name)
            if val is not None:
                # try adding the sanitized (no trace) version to the regular field
                try:
                    values[orig_name] = jax.lax.stop_gradient(val)

                # if it doesnt work, just pass the raw value (necessary to handle inf strings)
                except TypeError:
                    values[orig_name] = val

                # if the jax name was not specified directly, use the original traced value
                if jax_name not in values:
                    values[jax_name] = val

        return values

    @pd.root_validator(pre=True)
    def handle_array_jax_leafs(cls, values: dict) -> dict:
        """Convert jax_leafs that are passed as numpy arrays."""
        for jax_name in cls.get_jax_leaf_names():
            val = values.get(jax_name)
            if isinstance(val, np.ndarray):
                values[jax_name] = val.tolist()
        return values

    """ IO """

    # TODO: replace with JaxObject json encoder

    def _json(self, *args, **kwargs) -> str:
        """Overwritten method to get the json string to store in the files."""

        json_string_og = super()._json(*args, **kwargs)
        json_dict = json.loads(json_string_og)

        def strip_data_array(val: Any) -> Any:
            """Recursively strip any elements with type "JaxDataArray", replace with tag."""

            if isinstance(val, dict):
                if "type" in val and val["type"] == "JaxDataArray":
                    return JAX_DATA_ARRAY_TAG
                return {k: strip_data_array(v) for k, v in val.items()}

            elif isinstance(val, (tuple, list)):
                return [strip_data_array(v) for v in val]

            return val

        json_dict = strip_data_array(json_dict)
        return json.dumps(json_dict)

    # TODO: replace with implementing these in DataArray

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
