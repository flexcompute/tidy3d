"""Utilities for converting between tidy3d versions."""
from __future__ import annotations

from typing import Dict, Callable
import json
import functools

import yaml
import pydantic as pd

from .log import log
from .version import __version__
from .exceptions import FileError, SetupError
from .components.base import Tidy3dBaseModel


"""Storing version numbers."""


class Version(pd.BaseModel):
    """Stores a version number (excluding patch)."""

    major: int
    minor: int

    @classmethod
    def from_string(cls, string: str = None) -> Version:
        """Return Version from a version string."""
        if string is None:
            return cls.from_string(string=__version__)

        try:
            version_numbers = string.split(".")
            version = cls(major=version_numbers[0], minor=version_numbers[1])
        except Exception as e:
            raise SetupError(f"version string {string} can't be parsed.") from e
        return version

    @property
    def as_tuple(self):
        """version as a tuple, leave out patch for now."""
        return (self.major, self.minor)

    def __hash__(self):
        """define a hash."""
        return hash(self.as_tuple)

    def __str__(self):
        """Convert back to string."""
        return f"{self.major}.{self.minor}"

    def __eq__(self, other):
        """versions equal."""
        return (self.major == other.major) and (self.minor == other.minor)

    def __lt__(self, other):
        """self < other."""
        if self.major < other.major:
            return True
        if self.major == other.major:
            return self.minor < other.minor
        return False

    def __gt__(self, other):
        """self > other."""
        if self.major > other.major:
            return True
        if self.major == other.major:
            return self.minor > other.minor
        return False

    def __le__(self, other):
        """self <= other."""
        return (self < other) or (self == other)

    def __ge__(self, other):
        """self >= other."""
        return (self > other) or (self == other)


CurrentVersion = Version.from_string(__version__)

"""Class for updating simulation objects."""


class Updater(pd.BaseModel):
    """Converts a tidy3d simulation.json file to an up-to-date Simulation instance."""

    sim_dict: dict

    @classmethod
    def from_file(cls, fname: str) -> Updater:
        """Dictionary representing the simulation loaded from file."""

        # TODO: fix this, it broke
        if ".hdf5" in fname:
            sim_dict = Tidy3dBaseModel.from_file(fname=fname).dict()

        else:
            # try:
            with open(fname, "r", encoding="utf-8") as f:
                if ".json" in fname:
                    sim_dict = json.load(f)
                elif ".yaml" in fname:
                    sim_dict = yaml.safe_load(f)
                else:
                    raise FileError('file extension must be ".json", ".yaml", or ".hdf5"')

            # except Exception as e:
            #     raise FileError(f"Could not load file {fname}") from e

        return cls(sim_dict=sim_dict)

    @classmethod
    def from_string(cls, sim_dict_str: str) -> Updater:
        """Dictionary representing the simulation loaded from string."""
        sim_dict = json.loads(sim_dict_str)
        return cls(sim_dict=sim_dict)

    @property
    def version(self) -> Version:
        """Version of the supplied file."""
        version_string = self.sim_dict.get("version")
        if version_string is None:
            raise SetupError("Could not find a version in the supplied json.")
        return Version.from_string(version_string)

    def get_update_function(self):
        """Get the highest update verion <= self.version."""
        leq_versions = [v for v in UPDATE_MAP if v <= self.version]
        if not leq_versions:
            raise SetupError(f"An update version <= {self.version} not found in update map.")
        update_version = max(leq_versions)
        return UPDATE_MAP[update_version]

    def get_next_version(self) -> Version:
        """Get the next version after self.version."""
        gt_versions = [v for v in UPDATE_MAP if v > self.version]
        if not gt_versions:
            return CurrentVersion
        return str(min(gt_versions))

    def update_to_current(self) -> dict:
        """Update supplied simulation dictionary to current version."""
        if self.version == CurrentVersion:
            self.sim_dict["version"] = __version__
            return self.sim_dict
        log.warning(f"updating Simulation from {self.version} to {CurrentVersion}")
        while self.version != CurrentVersion:
            update_fn = self.get_update_function()
            self.sim_dict = update_fn(self.sim_dict)
            self.sim_dict["version"] = str(self.get_next_version())
        self.sim_dict["version"] = __version__
        return self.sim_dict

    def __eq__(self, other: Updater) -> bool:
        """Is Updater equal to another one?"""
        return self.sim_dict == other.sim_dict


"""Update conversion functions."""

# versions will be dynamically mapped in this table when the update functions are initialized.
UPDATE_MAP = {}


def updates_from_version(version_from_string: str):
    """Decorates a sim_dict update function to change the version."""

    # make sure the version strings are legit
    from_version = Version.from_string(version_from_string)

    def decorator(update_fn):
        """The actual decorator that gets returned by `updates_to_version('x.y.z')`"""

        @functools.wraps(update_fn)
        def new_update_function(sim_dict: dict) -> dict:
            """Update function that automatically adds version string."""

            return update_fn(sim_dict)

        UPDATE_MAP[from_version] = new_update_function

        return new_update_function

    return decorator


def iterate_update_dict(update_dict: Dict, update_types: Dict[str, Callable]):
    """Recursively iterate nested ``update_dict``. For any nested ``nested_dict`` found,
    apply an update function if its ``nested_dict["type"]`` is in the keys of the ``update_types``
    dictionary. Also iterates lists and tuples.
    """

    if isinstance(update_dict, dict):
        # Update if we need to, and iterate recursively all items
        if update_dict.get("type") in update_types.keys():
            update_types[update_dict["type"]](update_dict)
        for item in update_dict.values():
            iterate_update_dict(item, update_types)
    elif isinstance(update_dict, (list, tuple)):
        # Try other iterables
        for item in update_dict:
            iterate_update_dict(item, update_types)


@updates_from_version("1.8")
def update_1_8(sim_dict: dict) -> dict:
    """Updates version 1.8."""

    def fix_missing_scalar_field(mnt_dict: dict) -> dict:
        for key, val in mnt_dict["field_dataset"].items():
            if val == "XR.DATAARRAY":
                mnt_dict["field_dataset"][key] = "ScalarFieldDataArray"
        return mnt_dict

    iterate_update_dict(
        update_dict=sim_dict,
        update_types={
            "CustomFieldSource": fix_missing_scalar_field,
        },
    )
    return sim_dict


@updates_from_version("1.7")
def update_1_7(sim_dict: dict) -> dict:
    """Updates version 1.7."""

    def fix_angle_info(mnt_dict: dict) -> dict:
        mnt_dict["type"] = "FieldProjectionAngleMonitor"
        mnt_dict.pop("fields")
        mnt_dict.pop("medium")
        mnt_dict["proj_distance"] = 1e6
        return mnt_dict

    def fix_cartesian_info(mnt_dict: dict) -> dict:
        mnt_dict["type"] = "FieldProjectionCartesianMonitor"
        mnt_dict.pop("fields")
        mnt_dict.pop("medium")
        dist = mnt_dict.pop("plane_distance")
        mnt_dict["proj_distance"] = dist
        axis = mnt_dict.pop("plane_axis")
        mnt_dict["proj_axis"] = axis
        return mnt_dict

    def fix_kspace_info(mnt_dict: dict) -> dict:
        mnt_dict["type"] = "FieldProjectionKSpaceMonitor"
        mnt_dict.pop("fields")
        mnt_dict.pop("medium")
        mnt_dict["proj_distance"] = 1e6
        axis = mnt_dict.pop("u_axis")
        mnt_dict["proj_axis"] = axis
        return mnt_dict

    def fix_diffraction_info(mnt_dict: dict) -> dict:
        mnt_dict.pop("medium", None)
        mnt_dict.pop("orders_x", None)
        mnt_dict.pop("orders_y", None)
        return mnt_dict

    def fix_bloch_vec(mnt_dict: dict) -> dict:
        mnt_dict["bloch_vec"] = mnt_dict["bloch_vec"]["real"]
        return mnt_dict

    iterate_update_dict(
        update_dict=sim_dict,
        update_types={
            "Near2FarAngleMonitor": fix_angle_info,
            "Near2FarCartesianMonitor": fix_cartesian_info,
            "Near2FarKSpaceMonitor": fix_kspace_info,
            "DiffractionMonitor": fix_diffraction_info,
            "BlochBoundary": fix_bloch_vec,
        },
    )
    return sim_dict


@updates_from_version("1.6")
def update_1_6(sim_dict: dict) -> dict:
    """Updates version 1.6."""
    if "grid_size" in sim_dict:
        sim_dict.pop("grid_size")
    return sim_dict


@updates_from_version("1.5")
def update_1_5(sim_dict: dict) -> dict:
    """Updates version 1.5."""

    def fix_mode_field_mnt(mnt_dict: dict) -> dict:
        mnt_dict["type"] = "ModeSolverMonitor"
        return mnt_dict

    iterate_update_dict(update_dict=sim_dict, update_types={"ModeFieldMonitor": fix_mode_field_mnt})
    return sim_dict


@updates_from_version("1.4")
def update_1_4(sim_dict: dict) -> dict:
    """Updates version 1.4."""

    def fix_polyslab(geo_dict):
        """Fix a PolySlab dictionary."""
        geo_dict.pop("length", None)
        geo_dict.pop("center", None)

    def fix_modespec(ms_dict):
        """Fix a ModeSpec dictionary."""
        sort_by = ms_dict.pop("sort_by", None)
        if sort_by and sort_by != "largest_neff":
            log.warning(
                "ModeSpec.sort_by was removed in Tidy3D 1.5.0, reverting to sorting by "
                "largest effective index. Use ModeSpec.filter_pol to select polarization instead."
            )

    def fix_geometry_group(geo_dict):
        """Fix a GeometryGroup dictionary."""
        geo_dict.pop("center", None)

    update_types = {
        "PolySlab": fix_polyslab,
        "ModeSpec": fix_modespec,
        "GeometryGroup": fix_geometry_group,
    }

    iterate_update_dict(update_dict=sim_dict, update_types=update_types)

    return sim_dict


@updates_from_version("1.3")
def update_1_3(sim_dict: dict) -> dict:
    """Updates version 1.3."""

    sim_dict["boundary_spec"] = {"x": {}, "y": {}, "z": {}}
    for dim, pml_layer in zip(["x", "y", "z"], sim_dict["pml_layers"]):
        sim_dict["boundary_spec"][dim]["plus"] = pml_layer
        sim_dict["boundary_spec"][dim]["minus"] = pml_layer
    sim_dict.pop("pml_layers")
    return sim_dict
