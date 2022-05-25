"""Utilities for converting between tidy3d versions."""
import json
import functools

import pydantic as pd

from .version import __version__
from .log import FileError, SetupError


"""Storing version numbers."""


class Version(pd.BaseModel):
    """Stores a version number (excluding patch)."""

    major: int
    minor: int

    @classmethod
    def from_string(cls, string) -> "Version":
        """Return Version from a version string."""
        try:
            major, minor, _ = string.split(".")
            version = cls(major=major, minor=minor)
        except Exception as e:
            raise SetupError(f"version string {version_string} can't be parsed.") from e
        return version

    @property
    def as_tuple(self):
        """version as a tuple, leave out patch for now."""
        return (self.major, self.minor, None)

    def __hash__(self):
        """define a hash."""
        return hash(self.as_tuple)


CurrentVersion = Version.from_string(__version__)

"""Class for updating simulation objects."""


class Updater(pd.BaseModel):
    """Converts a tidy3d simulation.json file to an up-to-date Simulation instance."""

    sim_dict: dict

    @classmethod
    def from_file(cls, fname: str) -> "Updater":
        """Dictionary representing the simulation loaded from file."""

        try:
            with open(fname, "r") as f:
                sim_dict = json.load(f)
        except Exception as e:
            raise FileError(f"Could not load file {fname}") from e
        return cls(sim_dict=sim_dict)

    @property
    def version(self) -> Version:
        """Version of the supplied file."""
        version_string = self.sim_dict.get("version")
        if version_string is None:
            raise SetupError("Could not find a version in the supplied json.")
        return Version.from_string(version_string)

    def update_to_current(self) -> dict:
        """Update supplied simulation dictionary to current version."""
        while self.version != CurrentVersion:
            update_fn = UPDATE_MAP.get(self.version)
            if update_fn is None:
                raise SetupError(f"version {self.version} not found in update map.")
            self.sim_dict = update_fn(self.sim_dict)
        return self.sim_dict


"""Update conversion functions."""

# versions will be dynamically mapped in this table when the update functions are initialized.
UPDATE_MAP = {}


def updates_to_version(version_from_string, version_to_string):
    """Decorates a sim_dict update function to change the version."""

    # make sure the version strings are legit
    from_version = Version.from_string(version_from_string)
    _ = Version.from_string(version_to_string)

    def decorator(update_fn):
        """The actual decorator that gets returned by `updates_to_version('x.y.z')`"""

        @functools.wraps(update_fn)
        def new_update_function(sim_dict: dict) -> dict:
            """Update function that automatically adds version string."""

            sim_dict_updated = update_fn(sim_dict)
            sim_dict_updated["version"] = version_to_string
            return sim_dict_updated

        UPDATE_MAP[from_version] = new_update_function

        return new_update_function

    return decorator


@updates_to_version(version_from_string="1.3.0", version_to_string="1.4.0")
def update_1_3(sim_dict: dict) -> dict:
    """Updates version 1.3 to 1.4."""

    sim_dict["boundary_spec"] = {"x": {}, "y": {}, "z": {}}
    for dim, pml_layer in zip(["x", "y", "z"], sim_dict["pml_layers"]):
        sim_dict["boundary_spec"][dim]["plus"] = pml_layer
        sim_dict["boundary_spec"][dim]["minus"] = pml_layer
    sim_dict.pop("pml_layers")
    return sim_dict
