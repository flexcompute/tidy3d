"""Utilities for converting between tidy3d versions."""
import json

import pydantic as pd

from .version import __version__
from .log import FileError, SetupError
from .components.simulation import Simulation


"""Storing version numbers."""


class Version(pd.BaseModel):
    """Stores a version number (excluding patch)."""

    major: int
    minor: int

    @classmethod
    def from_string(cls, string) -> "Version":
        """Return Version from a version string."""
        major, minor, _ = string.split(".")
        return cls(major=major, minor=minor)

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

    filename: str
    sim_dict: dict = None

    @pd.validator("sim_dict", always=True)
    def open_file(cls, _, values):
        """Dictionary representing the simulation loaded from file."""
        fname = values.get("filename")
        if fname is None:
            raise SetupError("file not given.")
        try:
            with open(fname, "r") as f:
                sim_dict = json.load(f)
        except Exception as e:
            raise FileError(f"Could not load file {fname}") from e
        return sim_dict

    @property
    def version(self) -> Version:
        """Version of the supplied file."""
        version_string = self.sim_dict.get("version")
        if version_string is None:
            raise SetupError("Could not find a version in the supplied json.")
        return Version.from_string(version_string)

    def update_to_current(self) -> Simulation:
        """Update supplied file to current."""
        while self.version != CurrentVersion:
            update_fn = UPDATE_MAP.get(self.version)
            if update_fn is None:
                raise SetupError(f"version {self.version} not found in update map.")
            self.sim_dict = update_fn(self.sim_dict)
        return Simulation.parse_obj(self.sim_dict)


"""Update conversion functions."""


def update_1_3(sim_dict: dict) -> dict:
    """Updates version 1.3 to 1.4."""

    sim_dict["boundary_spec"] = {"x": {}, "y": {}, "z": {}}
    for dim, pml_layer in zip(["x", "y", "z"], sim_dict["pml_layers"]):
        sim_dict["boundary_spec"][dim]["plus"] = pml_layer
        sim_dict["boundary_spec"][dim]["minus"] = pml_layer
    sim_dict.pop("pml_layers")
    sim_dict["version"] = "1.4.0"
    return sim_dict


""" Map the "from" version to it's corresponding function."""

v1_3_0 = Version(major=1, minor=3)

UPDATE_MAP = {v1_3_0: update_1_3}
