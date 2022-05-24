"""Class holding methods to upgrade old versions of :class:`.Simulation` json dictionaries."""
from typing import Dict

from .version import __version__


class Updater:
    """Updater class. Every time a minor release is made, a function updating the previous minor
    release to the new one must be added. The assumption is that patch versions do not need to be
    updated.

    Example
    -------
    >>> old_sim_dict = {"version": "1.2.1"}
    >>> new_sim_dict = Updater().update(old_sim_dict)
    """

    def __getitem__(self, key):
        """Call appropriate upgrade function based on version string."""
        major, minor, _ = key.split(".")
        func_name = f"update_{major}_{minor}"
        return getattr(self, func_name)

    @staticmethod
    def update_1_3(sim_dict: Dict):
        """Updates version 1.3 to 1.4."""

        sim_dict["boundary_spec"] = {"x": {}, "y": {}, "z": {}}
        for dim, pml_layer in zip(["x", "y", "z"], sim_dict["pml_layers"]):
            sim_dict["boundary_spec"][dim]["plus"] = pml_layer
            sim_dict["boundary_spec"][dim]["minus"] = pml_layer
        sim_dict.pop("pml_layers")
        sim_dict["version"] = "1.4.0"
        return sim_dict

    def update(self, sim_dict: Dict):
        """Update an old Simulation dict to the current Tidy3D version."""

        new_dict = sim_dict.copy()
        major, minor, _ = new_dict["version"].split(".")
        current_major, current_minor, _ = __version__.split(".")

        while major != current_major or minor != current_minor:
            new_dict = self[new_dict["version"]](new_dict)
            major, minor, _ = new_dict["version"].split(".")

        new_dict["version"] = __version__

        return new_dict
