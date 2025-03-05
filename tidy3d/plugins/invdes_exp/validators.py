# validator utilities for invdes plugin

import typing

import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.base import skip_if_fields_missing

# warn if pixel size is > PIXEL_SIZE_WARNING_THRESHOLD * (minimum wavelength in material)
PIXEL_SIZE_WARNING_THRESHOLD = 0.1


def ignore_inherited_field(field_name: str) -> typing.Callable:
    """Create validator that ignores a field inherited but not set by user."""

    @pd.validator(field_name, always=True)
    def _ignore_field(cls, val):
        """Ignore supplied field value and warn."""
        if val is not None:
            td.log.warning(
                f"Field '{field_name}' was supplied but the 'invdes' plugin will automatically "
                "set this field internally using the design region specifications. "
                "The supplied value will be ignored. "
            )
        return None

    return _ignore_field


def check_pixel_size(sim_field_name: str):
    """make validator to check the pixel size of sim or list of sims in an ``InverseDesign``."""

    def check_pixel_size_sim(sim: td.Simulation, pixel_size: float, index: int = None) -> None:
        """Check a pixel size compared to the simulation min wvl in material."""
        if not sim.sources:
            td.log.warning(
                "Cannot validate pixel size in design region: simulation has no sources. "
                "Please ensure the pixel size is appropriate for your target wavelength."
            )
            return

        if pixel_size > PIXEL_SIZE_WARNING_THRESHOLD * sim.wvl_mat_min:
            sim_string = f"simulations[{index}]" if index else "the simulation"

            td.log.warning(
                f"'DesignRegion.pixel_size' is '{pixel_size}', which is large compared to "
                f"the minimum wavelength in {sim_string}: '{sim.wvl_mat_min}'. For best results, "
                f"we recommend setting it at most {PIXEL_SIZE_WARNING_THRESHOLD} times the minimum "
                f"wavelength ({PIXEL_SIZE_WARNING_THRESHOLD * sim.wvl_mat_min}). "
                "Note: to set the grid size within the design region independent of the parameter "
                "array resolution, one can set 'DesignRegion.override_structure_dl'."
            )

    @pd.root_validator(allow_reuse=True)
    @skip_if_fields_missing(["design_region"], root=True)
    def _check_pixel_size(cls, values):
        """Make sure region pixel_size isn't too large compared to sim's wavelength in material."""
        sim = values.get(sim_field_name)
        region = values.get("design_region")
        pixel_size = region.pixel_size

        if not sim and region:
            return values

        if isinstance(sim, (list, tuple)):
            for i, s in enumerate(sim):
                check_pixel_size_sim(sim=s, pixel_size=pixel_size, index=i)
        else:
            check_pixel_size_sim(sim=sim, pixel_size=pixel_size)

        return values

    return _check_pixel_size
