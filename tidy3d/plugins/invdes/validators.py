# validator utilities for invdes plugin
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import field_validator, model_validator

import tidy3d as td

if TYPE_CHECKING:
    from typing import Callable, Optional

    from pydantic._internal._decorators import ModelValidatorDecoratorInfo, PydanticDescriptorProxy

# warn if pixel size is > PIXEL_SIZE_WARNING_THRESHOLD * (minimum wavelength in material)
PIXEL_SIZE_WARNING_THRESHOLD = 0.1


def ignore_inherited_field(field_name: str) -> Callable:
    """Create validator that ignores a field inherited but not set by user."""

    @field_validator(field_name)
    def _ignore_field(val: Any) -> None:
        """Ignore supplied field value and warn."""
        if val is not None:
            td.log.warning(
                f"Field '{field_name}' was supplied but the 'invdes' plugin will automatically "
                "set this field internally using the design region specifications. "
                "The supplied value will be ignored. "
            )
        return

    return _ignore_field


def check_pixel_size(sim_field_name: str) -> PydanticDescriptorProxy[ModelValidatorDecoratorInfo]:
    """make validator to check the pixel size of sim or list of sims in an ``InverseDesign``."""

    def check_pixel_size_sim(
        sim: td.Simulation, pixel_size: float, index: Optional[int] = None
    ) -> None:
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
                f"the minimum wavelength in the material '{sim.wvl_mat_min}' in {sim_string}. For best results, "
                f"we recommend setting it at most {PIXEL_SIZE_WARNING_THRESHOLD} times the minimum "
                f"wavelength ({PIXEL_SIZE_WARNING_THRESHOLD * sim.wvl_mat_min}). "
                "Note: to set the grid size within the design region independent of the parameter "
                "array resolution, one can set 'DesignRegion.override_structure_dl'."
            )

    @model_validator(mode="after")
    def _check_pixel_size(self: Any) -> Any:
        """Make sure region pixel_size isn't too large compared to sim's wavelength in material."""
        sim = getattr(self, sim_field_name)
        region = self.design_region
        pixel_size = region.pixel_size

        if not sim and region:
            return self

        if isinstance(sim, (list, tuple)):
            for i, s in enumerate(sim):
                check_pixel_size_sim(sim=s, pixel_size=pixel_size, index=i)
        else:
            check_pixel_size_sim(sim=sim, pixel_size=pixel_size)

        return self

    return _check_pixel_size
