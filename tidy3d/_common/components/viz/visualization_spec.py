from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from tidy3d._common.components.base import Tidy3dBaseModel
from tidy3d._common.log import log

if TYPE_CHECKING:
    from pydantic import ValidationInfo

MATPLOTLIB_IMPORTED = True
try:
    from matplotlib.colors import is_color_like
except ImportError:
    is_color_like = None
    MATPLOTLIB_IMPORTED = False


def is_valid_color(value: str) -> str:
    if not MATPLOTLIB_IMPORTED:
        log.warning(
            "matplotlib was not successfully imported, but is required "
            "to validate colors in the VisualizationSpec. The specified colors "
            "have not been validated."
        )
    else:
        if is_color_like is not None and not is_color_like(value):
            raise ValueError(f"{value} is not a valid plotting color")

    return value


class VisualizationSpec(Tidy3dBaseModel):
    """Defines specification for visualization when used with plotting functions."""

    facecolor: str = Field(
        "",
        title="Face color",
        description="Color applied to the faces in visualization.",
    )

    edgecolor: str = Field(
        "",
        title="Edge color",
        description="Color applied to the edges in visualization.",
    )

    alpha: float = Field(
        1.0,
        title="Opacity",
        description="Opacity/alpha value in plotting between 0 and 1.",
        ge=0,
        le=1,
    )

    @field_validator("facecolor")
    @classmethod
    def _validate_facecolor(cls, value: str) -> str:
        return is_valid_color(value)

    @field_validator("edgecolor")
    @classmethod
    def _ensure_edgecolor(cls, value: str, info: ValidationInfo) -> str:
        # if no explicit edgecolor given, fall back to facecolor
        if (value == "") and "facecolor" in info.data:
            return is_valid_color(info.data["facecolor"])
        return is_valid_color(value)
