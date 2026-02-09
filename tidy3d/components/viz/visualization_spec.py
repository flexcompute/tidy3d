from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import ValidationInfo

# Cached reference to is_color_like, set on first use
_is_color_like = None

# Check if matplotlib is available without importing it
MATPLOTLIB_IMPORTED = find_spec("matplotlib") is not None


def is_valid_color(value: str) -> str:
    global _is_color_like
    # Check MATPLOTLIB_IMPORTED first to allow test mocking
    if not MATPLOTLIB_IMPORTED:
        log.warning(
            "matplotlib was not successfully imported, but is required "
            "to validate colors in the VisualizationSpec. The specified colors "
            "have not been validated."
        )
        return value

    if _is_color_like is None:
        try:
            from matplotlib.colors import is_color_like

            _is_color_like = is_color_like
        except ImportError:
            _is_color_like = False  # Sentinel to indicate import failed

    if _is_color_like is False:
        log.warning(
            "matplotlib was not successfully imported, but is required "
            "to validate colors in the VisualizationSpec. The specified colors "
            "have not been validated."
        )
    else:
        if not _is_color_like(value):
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
