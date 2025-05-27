from __future__ import annotations

from typing import Any, Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.log import log

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

    facecolor: str = pd.Field(
        "",
        title="Face color",
        description="Color applied to the faces in visualization.",
    )

    edgecolor: Optional[str] = pd.Field(
        "",
        title="Edge color",
        description="Color applied to the edges in visualization.",
    )

    alpha: Optional[pd.confloat(ge=0.0, le=1.0)] = pd.Field(
        1.0,
        title="Opacity",
        description="Opacity/alpha value in plotting between 0 and 1.",
    )

    @pd.validator("facecolor", always=True)
    def validate_color(value: str) -> str:
        return is_valid_color(value)

    @pd.validator("edgecolor", always=True)
    def validate_and_copy_color(value: str, values: dict[str, Any]) -> str:
        if (value == "") and "facecolor" in values:
            return is_valid_color(values["facecolor"])

        return is_valid_color(value)
