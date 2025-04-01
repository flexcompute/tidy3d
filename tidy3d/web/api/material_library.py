"""Material Library API."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import Field, field_validator, parse_obj_as

from tidy3d.components.medium import MediumType
from tidy3d.web.core.http_util import http
from tidy3d.web.core.types import Queryable


class MaterialLibrary(Queryable):
    """Material Library Resource interface."""

    id: str = Field(
        title="Material Library ID",
        description="Material Library ID",
    )
    name: str = Field(
        title="Material Library Name",
        description="Material Library Name",
    )
    medium: Optional[MediumType] = Field(
        None,
        title="medium",
        description="medium",
        alias="calcResult",
    )
    medium_type: Optional[str] = Field(
        None,
        title="medium type",
        description="medium type",
        alias="mediumType",
    )
    json_input: Optional[dict] = Field(
        None,
        title="json input",
        description="original input",
        alias="jsonInput",
    )

    @field_validator("medium", "json_input", mode="before")
    @classmethod
    def parse_result(cls, values):
        """Automatically parsing medium and json_input from string to object."""
        return json.loads(values)

    @classmethod
    def list(cls) -> list[MaterialLibrary]:
        """List all material libraries.

        Returns
        -------
        tasks : list[:class:`.MaterialLibrary`]
            List of material libraries/
        """
        resp = http.get("tidy3d/libraries")
        return parse_obj_as(list[MaterialLibrary], resp) if resp else None
