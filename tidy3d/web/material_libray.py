"""Material Library API."""

from __future__ import annotations

import json

from pydantic import Field, parse_obj_as, validator

from tidy3d.components.medium import MediumType

from .http_management import http
from .types import Queryable


class MaterialLibray(Queryable, smart_union=True):
    """Material Library Resource interface."""

    id: str = Field(title="Material Library ID", description="Material Library ID")
    name: str = Field(title="Material Library Name", description="Material Library Name")
    medium: MediumType | None = Field(title="medium", description="medium", alias="calcResult")
    medium_type: str | None = Field(
        title="medium type", description="medium type", alias="mediumType"
    )
    json_input: dict | None = Field(
        title="json input", description="original input", alias="jsonInput"
    )

    # pylint: disable=no-self-argument
    @validator("medium", "json_input", pre=True)
    def parse_result(cls, values):
        """Automatically parsing medium and json_input from string to object."""
        return json.loads(values)

    # pylint: disable=arguments-differ
    @classmethod
    def list(cls) -> list[MaterialLibray]:
        """List all material libraries.

        Returns
        -------
        tasks : List[:class:`.MaterialLibray`]
            List of material libraries/
        """
        resp = http.get("tidy3d/libraries")
        return parse_obj_as(list[MaterialLibray], resp) if resp else None
