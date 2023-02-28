"""Material Library API."""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import Field, parse_obj_as, validator
from tidy3d.components.medium import MediumType

from .http_management import http
from .types import Queryable


class MaterialLibray(Queryable, smart_union=True):
    """Material Library Resource interface."""

    id: str = Field(title="Material Library ID", description="Material Library ID")
    name: str = Field(title="Material Library Name", description="Material Library Name")
    medium: Optional[MediumType] = Field(title="medium", description="medium", alias="calcResult")
    medium_type: Optional[str] = Field(
        title="medium type", description="medium type", alias="mediumType"
    )
    json_input: Optional[dict] = Field(
        title="json input", description="original input", alias="jsonInput"
    )

    # pylint: disable=no-self-argument
    @validator("medium", "json_input", pre=True)
    def parse_result(cls, values):
        """Automatically parsing medium and json_input from string to object."""
        return json.loads(values)

    # pylint: disable=arguments-differ
    @classmethod
    def list(cls) -> List[MaterialLibray]:
        """List all material libraries.

        Returns
        -------
        tasks : List[:class:`.MaterialLibray`]
            List of material libraries/
        """
        resp = http.get("tidy3d/libraries")
        return parse_obj_as(List[MaterialLibray], resp) if resp else None
