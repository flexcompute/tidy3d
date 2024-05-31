"""Material Library API."""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic.v1 import Field, parse_obj_as, validator

from ...components.medium import MediumType
from ..core.http_util import http
from ..core.types import Queryable


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

    @validator("medium", "json_input", pre=True)
    def parse_result(cls, values):
        """Automatically parsing medium and json_input from string to object."""
        return json.loads(values)

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
