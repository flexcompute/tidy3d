"""Tidy3d system config."""

from __future__ import annotations

from typing import Optional

from pydantic.v1 import Extra, Field

from .http_util import http
from .types import Tidy3DResource


class SystemConfig(Tidy3DResource, extra=Extra.allow):
    """Tidy3D system config."""

    end_statuses: Optional[tuple] = Field(
        None,
        title="End Statuses",
        description="Tuple of status keys that signify that the task has completed.",
        alias="endStatuses",
    )
    run_statuses: Optional[tuple] = Field(
        None,
        title="Run Statuses",
        description="Tuple of ordered status keys that signify that the task is in progress.",
        alias="runStatuses",
    )

    @classmethod
    def get(cls):
        """Get user SystemConfig information.

        Parameters
        ----------

        Returns
        -------
        systemConfig : SystemConfig
        """
        resp = http.get("tidy3d/system/py/config")
        if resp:
            config = SystemConfig(**resp)
            return config
        else:
            return None
