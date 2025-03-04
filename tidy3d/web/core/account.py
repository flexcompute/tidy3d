"""Tidy3d user account."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic.v1 import Extra, Field

from .http_util import http
from .types import Tidy3DResource


class Account(Tidy3DResource, extra=Extra.allow):
    """Tidy3D User Account."""

    allowance_cycle_type: Optional[str] = Field(
        None,
        title="AllowanceCycleType",
        description="Daily or Monthly",
        alias="allowanceCycleType",
    )
    credit: Optional[float] = Field(
        0, title="credit", description="Current FlexCredit balance", alias="credit"
    )
    credit_expiration: Optional[datetime] = Field(
        None,
        title="creditExpiration",
        description="Expiration date",
        alias="creditExpiration",
    )
    allowance_current_cycle_amount: Optional[float] = Field(
        0,
        title="allowanceCurrentCycleAmount",
        description="Daily/Monthly free simulation balance",
        alias="allowanceCurrentCycleAmount",
    )
    allowance_current_cycle_end_date: Optional[datetime] = Field(
        None,
        title="allowanceCurrentCycleEndDate",
        description="Daily/Monthly free simulation balance expiration date",
        alias="allowanceCurrentCycleEndDate",
    )
    daily_free_simulation_counts: Optional[int] = Field(
        0,
        title="dailyFreeSimulationCounts",
        description="Daily free simulation counts",
        alias="dailyFreeSimulationCounts",
    )

    @classmethod
    def get(cls):
        """Get user account information.

        Parameters
        ----------

        Returns
        -------
        account : Account
        """
        resp = http.get("tidy3d/py/account")
        if resp:
            account = Account(**resp)
            return account
        else:
            return None
