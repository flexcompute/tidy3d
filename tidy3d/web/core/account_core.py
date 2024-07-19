"""Tidy3d user Account."""

from __future__ import annotations

from pydantic.v1 import Extra, Field
from .http_util import http
from .types import Tidy3DResource
from typing import Optional
from datetime import datetime


class UserAccount(Tidy3DResource, extra=Extra.allow):
    """Tidy3D User Account."""

    allowance_cycle_type: str = Field(
        ..., title="AllowanceCycleType", description="Daily or Monthly",alias="allowanceCycleType"
    )
    credit: float = Field(..., title="credit", description="Current FlexCredit balance", alias="credit")
    credit_expiration: Optional[datetime] = Field(
        None, title="creditExpiration", description="Expiration date", alias="creditExpiration"
    )
    allowance_current_cycle_amount: float = Field(
        ..., title="allowanceCurrentCycleAmount", description="Daily/Monthly free simulation balance",
        alias="allowanceCurrentCycleAmount"
    )
    allowance_current_cycle_end_date: Optional[datetime] = Field(
        None, title="allowanceCurrentCycleEndDate", description="Daily/Monthly free simulation balance"
        "expiration date", alias="allowanceCurrentCycleEndDate"
    )
    daily_free_simulation_counts: int = Field(
        ..., title="dailyFreeSimulationCounts", description="Daily free simulation counts",
        alias="dailyFreeSimulationCounts"
    )

    @classmethod
    def get(cls):
        """Get user account information.

        Parameters
        ----------

        Returns
        -------
        account : UserAccount
        """
        resp = http.get(f"tidy3d/py/account")
        if resp:
            account = UserAccount(**resp)
            return account
        else:
            return None
