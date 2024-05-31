"""Abstract base for monitor data structures."""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from ...data.dataset import Dataset
from ..monitor import AbstractMonitor


class AbstractMonitorData(Dataset, ABC):
    """Abstract base class of objects that store data pertaining to a single
    :class:`AbstractMonitor`.
    """

    monitor: AbstractMonitor = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    @property
    def symmetry_expanded_copy(self) -> AbstractMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()
