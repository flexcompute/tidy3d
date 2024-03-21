"""Monitor level data, store the DataArrays associated with a single charge monitor."""
from __future__ import annotations
from typing import Union, Tuple, Optional

from abc import ABC

import pydantic.v1 as pd

from ..monitor import PotentialMonitor, ChargeDensityMonitor, SemiConDevMonitorType
from ...base import skip_if_fields_missing
from ...base_sim.data.monitor_data import AbstractMonitorData
from ...data.data_array import SpatialDataArray
from ...data.dataset import TriangularGridDataset, TetrahedralGridDataset
from ...types import ScalarSymmetry, Coordinate, annotate_type
from ....constants import VOLT, PERCMCUBE

from ....log import log


class SemiConDevMonitorData(AbstractMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`ChargeMonitor`."""

    monitor: SemiConDevMonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: Tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        (0, 0, 0),
        title="Symmetry Center",
        description="Symmetry center of the original simulation in x, y, and z.",
    )

    @property
    def symmetry_expanded_copy(self) -> SemiConDevMonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()


class PotentialData(SemiConDevMonitorData):
    """Data associated with a :class:`PotentialMonitor`: spatial temperature field.

    # TODO: provide example!
    Example
    -------
    >>> from tidy3d import TemperatureMonitor, SpatialDataArray
    >>> import numpy as np
    >>> temp_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="temperature")
    >>> temp_mnt_data = TemperatureData(
    ...     monitor=temp_mnt, temperature=temp_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> temp_mnt_data_expanded = temp_mnt_data.symmetry_expanded_copy
    """

    monitor: PotentialMonitor = pd.Field(
        ..., title="Monitor", description="Potential monitor associated with the data."
    )

    potential: Optional[
        Union[SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])]
    ] = pd.Field(
        ...,
        title="potential",
        description="Spatial potential field.",
        units=VOLT,
    )

    @pd.validator("potential", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> PotentialData:
        """Return copy of self with symmetry applied."""

        if self.potential is None:
            return self.updated_copy(symmetry=(0, 0, 0))

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        new_phi = self.potential

        for dim in range(3):
            # do not expand monitor with zero size along symmetry direction
            # this is done because 2d unstructured data does not support this
            if self.symmetry[dim] == 1 and self.monitor.size[dim] != 0:
                new_phi = new_phi.reflect(axis=dim, center=self.symmetry_center[dim])

        return self.updated_copy(potential=new_phi, symmetry=(0, 0, 0))


class ChargeDensityData(SemiConDevMonitorData):
    """Data associated with a :class:`ChargeDensityMonitor`: spatial temperature field.

    # TODO: provide example!
    Example
    -------
    >>> from tidy3d import TemperatureMonitor, SpatialDataArray
    >>> import numpy as np
    >>> temp_data = SpatialDataArray(
    ...     np.ones((2, 3, 4)), coords={"x": [0, 1], "y": [0, 1, 2], "z": [0, 1, 2, 3]}
    ... )
    >>> temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="temperature")
    >>> temp_mnt_data = TemperatureData(
    ...     monitor=temp_mnt, temperature=temp_data, symmetry=(0, 1, 0), symmetry_center=(0, 0, 0)
    ... )
    >>> temp_mnt_data_expanded = temp_mnt_data.symmetry_expanded_copy
    """

    monitor: ChargeDensityMonitor = pd.Field(
        ..., title="Monitor", description="Charge density monitor associated with the data."
    )

    charge_density: Optional[
        Union[SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])]
    ] = pd.Field(
        ...,
        title="charge_density",
        description="Spatial charge density distribution.",
        units=PERCMCUBE,
    )

    @pd.validator("charge_density", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> ChargeDensityData:
        """Return copy of self with symmetry applied."""

        if self.charge_density is None:
            return self.updated_copy(symmetry=(0, 0, 0))

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        new_rho = self.charge_density

        for dim in range(3):
            # do not expand monitor with zero size along symmetry direction
            # this is done because 2d unstructured data does not support this
            if self.symmetry[dim] == 1 and self.monitor.size[dim] != 0:
                new_rho = new_rho.reflect(axis=dim, center=self.symmetry_center[dim])

        return self.updated_copy(charge_density=new_rho, symmetry=(0, 0, 0))


SemiConDevMonitorDataType = Union[PotentialData, ChargeDensityData]
