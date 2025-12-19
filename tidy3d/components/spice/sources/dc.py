"""
Our DC sources ultimately need to follow this standard form if we want to enable full electrical integration.

```
11.3.2 .DC: DC Transfer Function

General form:

    .dc srcnam vstart vstop vincr [src2 start2 stop2 incr2]

Examples:

    .dc VIN 0.25 5.0 0.25
    .dc VDS 0 10 .5 VGS 0 5 1
    .dc VCE 0 10 .25 IB 0 10u 1u
    .dc RLoad 1k 2k 100
    .dc TEMP -15 75 5
```

"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat1D
from tidy3d.constants import AMP, VOLT
from tidy3d.constants import inf as td_inf
from tidy3d.log import log


class DCVoltageSource(Tidy3dBaseModel):
    """
    DC voltage source in volts.

    Notes
    -----

        This voltage refers to potential above the equivalent simulation ground. Currently, electrical ports
        are not defined.

    Examples
    --------
    >>> import tidy3d as td
    >>> voltages = [-0.5, 0, 1, 2, 3, 4]
    >>> voltage_source = td.DCVoltageSource(voltage=voltages)
    """

    name: Optional[str] = pd.Field(
        None,
        title="Name",
        description="Unique name for the DC voltage source",
        min_length=1,
    )

    voltage: ArrayFloat1D = pd.Field(
        ...,
        title="Voltage",
        description="DC voltage usually used as source in :class:`VoltageBC` boundary conditions.",
        units=VOLT,
    )

    # TODO: This should have always been in the field above but was introduced wrongly as a
    # standalone field. Keeping for compatibility, remove in 3.0.
    units: Literal[VOLT] = VOLT

    @pd.validator("voltage")
    def check_voltage(cls, val):
        for v in val:
            if v == td_inf:
                raise ValueError(f"Voltages must be finite. Currently  voltage={val}.")
        return val

    @staticmethod
    def _count_unique_with_tolerance(arr, rtol=1e-9, atol=1e-12):
        """Count unique values treating values within tolerance as duplicates.

        Uses sorted comparison to group values that are practically equal
        due to floating-point representation differences (e.g., single vs double precision).
        """
        if len(arr) == 0:
            return 0
        sorted_arr = np.sort(arr)
        # Count values that are "different enough" from their predecessor
        unique_count = 1
        for i in range(1, len(sorted_arr)):
            if not np.isclose(sorted_arr[i], sorted_arr[i - 1], rtol=rtol, atol=atol):
                unique_count += 1
        return unique_count

    @pd.validator("voltage")
    def check_repeated_voltage(cls, val):
        """Warn if repeated voltage values are present, treating 0 and -0 as the same value.

        Uses tolerance-based comparison to handle floating-point representation
        differences (e.g., values from single vs double precision sources).
        """
        # Normalize all zero values (both 0.0 and -0.0) to 0.0 so they are treated as duplicates
        normalized = np.where(np.isclose(val, 0, atol=1e-10), 0.0, val)
        unique_count = cls._count_unique_with_tolerance(normalized)
        if unique_count < len(val):
            log.warning(
                "Duplicate voltage values detected in 'voltage' array. "
                f"Found {len(val)} values but only {unique_count} are unique. "
                "Note: values within floating-point tolerance are considered duplicates."
            )
        return val


class GroundVoltage(Tidy3dBaseModel):
    """
    Ground voltage source (0V reference).


    Notes
    -----
    This source explicitly sets the ground reference (0V) for the simulation.
    It is equivalent to :class:`DCVoltageSource(voltage=0)` but more explicit about
    establishing the ground reference.

    If no :class:`GroundVoltage` is specified, the smallest voltage among all
    sources will be considered as the ground reference. Note that the boundary
    conditions defined using a voltage array will be ignored during this
    process and cannot be used as a default ground.

    Example
    -------
    >>> import tidy3d as td
    >>> ground_source = td.GroundVoltage()
    >>> voltage_bc = td.VoltageBC(source=ground_source)
    """


class DCCurrentSource(Tidy3dBaseModel):
    """
    DC current source in amperes.

    Example
    -------
    >>> import tidy3d as td
    >>> current_source = td.DCCurrentSource(current=0.4)
    """

    name: Optional[str] = pd.Field(
        None,
        title="Name",
        description="Unique name for the DC current source",
        min_length=1,
    )

    current: pd.FiniteFloat = pd.Field(
        title="Current",
        description="DC current usually used as source in :class:`CurrentBC` boundary conditions.",
        units=AMP,
    )

    # TODO: This should have always been in the field above but was introduced wrongly as a
    # standalone field. Keeping for compatibility, remove in 3.0.
    units: Literal[AMP] = AMP
