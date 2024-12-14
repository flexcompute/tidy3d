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

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel


class AbstractStaticTransferSourceDC(Tidy3dBaseModel):
    name: str
    start: Optional[pd.PositiveFloat] = 0
    stop: Optional[pd.PositiveFloat] | None = None
    """
    TODOMARC chat Either they define the stop or they define the step, how do we want to enforce, validator?
    """

    step: Optional[pd.PositiveFloat] = pd.Field(
        1.0,
        title="Bias step.",
        description="By default, a solution is computed at 0 bias. "
        "If a bias different than 0 is requested, DEVSIM will start at 0 and increase bias "
        "at 'dV' intervals until the required bias is reached. ",
    )
    # TODO units


StaticTransferSourceDC = AbstractStaticTransferSourceDC
MultiStaticTransferSourceDC = list[StaticTransferSourceDC]
