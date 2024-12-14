"""
This class defines standard SPICE analysis types (electrical simulations configurations).
"""

from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.spice.sources.dc import MultiStaticTransferSourceDC, StaticTransferSourceDC


class OperatingPointDC(Tidy3dBaseModel):
    """
    Equivalent to Section 11.1.2 in the ngspice manual.
    """


class TransferFunctionDC(Tidy3dBaseModel):
    """This class sets parameters used in DC simulations.

    Ultimately, equivalent to Section 11.3.2 in the ngspice manual.

    Example
    -------
    TODOUPDATE Example.
    >>> import tidy3d as td
    >>> dc_spec = td.TransferFunctionDC(dv=0.1)

    This class sets some Charge tolerance parameters.

    Example
    -------
    >>> import tidy3d as td
    >>> charge_settings = td.ChargeToleranceSpec(abs_tol=1e8, rel_tol=1e-10, max_iters=30)
    """

    sources: StaticTransferSourceDC | MultiStaticTransferSourceDC = []

    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(
        default=1e10,
        title="Absolute tolerance.",
        description="Absolute tolerance used as stop criteria when converging towards a solution. Should be equivalent to the"
        "SPICE ABSTOL DC transfer parameter. TODO MARC units, TODO check equivalence. TODO what does this mean?",
    )

    relative_tolerance: Optional[pd.PositiveFloat] = pd.Field(
        default=1e-10,
        title="Relative tolerance.",
        description="Relative tolerance used as stop criteria when converging towards a solution.  Should be equivalent to the"
        "SPICE RELTOL DC transfer parameter. TODO MARC units, TODO check equivalence. TODO what does this "
        "mean?",
    )

    dc_iteration_limit: Optional[pd.PositiveInt] = pd.Field(
        default=30,
        title="Maximum number of iterations.",
        description="Indicates the maximum number of iterations to be run. "
        "The solver will stop either when this maximum of iterations is met "
        "or when the tolerance criteria has been met. Should be equivalent to the ngspice ITL1 parameter. TODO units, "
        "TODO devsim ngspice equivalence",
    )
