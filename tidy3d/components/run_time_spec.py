# Defines specifications for how long to run a simulation
import pydantic.v1 as pd

from .base import Tidy3dBaseModel


class RunTimeSpec(Tidy3dBaseModel):
    """Defines specification for how long to run a simulation when added to ``Simulation.run_time``.

    Notes
    -----

        The evaluated ``run_time`` will be computed from a ``RunTimeSpec()`` as follows:

        .. math::

            \\text{run_time} =  \\text{source_factor} * T_{src_max} + \\text{quality_factor} n_{max} L_{max} / c_{0}

        Where: ``source_factor`` and ``quality_factor`` are fields in the spec,
        :math:`T_{src_max}` is the longest time that a source is non-zero,
        :math:`n_{max}` is the maximum refractive index in the simulation,
        :math:`L_{max}` is the distance along the largest dimension in the simulation, and
        :math:`c_0` is the speed of light in vacuum.

    """

    quality_factor: pd.PositiveFloat = pd.Field(
        ...,
        title="Quality Factor",
        description="Quality factor expected in the device. This determines how long the "
        "simulation will run as it assumes a field decay time that scales proportionally to "
        "this value.",
    )

    source_factor: pd.PositiveFloat = pd.Field(
        3,
        title="Source Factor",
        description="The contribution to the ``run_time`` from the longest source is computed from "
        "the ``source_time`` length times ``source_factor``. Larger values provide more buffer "
        "at the expense of potentially giving ``run_time`` values that are larger than needed.",
    )
