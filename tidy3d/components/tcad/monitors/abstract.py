"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from tidy3d.components.base_sim.monitor import AbstractMonitor
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import ValidationInfo

    from tidy3d.components.types import ArrayFloat1D

BYTES_REAL = 4


class HeatChargeMonitor(AbstractMonitor, ABC):
    """Abstract base class for heat-charge monitors."""

    unstructured: bool = Field(
        True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid. Setting this to ``False`` "
        "is deprecated and will be removed in Tidy3D 3.0, after which monitor data is "
        "always returned on the solver's own grid; resample it with "
        ":meth:`HeatChargeMonitorData.to_spatial_data_array` where a Cartesian "
        ":class:`.SpatialDataArray` is needed. Most heat-charge monitors already accept only "
        "``True``; :class:`.TemperatureMonitor` and :class:`.SteadyPotentialMonitor` still "
        "take ``False`` in steady Heat, Conduction, and Charge runs, always with the "
        "deprecation warning above and, in Charge, a second warning that only unstructured "
        "monitors are supported there. Unsteady Heat rejects it outright.",
    )

    @field_validator("unstructured")
    @classmethod
    def _unstructured_false_deprecated(cls, val: bool) -> bool:
        """Warn that structured monitor output is on its way out.

        Only reachable on the monitors that still type this field as ``bool``; the rest
        narrow it to ``Literal[True]`` and reject ``False`` outright.
        """
        if not val:
            log.warning(
                "'unstructured=False' is deprecated and will be removed in Tidy3D 3.0. "
                "Monitor data will always be returned on the solver's unstructured grid. Set "
                "'unstructured=True', and where a Cartesian grid is needed, resample the "
                "recorded field with 'to_spatial_data_array()' on the monitor data.",
                # 'updated_copy' re-validates, so a bias sweep would repeat this per point
                log_once=True,
            )
        return val

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_fields(cls, data: Any, info: ValidationInfo) -> Any:
        """Drop fields removed from heat-charge monitors when loading legacy files.

        Gated on the ``from_file`` load context so it fires only on real file loads
        (``from_file``/``from_hdf5``), wherever a monitor appears — standalone, in monitor
        data, or nested in a simulation/mesher. In-memory construction (object or dict)
        still raises via ``extra="forbid"``, keeping one consistent validation contract.
        """
        context = info.context or {}
        if context.get("from_file") and hasattr(data, "get") and "conformal" in data:
            # ``conformal`` was removed in 2.12; drop it so legacy files clear the
            # ``extra="forbid"`` check instead of failing to load.
            conformal = data.pop("conformal", None)
            # Only ``conformal=True`` changed behavior (it requested conformal meshing);
            # ``conformal=False`` already matched today's interpolation-based behavior, so
            # drop it quietly and warn only when the loaded config actually differs.
            if conformal:
                log.warning(
                    "The 'conformal' flag was removed from heat-charge monitors in 2.12. "
                    "This file sets conformal=True, but the flag is now ignored: monitor "
                    "geometry no longer affects meshing, and recorded data follows the "
                    "current interpolation-based behavior."
                )
        return data

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)
