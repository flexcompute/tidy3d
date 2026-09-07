from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import (
    Field,
    model_validator,
)

from tidy3d.components.data.dataset import (
    PointCloudFieldDataset,
    PointCloudPermittivityDataset,
)
from tidy3d.components.data.em_fields import frequency_normalized_field_components
from tidy3d.components.data.monitor_data.base import MonitorData
from tidy3d.components.monitor import (
    PointCloudFieldMonitor,
    PointCloudPermittivityMonitor,
)
from tidy3d.components.validators import enforce_monitor_fields_present
from tidy3d.exceptions import Tidy3dNotImplementedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.compat import Self
    from tidy3d.components.source.base import Source


class PointCloudFieldData(MonitorData, PointCloudFieldDataset):
    """
    Data associated with a :class:`.PointCloudFieldMonitor`: scalar components of E, H, and
    ``D / epsilon_0`` at point-cloud coordinates, where ``epsilon_0`` is the vacuum
    permittivity.

    Example
    -------
    >>> from tidy3d import IndexedFreqDataArray, PointCloudFieldMonitor, PointDataArray
    >>> points = PointDataArray(
    ...     [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
    ...     coords={"index": [0, 1], "axis": [0, 1, 2]},
    ... )
    >>> field = IndexedFreqDataArray(
    ...     np.ones((2, 1)) + 0j,
    ...     coords={"index": [0, 1], "f": [200e12]},
    ... )
    >>> monitor = PointCloudFieldMonitor(points=points, freqs=[200e12], fields=["Ex"], name="pc")
    >>> data = PointCloudFieldData(monitor=monitor, points=points, Ex=field)
    """

    monitor: PointCloudFieldMonitor = Field(
        ..., title="Monitor", description="Frequency-domain point-cloud field monitor."
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @model_validator(mode="after")
    def _frequencies_match_monitor(self) -> Self:
        """Ensure stored field frequency coordinates match the associated monitor."""
        monitor_freqs = np.asarray(self.monitor.freqs)
        for field_name, field_data in self.field_components.items():
            field_freqs = np.asarray(field_data.coords["f"].values)
            if not np.array_equal(field_freqs, monitor_freqs):
                self._raise_validation_error_at_loc(
                    f"Field component '{field_name}' has frequency coordinates that do not "
                    "match the associated point-cloud field monitor frequencies.",
                    field_name,
                )
        return self

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> PointCloudFieldData:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy(
            deep=False,
            update=frequency_normalized_field_components(self.field_components, source_spectrum_fn),
        )

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Reject adjoint use until a batched point-cloud adjoint source is available."""
        if not dataset_names:
            return []

        raise Tidy3dNotImplementedError(
            "Adjoint objectives depending on PointCloudFieldData are currently unsupported."
        )


class PointCloudPermittivityData(MonitorData, PointCloudPermittivityDataset):
    """Data associated with a :class:`.PointCloudPermittivityMonitor`.

    Diagonal permittivity components are indexed by requested point row and frequency. The ``points``
    array stores the requested coordinates, while each component value is sampled from its nearest
    native Yee-grid location.
    """

    monitor: PointCloudPermittivityMonitor = Field(
        ..., title="Monitor", description="Frequency-domain point-cloud permittivity monitor."
    )

    @model_validator(mode="after")
    def _frequencies_match_monitor(self) -> Self:
        """Ensure stored component frequency coordinates match the associated monitor."""
        monitor_freqs = np.asarray(self.monitor.freqs)
        for component_name, component_data in self.field_components.items():
            component_freqs = np.asarray(component_data.coords["f"].values)
            if not np.array_equal(component_freqs, monitor_freqs):
                self._raise_validation_error_at_loc(
                    f"Permittivity component '{component_name}' has frequency coordinates that "
                    "do not match the associated point-cloud permittivity monitor frequencies.",
                    component_name,
                )
        return self

    def normalize(
        self, source_spectrum_fn: Callable[[float], complex]
    ) -> PointCloudPermittivityData:
        """Return copy of self; permittivity data is not source-normalized."""
        return self.copy(deep=False)

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Reject adjoint use until a batched point-cloud adjoint source is available."""
        if not dataset_names:
            return []

        raise Tidy3dNotImplementedError(
            "Adjoint objectives depending on PointCloudPermittivityData are currently unsupported."
        )
