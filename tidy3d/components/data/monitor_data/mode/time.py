from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from tidy3d.components.data.data_array import (
    ModeAmpsTimeDataArray,
    ModeIndexDataArray,
)
from tidy3d.components.data.monitor_data.base import MonitorData
from tidy3d.components.monitor import ModeTimeMonitor

if TYPE_CHECKING:
    from collections.abc import Callable


class ModeTimeData(MonitorData):
    """Data associated with a :class:`.ModeTimeMonitor`: modal amplitude time series at a
    waveguide cross-section.

    Example
    -------
    >>> from tidy3d import ModeAmpsTimeDataArray, ModeIndexDataArray, ModeSpec
    >>> import numpy as np
    >>> direction = ["+", "-"]
    >>> t = [0, 1e-12, 2e-12]
    >>> mode_index = np.arange(3)
    >>> freqs = [2e14]
    >>> amp_coords = dict(direction=direction, t=t, mode_index=mode_index)
    >>> amp_data = ModeAmpsTimeDataArray(
    ...     (1+1j) * np.random.random((2, 3, 3)), coords=amp_coords
    ... )
    >>> n_complex = ModeIndexDataArray(
    ...     (1.5 + 0.01j) * np.ones((1, 3)), coords=dict(f=freqs, mode_index=mode_index)
    ... )
    >>> monitor = ModeTimeMonitor(
    ...     size=(2, 2, 0),
    ...     mode_spec=ModeSpec(num_modes=3),
    ...     interval=1,
    ...     name='mode_time',
    ... )
    >>> data = ModeTimeData(monitor=monitor, amps=amp_data, n_complex=n_complex)
    """

    monitor: ModeTimeMonitor = Field(
        title="Monitor",
        description="Time-domain mode monitor associated with the data.",
    )

    amps: ModeAmpsTimeDataArray = Field(
        title="Mode Amplitudes",
        description="Complex-valued modal amplitudes with dimensions (direction, t, mode_index).",
    )

    n_complex: ModeIndexDataArray = Field(
        ...,
        title="Complex Effective Index",
        description="Complex effective propagation indices of the monitored modes at the single "
        "frequency at which the mode profiles are solved (``ModeTimeMonitor.freq_spec``). Useful "
        "for identifying which amplitudes correspond to which physical modes.",
    )

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> ModeTimeData:
        """Time-domain amplitudes do not require frequency normalization."""
        return self
