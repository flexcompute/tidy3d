from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd.source_factory import gaussian_source_from_monitor
from tidy3d.components.base import TYPE_TAG_STR
from tidy3d.components.data.data_array import ModeAmpsDataArray
from tidy3d.components.data.monitor_data._utils import _make_adjoint_sources_from_modal_amps
from tidy3d.components.data.monitor_data.field.electromagnetic import ElectromagneticFieldData
from tidy3d.components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    GaussianOverlapMonitor,
    ThinLensOverlapMonitor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.source.base import Source


class AbstractOverlapData(ElectromagneticFieldData):
    amps: ModeAmpsDataArray = Field(
        title="Amplitudes",
        description="Complex-valued amplitudes of the overlap decomposition.",
    )

    def normalize(self, source_spectrum_fn: Callable) -> AbstractOverlapData:
        """Return copy of self after normalization is applied using source spectrum function."""
        if self.amps is None:
            return self.copy()
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        new_amps = (self.amps / source_freq_amps).astype(self.amps.dtype)
        return self.copy(deep=False, update={"amps": new_amps})

    @property
    def time_reversed_copy(self) -> AbstractOverlapData:
        """Make a copy of the data with direction-reversed fields. In lossy or gyrotropic systems,
        the time-reversed fields will not be the same as the backward-propagating modes.
        Note: this only reverses the store fields, any other stored quantities are untouched.
        """
        mnt = self.monitor

        if not mnt.store_fields_direction:
            return self.copy()

        # Time reversal
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)

        # switch direction in the monitor
        new_dir = "+" if mnt.store_fields_direction == "-" else "-"
        update_dict = {"store_fields_direction": new_dir}
        if hasattr(mnt, "direction"):
            update_dict["direction"] = new_dir
        new_data["monitor"] = mnt.updated_copy(**update_dict)
        return self.copy(deep=False, update=new_data)


class FieldOverlapData(AbstractOverlapData):
    monitor: GaussianOverlapMonitor | AstigmaticGaussianOverlapMonitor | ThinLensOverlapMonitor = (
        Field(
            discriminator=TYPE_TAG_STR,
            title="Monitor",
            description="Monitor associated with the data.",
        )
    )

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Get all adjoint sources for ``FieldOverlapData``."""
        adjoint_sources = []

        for name in dataset_names:
            if name == "amps":
                adjoint_sources += self._make_adjoint_sources_amps(fwidth=fwidth)
            else:
                raise NotImplementedError(
                    f"Unsupported adjoint field '{name}' for 'FieldOverlapData' "
                    f"on monitor '{self.monitor.name}'. Only 'amps' is supported."
                )

        return adjoint_sources

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[Source]:
        """Generate adjoint sources for ``FieldOverlapData.amps``."""

        def source_from_amp(
            freq: float, direction: str, _mode_index: int, coefficient: complex
        ) -> Source:
            return gaussian_source_from_monitor(
                monitor=self.monitor,
                freq=freq,
                direction=direction,
                coefficient=coefficient,
                fwidth=fwidth,
            )

        return _make_adjoint_sources_from_modal_amps(
            self.amps,
            source_from_amp,
            skip_nan=True,
        )

    def _adjoint_source_amp(self, amp: DataArray, fwidth: float) -> Source:
        """Generate an adjoint Gaussian-like source for a single overlap amplitude."""
        coords = amp.coords
        freq0 = coords["f"]
        direction = coords["direction"]

        amp_complex = self.get_amplitude(amp)

        return gaussian_source_from_monitor(
            monitor=self.monitor,
            freq=float(freq0),
            direction=direction,
            coefficient=amp_complex,
            fwidth=fwidth,
        )
