from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flex_em.numerical.raw import source_normalization as source_normalization_numerics
from pydantic import Field

from tidy3d.components.data.data_array import (
    FluxDataArray,
    FluxTimeDataArray,
)
from tidy3d.components.monitor import (
    FluxMonitor,
    FluxTimeMonitor,
)

from .base import MonitorData

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.source.current import CustomCurrentSource, PointDipole

__all__ = ["FluxData", "FluxTimeData"]


class FluxData(MonitorData):
    """
    Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        We can access the data for each monitor by indexing into the :class:`SimulationData` with the monitor
        ``.name``. For the flux monitor data, we can access the raw flux data as a function of frequency with
        ``.flux``. As most data are multidimensional, it’s often very helpful to print out the data and directly
        inspect its structure.

    Example
    -------
    >>> from tidy3d import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)

    See Also
    --------

    **Notebooks:**
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
    """

    monitor: FluxMonitor = Field(
        title="Monitor",
        description="Frequency-domain flux monitor associated with the data.",
    )

    flux: FluxDataArray = Field(
        title="Flux",
        description="Flux values in the frequency-domain.",
    )

    def _make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> list[CustomCurrentSource | PointDipole]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        # avoids error in edge case where there are extraneous flux monitors not used in objective
        if np.all(self.flux.values == 0.0):
            return []

        raise NotImplementedError(
            "Could not formulate adjoint source for 'FluxMonitor' output. To compute derivatives "
            "with respect to flux data, hidden field data must be stored during the autograd "
            "forward run. Set 'FluxMonitor.enable_adjoint=True' and rerun the forward "
            "simulation, or use a 'FieldMonitor' and call '.flux' on the resulting "
            "'FieldData' object. See "
            "https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/"
            "tidy3d.FluxMonitor.html."
        )

    def normalize(self, source_spectrum_fn: Callable[[DataArray], NDArray]) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        new_flux = source_normalization_numerics.normalize_flux(self.flux, source_freq_amps)
        return self.copy(deep=False, update={"flux": new_flux})


class FluxTimeData(MonitorData):
    """
    Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import FluxTimeDataArray
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxTimeMonitor = Field(
        title="Monitor",
        description="Time-domain flux monitor associated with the data.",
    )

    flux: FluxTimeDataArray = Field(
        title="Flux",
        description="Flux values in the time-domain.",
    )
