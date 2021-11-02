"""Objects that define how data is recorded from simulation."""
from abc import ABC
from typing import List, Union

import pydantic

from .types import Literal, Ax, Direction, FieldType, EMField, Array
from .geometry import Box
from .validators import assert_plane, validate_name_str
from .mode import Mode
from .viz import add_ax_if_none, MonitorParams
from ..log import SetupError


class Monitor(Box, ABC):
    """Abstract base class for monitors."""

    name: str

    _name_validator = validate_name_str()

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot the monitor geometry on a cross section plane.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs = MonitorParams().update_params(**kwargs)
        ax = self.geometry.plot(x=x, y=y, z=z, ax=ax, **kwargs)
        return ax

    @property
    def geometry(self):
        """:class:`Box` representation of monitor.

        Returns
        -------
        :class:`Box`
            Representation of the monitor geometry as a :class:`Box`.
        """
        return Box(center=self.center, size=self.size)


class FreqMonitor(Monitor, ABC):
    """Stores data in the frequency-domain."""

    freqs: Union[List[float], Array[float]]


class TimeMonitor(Monitor, ABC):
    """Stores data in the time-domain."""

    start: pydantic.NonNegativeFloat = 0.0
    stop: pydantic.NonNegativeFloat = None
    interval: pydantic.PositiveInt = 1

    @pydantic.validator("stop", always=True)
    def stop_greater_than_start(cls, val, values):
        """Ensure sure stop is greater than or equal to start."""
        start = values.get("start")
        if val and val < start:
            raise SetupError("Monitor start time is greater than stop time.")
        return val


class AbstractFieldMonitor(Monitor, ABC):
    """Stores electromagnetic field data as a function of x,y,z."""

    fields: List[EMField] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


class PlanarMonitor(Monitor, ABC):
    """Monitors that must have planar geometry."""

    _plane_validator = assert_plane()


class AbstractFluxMonitor(PlanarMonitor, ABC):
    """stores flux through a plane"""


class FieldMonitor(AbstractFieldMonitor, FreqMonitor):
    """Stores a collection of electromagnetic fields in the frequency domain.

    Parameters
    ----------
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of monitor.
    size: Tuple[float, float, float]
        Size of monitor.
        All elements must be non-negative.
    fields: List[str] = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        Specifies the electromagnetic field components to record.
        If wanting to conserve data, can specify fewer components.
    freqs: List[float] or np.ndarray
        List of frequencies in Hertz to store fields at.
    name : str
        (Required) name used to access data after simulation is finished.

    Example
    -------
    >>> monitor = FieldMonitor(size=(2,2,2), freqs=[200e12, 210e12], fields=['Ex', 'Ey', 'Hz'], name='freq_domain_fields')
    """

    fields: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    type: Literal["FieldMonitor"] = "FieldMonitor"
    data_type: Literal["ScalarFieldData"] = "ScalarFieldData"


class FieldTimeMonitor(AbstractFieldMonitor, TimeMonitor):
    """Stores a collection of electromagnetic fields in the time domain.

    Parameters
    ----------
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of monitor.
    size: Tuple[float, float, float]
        Size of monitor.
        All elements must be non-negative.
    fields: List[str] = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        Specifies the electromagnetic field components to record.
        If wanting to conserve data, can specify fewer components.
    start : float = 0.0
        Time (seconds) to start recording fields.
    stop : float = None
        Time (seconds) to stop recording fields.
        Must be greater than or equal to ``start``.
        If not specified, records until the end of the simulation.
    interval : int = 1
        Number of time steps between measurements.
        To conserve data, intervals > 1 may be specified to record data more sparsely sampled data.
        Must be positive.
    name : str
        (Required) name used to access data after simulation is finished.

    Example
    -------
    >>> monitor = FieldTimeMonitor(size=(2,2,2), fields=['Hx'], start=1e-13, stop=5e-13, interval=2, name='movie_monitor')
    """

    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"
    data_type: Literal["ScalarFieldTimeData"] = "ScalarFieldTimeData"


class FluxMonitor(AbstractFluxMonitor, FreqMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of monitor.
    size: Tuple[float, float, float]
        Size of monitor.
        All elements must be non-negative.
        One element must be 0.0 to define flux plane.
    freqs: List[float] or np.ndarray
        List of frequencies in Hertz to store fields at.
    name : str
        (Required) name used to access data after simulation is finished.

    Example
    -------
    >>> monitor = FluxMonitor(size=(2,2,0), freqs=[200e12, 210e12], name='flux_monitor')
    """

    type: Literal["FluxMonitor"] = "FluxMonitor"
    data_type: Literal["FluxData"] = "FluxData"


class FluxTimeMonitor(AbstractFluxMonitor, TimeMonitor):
    """Stores power flux through a plane as a function of time.

    Parameters
    ----------
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of monitor.
    size: Tuple[float, float, float]
        Size of monitor.
        All elements must be non-negative.
        One element must be 0.0 to define flux plane.
    start : float = 0.0
        Time (seconds) to start recording fields.
    stop : float = None
        Time (seconds) to stop recording fields.
        Must be greater than or equal to ``start``.
        If not specified, records until the end of the simulation.
    interval : int = 1
        Number of time steps between measurements.
        To conserve data, intervals > 1 may be specified to record data more sparsely sampled data.
        Must be positive.
    name : str
        (Required) name used to access data after simulation is finished.

    Example
    -------
    >>> monitor = FluxTimeMonitor(size=(2,2,0), start=1e-13, stop=5e-13, interval=2, name='flux_time')
    """

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"
    data_type: Literal["FluxTimeData"] = "FluxTimeData"


class ModeMonitor(PlanarMonitor, FreqMonitor):
    """Stores amplitudes found through modal decomposition of fields on plane.

    Parameters
    ----------
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        Center of monitor.
    size: Tuple[float, float, float]
        Size of monitor.
        All elements must be non-negative.
        One element must be 0.0 to define mode plane.
    freqs: List[float] or np.ndarray
        List of frequencies in Hertz to compute the modal decomposition on.
    modes : List[:class:`Mode`]
        List of mode specifications to compute modal overlaps with.
    name : str
        (Required) name used to access data after simulation is finished.

    Example
    -------
    >>> modes = [Mode(mode_index=0), Mode(mode_index=1)]
    >>> monitor = ModeMonitor(size=(2,2,0), freqs=[200e12, 210e12], modes=modes, name='mode_monitor')
    """

    direction: List[Direction] = ["+", "-"]
    modes: List[Mode]
    type: Literal["ModeMonitor"] = "ModeMonitor"
    data_type: Literal["ModeData"] = "ModeData"


# types of monitors that are accepted by simulation
MonitorType = Union[FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor]
