"""Objects that define how data is recorded from simulation."""
from abc import ABC
from typing import List, Union

import pydantic

from .types import Literal, Ax, Direction, FieldType, EMField, Array
from .geometry import Box
from .validators import assert_plane, validate_name_str
from .mode import ModeSpec
from .viz import add_ax_if_none, MonitorParams
from ..log import SetupError, ValidationError


class Monitor(Box, ABC):
    """Abstract base class for monitors."""

    name: str

    _name_validator = validate_name_str()

    @add_ax_if_none
    def plot(  # pylint:disable=duplicate-code
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
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.  # pylint: disable=line-too-long

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

    @pydantic.validator("freqs", always=True)
    def freqs_nonempty(cls, val):
        """Ensure freqs has at least one element"""
        if len(val) == 0:
            raise ValidationError("Monitor 'freqs' should have at least one element.")
        return val


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
    >>> monitor = FieldMonitor(
    ...     size=(2,2,2),
    ...     freqs=[200e12, 210e12],
    ...     fields=['Ex', 'Ey', 'Hz'],
    ...     name='freq_domain_fields')
    """

    fields: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    type: Literal["FieldMonitor"] = "FieldMonitor"
    data_type: Literal["ScalarFieldData"] = "ScalarFieldData"

    def surfaces(self):
        """Returns a list of 6 monitors corresponding to each surface of the box monitor.

        Returns
        -------
        List[td.FieldMonitor]
            List of 6 surface monitors for each side of the box monitor.
        """

        if any(s == 0.0 for s in self.size):
            raise ValidationError("Not applicable for the given monitor because it has zero volume.")

        self_bmin, self_bmax = self.bounds
        center_x, center_y, center_z = self.center
        size_x, size_y, size_z = self.size

        # Set up geometry data and names for each surface:

        surface_centers = (
            (self_bmin[0], center_y, center_z), # x-
            (self_bmax[0], center_y, center_z), # x+
            (center_x, self_bmin[1], center_z), # y-
            (center_x, self_bmax[1], center_z), # y+
            (center_x, center_y, self_bmin[2]), # z-
            (center_x, center_y, self_bmax[2])) # z+

        surface_sizes = (
            (0.0, size_y, size_z), # x-
            (0.0, size_y, size_z), # x+
            (size_x, 0.0, size_z), # y-
            (size_x, 0.0, size_z), # y+
            (size_x, size_y, 0.0), # z-
            (size_x, size_y, 0.0)) # z+

        surface_names = (
            self.name + '_x-',
            self.name + '_x+',
            self.name + '_y-',
            self.name + '_y+',
            self.name + '_z-',
            self.name + '_z+')

        # Create "surface" monitors
        monitors = []
        for c, s, n in zip(surface_centers, surface_sizes, surface_names):
            monitors.append(FieldMonitor(
                fields=self.fields, 
                center=c,
                size=s,
                freqs=self.freqs,
                name=n,
                type=self.type,
                data_type=self.data_type))

        return monitors


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
    >>> monitor = FieldTimeMonitor(
    ...     size=(2,2,2),
    ...     fields=['Hx'],
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='movie_monitor')
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
    >>> monitor = FluxTimeMonitor(
    ...     size=(2,2,0),
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='flux_time')
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
    >>> mode_spec = ModeSpec(num_modes=3)
    >>> monitor = ModeMonitor(
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')
    """

    direction: List[Direction] = ["+", "-"]
    mode_spec: ModeSpec
    type: Literal["ModeMonitor"] = "ModeMonitor"
    data_type: Literal["ModeData"] = "ModeData"


# types of monitors that are accepted by simulation
MonitorType = Union[FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor]
