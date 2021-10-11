""" Objects that define how data is recorded from simulation """
from abc import ABC
from typing import List, Union

import numpy as np

from .types import Literal, Ax, Direction, FieldType, FloatArrayLike, IntArrayLike, EMField
from .geometry import Box
from .validators import assert_plane
from .mode import Mode
from .viz import add_ax_if_none, MonitorParams


""" Monitors """


class Monitor(Box, ABC):
    """base class for monitors, which all have Box shape"""

    @add_ax_if_none
    def plot(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """plot monitor geometry"""
        kwargs = MonitorParams().update_params(**kwargs)
        ax = self.geometry.plot(x=x, y=y, z=z, ax=ax, **kwargs)
        return ax


""" The following are abstract classes that separate the ``Monitor`` instances into different
    types depending on what they store. 
    They can be useful for keeping argument types and validations separated.
    For example, monitors that should always be defined on planar geometries can have an 
    ``_assert_plane()`` validation in the abstract base class ``PlanarMonitor``.
    This way, ``_assert_plane()`` will always be used if we add more ``PlanarMonitor`` objects in
    the future.
    This organization is also useful when doing conditions based on monitor / data type.
    For example, instead of 
    ``if isinstance(mon_data, (FieldMonitor, FieldTimeMonitor)):`` we can simply do 
    ``if isinstance(mon_data, ScalarFieldMonitor)`` and this will generalize if we add more
    ``ScalarFieldMonitor`` objects in the future.
"""


class FreqMonitor(Monitor, ABC):
    """stores data in frequency domain"""

    freqs: FloatArrayLike


class TimeMonitor(Monitor, ABC):
    """stores data in time domain"""

    times: IntArrayLike


class ScalarFieldMonitor(Monitor, ABC):
    """stores data as a function of x,y,z"""

    fields: List[EMField] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


class PlanarMonitor(Monitor, ABC):
    """stores quantities on a plane"""

    _plane_validator = assert_plane()


class AbstractFluxMonitor(PlanarMonitor, ABC):
    """stores flux through a plane"""


""" usable """


class FieldMonitor(ScalarFieldMonitor, FreqMonitor):
    """Stores EM fields or permittivity as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    fields: List[str], optional
        Electromagnetic field(s) to measure (E, H), defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``, also accepts diagonal components of permittivity tensor as ``'eps_xx', 'eps_yy',
        'eps_zz'``.
    freqs: List[float]
        Frequencies to measure fields at at.
    """

    fields: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    type: Literal["FieldMonitor"] = "FieldMonitor"


class FieldTimeMonitor(ScalarFieldMonitor, TimeMonitor):
    """Stores EM fields as a function of time.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    fields: List[str], optional
        Electromagnetic field(s) to measure (E, H), defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``.
    times: List[int]
        Time steps to measure the fields at.
    """

    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"


class FluxMonitor(AbstractFluxMonitor, FreqMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    freqs: List[float]
        Frequencies to measure flux at.
    """

    type: Literal["FluxMonitor"] = "FluxMonitor"


class FluxTimeMonitor(AbstractFluxMonitor, TimeMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    times: List[int]
        Time steps to measure flux at.
    """

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"


class ModeMonitor(PlanarMonitor, FreqMonitor):
    """stores overlap amplitudes associated with modes.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    freqs: List[float]
        Frequencies to measure flux at.
    modes: List[``Mode``]
        List of ``Mode`` objects specifying the modal profiles to measure amplitude overalap with.
    """

    direction: List[Direction] = ["+", "-"]
    modes: List[Mode]
    type: Literal["ModeMonitor"] = "ModeMonitor"


""" explanation of monitor_type_map:
    When we load monitor data from file, we need some way to know what type of ``Monitor`` created 
    the data.
    The ``Monitor``'s' ``type`` itself is not serilizable, so we can't store that directly in json.
    However, the ``Montior.type`` attribute stores a string representation of the ``MonitorType``, 
    so we can use that.
    This map allows one to recover the ``Monitor`` type from the ``.type`` attribute in the json 
    object and therefore load the correct monitor.
"""

monitor_type_map = {
    "FieldMonitor": FieldMonitor,
    "FieldTimeMonitor": FieldTimeMonitor,
    "FluxMonitor": FluxMonitor,
    "FluxTimeMonitor": FluxTimeMonitor,
    "ModeMonitor": ModeMonitor,
}

MonitorType = Union[tuple(monitor_type_map.values())]


""" Convenience methods to create evenly spaced times or frequencies
    note: later on, might want these to hold (start, stop, step) or equivalent and be stored in
    `freqs` and `times` instead of List to reduce clutter in json.
"""


def _uniform_arange(start: int, stop: int, step: int) -> IntArrayLike:
    """uniform spacing from start to stop with spacing of step."""
    assert start <= stop, "start must not be greater than stop"
    return list(np.arange(start, stop, step))


def _uniform_linspace(start: float, stop: float, num: int) -> FloatArrayLike:
    """uniform spacing from start to stop with num elements."""
    assert start <= stop, "start must not be greater than stop"
    return list(np.linspace(start, stop, num))


def uniform_times(t_start: int, t_stop: int, t_step: int = 1) -> IntArrayLike:
    """create times at evenly spaced steps."""
    assert isinstance(t_start, int), "`t_start` must be integer for time sampler"
    assert isinstance(t_stop, int), "`t_stop` must be integer for time sampler"
    assert isinstance(t_step, int), "`t_step` must be integer for time sampler"
    times = _uniform_arange(t_start, t_stop, t_step)
    return times


def uniform_freqs(f_start: float, f_stop: float, num_freqs: int) -> FloatArrayLike:
    """create frequencies at evenly spaced points."""
    freqs = _uniform_linspace(f_start, f_stop, num_freqs)
    return freqs
