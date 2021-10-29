""" Objects that define how data is recorded from simulation """
from abc import ABC
from typing import List, Union

import pydantic

from .types import Literal, Ax, Direction, FieldType, EMField, Array
from .geometry import Box
from .validators import assert_plane
from .mode import Mode
from .viz import add_ax_if_none, MonitorParams
from ..log import SetupError

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

    freqs: Union[List[float], Array[float]]


class TimeMonitor(Monitor, ABC):
    """stores data in time domain"""

    start: pydantic.NonNegativeFloat = 0.0
    stop: pydantic.NonNegativeFloat = None
    interval: pydantic.PositiveInt = 1

    @pydantic.validator("stop", always=True)
    def stop_greater_than_start(cls, val, values):
        """make sure stop is greater than or equal to start"""
        start = values.get("start")
        if val and val < start:
            raise SetupError("Monitor start time is greater than stop time.")
        return val


class AbstractFieldMonitor(Monitor, ABC):
    """stores data as a function of x,y,z"""

    fields: List[EMField] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


class PlanarMonitor(Monitor, ABC):
    """stores quantities on a plane"""

    _plane_validator = assert_plane()


class AbstractFluxMonitor(PlanarMonitor, ABC):
    """stores flux through a plane"""


""" usable """


class FieldMonitor(AbstractFieldMonitor, FreqMonitor):
    """Stores EM fields or permittivity as a function of frequency.

    Parameters
    ----------
    center: ``(float, float, float)``, optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: ``(float, float, float)``
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    fields: ``[str]``, optional
        Electromagnetic field(s) to measure (E, H), defaults to ``['Ex', 'Ey', 'Ez', 'Hx', 'Hy',
        'Hz']``.
    freqs: ``[float]``
        Frequencies to measure fields at at (Hz),

    """

    fields: List[FieldType] = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    type: Literal["FieldMonitor"] = "FieldMonitor"
    data_type: Literal["ScalarFieldData"] = "ScalarFieldData"


class FieldTimeMonitor(AbstractFieldMonitor, TimeMonitor):
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
    start: ``float = 0.0``
        (seconds) Time to start monitoring fields.
    stop: ``float = None``
        (seconds) Time to stop monitoring fields, end of simulation if not specified.
    interval: ``int = 1``
        Records data at every ``interval`` time steps in the simulation.
    """

    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"
    data_type: Literal["ScalarFieldTimeData"] = "ScalarFieldTimeData"


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
    data_type: Literal["FluxData"] = "FluxData"


class FluxTimeMonitor(AbstractFluxMonitor, TimeMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float].
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    start: ``float = 0.0``
        (seconds) Time to start monitoring flux.
    stop: ``float = None``
        (seconds) Time to stop monitoring flux, end of simulation if not specified.
    interval: ``int = 1``
        Records data at every ``interval`` time steps in the simulation.
    """

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"
    data_type: Literal["FluxTimeData"] = "FluxTimeData"


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
    data_type: Literal["ModeData"] = "ModeData"


""" explanation of monitor_type_map:
    When we load monitor data from file, we need some way to know what type of ``Monitor`` created 
    the data.
    The ``Monitor``'s' ``type`` itself is not serilizable, so we can't store that directly in json.
    However, the ``Montior.type`` attribute stores a string representation of the ``MonitorType``, 
    so we can use that.
    This map allows one to recover the ``Monitor`` type from the ``.type`` attribute in the json 
    object and therefore load the correct monitor.
"""

MonitorType = Union[FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor, ModeMonitor]
