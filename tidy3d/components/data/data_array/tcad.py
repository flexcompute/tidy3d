"""TCAD results, plus the heat and charge lookup tables used by parameter perturbations."""

from __future__ import annotations

import numpy as np  # noqa: F401 - used by doctest examples

from .abstract_spatial import AbstractSpatialDataArray
from .base import DataArray


class HeatDataArray(DataArray):
    """Heat data array.

    Example
    -------
    >>> T = [0, 1e-12, 2e-12]
    >>> td = HeatDataArray((1+1j) * np.random.random((3,)), coords=dict(T=T))
    """

    __slots__ = ()
    _dims = ("T",)


class ChargeDataArray(DataArray):
    """Charge data array.

    Example
    -------
    >>> n = [0, 1e-12, 2e-12]
    >>> p = [0, 3e-12, 4e-12]
    >>> td = ChargeDataArray((1+1j) * np.random.random((3,3)), coords=dict(n=n, p=p))
    """

    __slots__ = ()
    _dims = ("n", "p")


class SteadyVoltageDataArray(DataArray):
    """Steady voltage data array. Data array used with steady state
    simulations with voltage as dimension.

    Example
    -------
    >>> import tidy3d as td
    >>> intensities = [0, 1, 4]
    >>> V = [-1, -0.5, 0]
    >>> voltage_dataarray = td.SteadyVoltageDataArray(data=intensities, coords={"v": V})
    """

    __slots__ = ()
    _dims = ("v",)


class FreqVoltageDataArray(DataArray):
    """Frequency-domain array.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> v = [0.1, 0.2, 0.3]
    >>> coords = dict(f=f, v=v)
    >>> fd = FreqVoltageDataArray((1+1j) * np.random.random((2, 3)), coords=coords)
    """

    __slots__ = ()
    _dims = (
        "f",
        "v",
    )


class ConvergenceHistoryDataArray(DataArray):
    """Per-iteration residual trace across a voltage sweep and solution components.

    Dimensions:

    * ``v`` -- sweep bias [V]
    * ``pseudo_step`` -- Newton iteration index (0-based); ragged biases are
      NaN-padded on this axis to the longest trace in the sweep.
    * ``component`` -- solution component label (e.g. ``"potential"``,
      ``"electrons"``, ``"holes"``, ``"temperature"``).

    Values are residual norms in the native units of the corresponding solution
    variable (volts for ``"potential"``; um^-3 for the carrier concentrations).
    NaN at ``(v, k, c)`` means bias ``v`` did not reach iteration ``k`` for
    component ``c``.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> V = [-1.0, 0.0]
    >>> steps = [0, 1, 2]
    >>> comps = ["potential"]
    >>> data = np.array([[[3.2e-1], [1.7e-1], [1.1e-8]],
    ...                  [[2.1e-1], [9.0e-2], [np.nan]]])
    >>> history = td.ConvergenceHistoryDataArray(
    ...     data=data, coords={"v": V, "pseudo_step": steps, "component": comps}
    ... )
    """

    __slots__ = ()
    _dims = ("v", "pseudo_step", "component")


class CFLHistoryDataArray(DataArray):
    """Per-iteration CFL trace across a voltage sweep.

    Dimensions:

    * ``v`` -- sweep bias [V]
    * ``pseudo_step`` -- Newton iteration index (0-based); ragged biases are
      NaN-padded on this axis to the longest trace in the sweep, matching
      ``ConvergenceHistoryDataArray`` so the two share the axis.

    Values are the dimensionless CFL number in force at that iteration. The
    adaptive controller keeps it within ``[ChargeToleranceSpec.cfl_min,
    ChargeToleranceSpec.cfl_number]`` and lowers it for several reasons, so a
    value at a bound says where the controller ended, not why. The exception is
    the first bias solved, which starts at ``1`` before any bound is applied, so
    its first sample can lie outside the range on either side. Every later bias
    starts from the CFL carried over from the previous one, brought within range
    and lowered after a bias that had to back off. A drop between consecutive
    samples means the controller backed off; only one sample is recorded per
    ``pseudo_step``.

    Read it alongside ``residual_history`` at a bias that did not converge to
    see whether the controller ended at its lower bound, at its upper bound,
    or was still adapting between them. With ``cfl_min == cfl_number`` the CFL
    is constant and the trace says nothing about the bounds.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> V = [0.0, 0.7]
    >>> steps = [0, 1, 2]
    >>> data = np.array([[1.0, 2.0, 4.0],
    ...                  [1.0, 0.5, np.nan]])
    >>> history = td.CFLHistoryDataArray(
    ...     data=data, coords={"v": V, "pseudo_step": steps}
    ... )
    """

    __slots__ = ()
    _dims = ("v", "pseudo_step")


class SpatialVoltageDataArray(AbstractSpatialDataArray):
    """Spatial distribution with voltage mapping.

    Example
    -------
    >>> x = [1,2]
    >>> y = [2,3,4]
    >>> z = [3,4,5,6]
    >>> v = [-1, 1]
    >>> coords = dict(x=x, y=y, z=z, voltage=v)
    >>> fd = SpatialVoltageDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    """

    __slots__ = ()
    _dims = ("x", "y", "z", "voltage")


class PerturbationCoefficientDataArray(DataArray):
    __slots__ = ()
    _dims = ("wvl", "coeff")
