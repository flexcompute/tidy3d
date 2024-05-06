"""EME monitors"""

from __future__ import annotations

import pydantic.v1 as pd

from typing import Union, List, Optional, Tuple, Literal

from abc import ABC, abstractmethod

from ..base_sim.monitor import AbstractMonitor
from ..monitor import AbstractFieldMonitor, ModeSolverMonitor


BYTES_COMPLEX = 8


class EMEMonitor(AbstractMonitor, ABC):
    """Abstract EME monitor."""

    freqs: Optional[List[pd.PositiveFloat]] = pd.Field(
        None,
        title="Monitor Frequencies",
        description="Frequencies at which the monitor will record. "
        "Must be a subset of the simulation 'freqs'.",
    )

    num_modes: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        title="Number of Modes",
        description="Maximum number of modes for the monitor to record. "
        "Cannot exceed the greatest number of modes in any EME cell.",
    )

    interval_space: Tuple[Literal[1], Literal[1], Literal[1]] = pd.Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. "
        "Not all monitors support values different from 1.",
    )

    eme_cell_interval_space: Literal[1] = pd.Field(
        1,
        title="EME Cell Interval",
        description="Number of eme cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    colocate: Literal[True] = pd.Field(
        True,
        title="Colocate Fields",
        description="Defines whether fields are colocated to grid cell boundaries (i.e. to the "
        "primal grid) on-the-fly during a solver run. Can be toggled for field recording monitors "
        "and is hard-coded for other monitors depending on their specific function.",
    )

    @abstractmethod
    def storage_size(
        self, num_cells: int, num_eme_cells: int, num_freqs: int, num_modes: int
    ) -> int:
        """Size of monitor storage given the number of points after discretization.

        Parameters
        ----------
        num_cells : int
            Number of grid cells within the monitor after discretization by a :class:`Simulation`.
        num_eme_cells: int
            Number of EME cells intersecting the monitor.
        num_freqs: int
            Number of frequencies in the monitor.
        num_modes: int
            Number of modes in the monitor.

        Returns
        -------
        int
            Number of bytes to be stored in monitor.
        """


class EMEModeSolverMonitor(EMEMonitor):
    """EME mode solver monitor.
    Records EME modes computed in planes intersecting the monitor geometry.

    Example
    -------
    >>> monitor = EMEModeSolverMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     name="eme_modes"
    ... )
    """

    interval_space: Tuple[Literal[1], Literal[1], Literal[1]] = pd.Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Note: not yet supported. Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. Note: the interval "
        "in the propagation direction is not used. Note: this is not yet supported.",
    )

    eme_cell_interval_space: pd.PositiveInt = pd.Field(
        1,
        title="EME Cell Interval",
        description="Number of eme cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    colocate: bool = pd.Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default (False) is used internally in EME propagation.",
    )

    def storage_size(
        self, num_cells: int, num_eme_cells: int, num_freqs: int, num_modes: int
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        bytes_single = 6 * BYTES_COMPLEX * num_cells * num_freqs * num_modes
        return bytes_single


class EMEFieldMonitor(EMEMonitor, AbstractFieldMonitor):
    """EME monitor for propagated field.

    Example
    -------
    >>> monitor = EMEFieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     name="eme_field"
    ... )
    """

    interval_space: Tuple[Literal[1], Literal[1], Literal[1]] = pd.Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Note: not yet supported. Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included.",
    )

    eme_cell_interval_space: Literal[1] = pd.Field(
        1,
        title="EME Cell Interval",
        description="Number of eme cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1. Note: this field is not used for "
        "EME field monitor.",
    )

    colocate: bool = pd.Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default (False) is used internally in EME propagation.",
    )

    num_modes: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        title="Number of Modes",
        description="Maximum number of modes for the monitor to record. "
        "For 'EMEFieldMonitor', refers to the number of modes at each port."
        "Cannot exceed the max of the number of modes in the two ports. ",
    )

    def storage_size(
        self, num_cells: int, num_eme_cells: int, num_freqs: int, num_modes: int
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        bytes_single = 6 * BYTES_COMPLEX * num_cells * num_freqs * num_modes * 2
        return bytes_single


class EMECoefficientMonitor(EMEMonitor):
    """EME monitor for mode coefficients.
    Records the amplitudes of the forward and backward modes in each cell
    intersecting the monitor geometry.

    Example
    -------
    >>> monitor = EMECoefficientMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     name="eme_coeffs"
    ... )
    """

    interval_space: Tuple[Literal[1], Literal[1], Literal[1]] = pd.Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. "
        "Not all monitors support values different from 1. Note: This field is not used "
        "for 'EMECoefficientMonitor'.",
    )

    eme_cell_interval_space: pd.PositiveInt = pd.Field(
        1,
        title="EME Cell Interval",
        description="Number of eme cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    def storage_size(
        self, num_cells: int, num_eme_cells: int, num_freqs: int, num_modes: int
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        bytes_single = 4 * BYTES_COMPLEX * num_freqs * num_modes * num_modes * num_eme_cells
        return bytes_single


EMEMonitorType = Union[
    EMEModeSolverMonitor, EMEFieldMonitor, EMECoefficientMonitor, ModeSolverMonitor
]
