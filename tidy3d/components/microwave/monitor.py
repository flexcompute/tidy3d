"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

import pydantic.v1 as pydantic

from tidy3d.components.microwave.base import MicrowaveBaseModel
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.monitor import ModeMonitor, ModeSolverMonitor


class MicrowaveModeMonitorBase(MicrowaveBaseModel):
    """Base class for microwave mode monitors that use :class:`.MicrowaveModeSpec`.

    This mixin provides the ``mode_spec`` field configured for RF and microwave applications,
    including characteristic impedance calculations and transmission line analysis.

    Notes
    -----
    This is a mixin class that provides the :class:`.MicrowaveModeSpec` field for mode monitors.
    It must be placed first in the inheritance list to ensure its ``mode_spec`` field takes
    precedence over the base :class:`.ModeSpec` field from :class:`.AbstractModeMonitor`.
    """

    mode_spec: MicrowaveModeSpec = pydantic.Field(
        default_factory=MicrowaveModeSpec._default_without_license_warning,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )


class MicrowaveModeMonitor(MicrowaveModeMonitorBase, ModeMonitor):
    """:class:`Monitor` that records amplitudes from modal decomposition of fields on plane.

    Notes
    ------

        The fields recorded by frequency monitors (and hence also mode monitors) are automatically
        normalized by the power amplitude spectrum of the source. For multiple sources, the user can
        select which source to use for the normalization too.

        We can also use the mode amplitudes recorded in the mode monitor to reveal the decomposition
        of the radiated power into forward- and backward-propagating modes, respectively.

        .. TODO give an example of how to extract the data from this mode.

        .. TODO add derivation in the notebook.

        .. TODO add link to method

        .. TODO add links to notebooks correspondingly

    Example
    -------
    >>> mode_spec = MicrowaveModeSpec(num_modes=3)
    >>> monitor = MicrowaveModeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')

    See Also
    --------

    **Notebooks**:
        * `ModalSourcesMonitors <../../notebooks/ModalSourcesMonitors.html>`_
    """


class MicrowaveModeSolverMonitor(MicrowaveModeMonitorBase, ModeSolverMonitor):
    """:class:`Monitor` that stores the mode field profiles returned by the mode solver in the
    monitor plane.

    Example
    -------
    >>> mode_spec = MicrowaveModeSpec(num_modes=3)
    >>> monitor = MicrowaveModeSolverMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')
    """
