.. currentmodule:: tidy3d

Propagation Sweeps
--------------------

Sweeps reuse previously computed EME mode data, so bent cells remain subject to the same
mode-consistency checks as the base simulation. In particular, bent anisotropic media with
``bend_medium_frame="global"`` are rejected when repetition or length changes would make a
reused mode see a different local tensor orientation, and bent custom media are only
supported when no global-frame remapping of custom data is required. If you instead resolve
global-frame anisotropic bends with multiple cells, check convergence with respect to the
number of EME cells.

.. warning::

   ``EMEFreqSweep`` is deprecated. For new simulations, list the desired frequencies in
   :class:`.EMESimulation` ``freqs`` and control the performance/accuracy tradeoff with
   :class:`.EMEModeSpec` ``interp_spec``.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMELengthSweep
   EMEModeSweep
   EMEPeriodicitySweep
   EMEFreqSweep
