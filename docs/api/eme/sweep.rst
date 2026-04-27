.. currentmodule:: tidy3d

Propagation Sweeps
--------------------

An EME sweep reuses previously computed mode data to evaluate the device at
multiple parameter points without re-solving the modes.  The sweep axis is
exposed as a ``sweep_index`` coordinate on the resulting S-matrix.  Broadband
frequency studies do not use a sweep type — list the desired frequencies in
:class:`.EMESimulation` ``freqs`` directly and control interpolation with
:class:`.EMEModeSpec` ``interp_spec``.

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
