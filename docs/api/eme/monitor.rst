.. currentmodule:: tidy3d

Monitors
----------

EME simulations always produce a device scattering matrix; monitors record
additional diagnostic data alongside it.  :class:`.EMEModeSolverMonitor`
returns the eigenmodes at each cell, :class:`.EMEFieldMonitor` returns the
propagated E and H fields on a plane, and :class:`.EMECoefficientMonitor`
returns the forward / backward modal coefficients per cell.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMECoefficientMonitor
   EMEModeSolverMonitor
   EMEFieldMonitor
   EMEMonitor
