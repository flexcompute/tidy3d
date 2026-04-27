.. currentmodule:: tidy3d

Simulation
-----------

An :class:`.EMESimulation` is the top-level object for EME: it defines the
simulation domain, the EME cells along the propagation axis, and the
frequencies of interest.  It is excitation-free — the solver always produces
the full bidirectional scattering matrix between the two ports.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMESimulation
