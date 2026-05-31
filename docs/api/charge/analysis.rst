.. currentmodule:: tidy3d

Analysis
-----------

The analysis specification controls *what* a charge simulation computes: the
DC (and optional small-signal AC) sweep. Attach one of these to
``HeatChargeSimulation.analysis_spec``. The solver itself is selected separately
via ``HeatChargeSimulation.use_accelerated_solver``; convergence and tolerance
settings are covered under :ref:`charge-accelerated-solver`.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   SteadyChargeDCAnalysis
   IsothermalSteadyChargeDCAnalysis
   SSACAnalysis
   IsothermalSSACAnalysis

.. seealso::

   The voltage and current sources used by these analyses are documented on
   the :doc:`/api/spice` page.
