.. currentmodule:: tidy3d

Monitors
----------

Monitors record steady-state charge quantities over a region: the electrostatic
potential, free-carrier concentrations, current density, electric field,
energy bands, generation-recombination rates, and per-equation residuals. A
charge simulation requires at least one of ``SteadyPotentialMonitor``,
``SteadyFreeCarrierMonitor``, ``SteadyCapacitanceMonitor``,
``SteadyCurrentDensityMonitor``, ``SteadyChargeResidualMonitor`` or
``SteadyGenerationRecombinationMonitor``; ``SteadyElectricFieldMonitor`` and
``SteadyEnergyBandMonitor`` are additional outputs and do not satisfy this
requirement on their own.

.. note::
   ``SteadyCapacitanceMonitor`` reports capacitance from a DC voltage sweep, so it
   requires a voltage source that sweeps an array of voltages
   (``len(voltage) > 1``).

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   SteadyPotentialMonitor
   SteadyFreeCarrierMonitor
   SteadyCapacitanceMonitor
   SteadyCurrentDensityMonitor
   SteadyElectricFieldMonitor
   SteadyEnergyBandMonitor
   SteadyChargeResidualMonitor
   SteadyGenerationRecombinationMonitor
   SelfHeatingMonitor
