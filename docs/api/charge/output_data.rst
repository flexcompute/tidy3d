.. currentmodule:: tidy3d

Output Data
-------------

.. important::
   **2D vs 3D simulations and units.**
   Charge simulations can be 2D (one ``size`` component is 0) or 3D.
   In 2D mode, the solver treats the geometry as a cross-section with
   infinite extrusion depth. Extensive output quantities (current,
   capacitance, resistance) are therefore reported **per unit length**
   (:math:`\mu\text{m}`). For example:

   - Current: :math:`\text{A}` (3D) :math:`\rightarrow` :math:`\text{A}/\mu\text{m}` (2D)
   - Capacitance: :math:`\text{fF}` (3D) :math:`\rightarrow` :math:`\text{fF}/\mu\text{m}` (2D)
   - Resistance: :math:`\Omega` (3D) :math:`\rightarrow` :math:`\Omega \cdot \mu\text{m}` (2D)

   Intensive quantities (potential, carrier concentration, electric
   field, energy bands) keep the same units regardless of
   dimensionality.

   To convert a 2D result to a total device quantity, multiply by
   the physical device depth in micrometers.


Simulation Data
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.HeatChargeSimulationData


Monitor Data
^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.SteadyPotentialData
   tidy3d.SteadyFreeCarrierData
   tidy3d.SteadyEnergyBandData
   tidy3d.SteadyCapacitanceData
   tidy3d.SteadyElectricFieldData
   tidy3d.SteadyCurrentDensityData

Device Data
^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.DeviceCharacteristics
