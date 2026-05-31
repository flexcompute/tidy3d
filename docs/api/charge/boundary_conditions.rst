.. currentmodule:: tidy3d

Boundary Conditions
-----------------------------

A boundary condition pairs a *condition* (the physical constraint, e.g. an
applied voltage) with a *placement* (where on the geometry it is applied) via a
:class:`HeatChargeBoundarySpec`. Collect these in
``HeatChargeSimulation.boundary_spec``.

.. note::
   A charge simulation requires at least **two** ``VoltageBC`` boundaries. When a
   ``SteadyCapacitanceMonitor`` is present, one of the voltage sources must sweep
   an array of voltages (``len(voltage) > 1``) so the capacitance can be
   computed from the DC sweep.

Specifications
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   HeatBoundarySpec
   HeatChargeBoundarySpec


Types
^^^^^^^^^^^^^^^^^

The condition applied at the boundary. ``VoltageBC`` and ``CurrentBC`` draw
their excitation from the SPICE source classes documented on the
:doc:`/api/spice` page.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   VoltageBC
   CurrentBC
   InsulatingBC

Placement
^^^^^^^^^^^^^^^^^

Where a condition is applied: on an interface between structures or mediums, on
a structure's outer boundary, or on the simulation domain boundary.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   StructureStructureInterface
   StructureBoundary
   MediumMediumInterface
   StructureSimulationBoundary
   SimulationBoundary