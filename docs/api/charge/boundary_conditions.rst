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

.. note::
   Schottky contacts are opt-in on :class:`VoltageBC` via
   ``model="schottky_mott"``. The Schottky-Mott barrier
   :math:`\phi_{Bn} = W - \chi`, :math:`\phi_{Bp} = E_g - \phi_{Bn}` is
   built from per-medium material properties: ``work_function`` on the
   adjacent :class:`ChargeConductorMedium`, and ``electron_affinity``,
   ``richardson_electron``, ``richardson_hole`` on the adjacent
   :class:`SemiconductorMedium`. The default ``model="ohmic"`` is
   the standard ohmic contact. Schottky contacts are supported only by
   the accelerated charge solver (``use_accelerated_solver=True``) and
   compose with DC sweeps, small-signal AC analyses, and Fermi-Dirac
   carrier statistics (``fermi_dirac=True``).

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