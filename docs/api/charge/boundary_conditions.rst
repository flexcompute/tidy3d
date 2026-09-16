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

.. note::
   A :class:`VoltageBC` whose faces all lie on an insulator is a gate: it
   takes its potential reference from the metal work function :math:`W`.
   Place it on the gate metal's
   :class:`StructureBoundary`, or on the
   :class:`StructureStructureInterface` between the metal and the insulator,
   with ``work_function`` set on the metal's :class:`ChargeConductorMedium`;
   the contacted face must adjoin a metal carrying that property.
   Because the gate potential is referenced to the vacuum level,
   every :class:`SemiconductorMedium` must then carry ``electron_affinity``.
   Reported potentials are measured against the electron affinity
   :math:`\chi_\mathrm{ref}` of the reference semiconductor, so a monitor
   reads :math:`V - (W - \chi_\mathrm{ref})` on the gate face. The reference
   semiconductor is the background ``medium`` when it is a
   :class:`SemiconductorMedium`, and otherwise the first semiconductor
   structure in priority order (for the default priority mode, the first one
   listed in ``structures``); in a device with one semiconductor material
   :math:`\chi_\mathrm{ref}` is simply its :math:`\chi`.
   Any :class:`VoltageBC` with a face on an insulator -- a gate, or an ohmic
   or Schottky contact whose metal is clad by an insulator -- is supported
   only by the accelerated charge solver (``use_accelerated_solver=True``),
   because only it references such a face to the metal work function.

   Touching contact faces must prescribe the same electric potential at their
   shared points and have compatible small-signal drives. Separate contact
   boundaries to bias those terminals independently.

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
