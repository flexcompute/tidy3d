.. currentmodule:: tidy3d

Heterojunctions
---------------

An interface where two different :class:`SemiconductorMedium` materials touch
is a heterojunction. Charge simulations detect such interfaces automatically:
whenever the two sides differ in band structure (band gap, effective densities
of states, band-gap narrowing, or electron affinity), the heterojunction
treatment described here is applied **by default** — no extra configuration is
required. Interfaces between identical semiconductor media remain ordinary
continuous interfaces.

.. note::

   Heterojunctions that intersect an electrical contact are not currently supported.

Band alignment
^^^^^^^^^^^^^^

Band offsets follow Anderson's rule from the configured electron affinities
:math:`\chi` (:attr:`SemiconductorMedium.electron_affinity`) and band gaps
:math:`E_g`:

.. math::

   \Delta E_c = \chi_1 - \chi_2, \qquad
   \Delta E_v = \Delta E_c - E_{g,2} + E_{g,1}.

The electrostatic potential is continuous across the interface; the band
offsets appear as steps in the conduction and valence band edges, visible in
:class:`SteadyEnergyBandMonitor` data. An unset ``electron_affinity`` is
treated as 0 eV, so physical alignments require setting the affinity on every
semiconductor material of the device.

Interface transport
^^^^^^^^^^^^^^^^^^^

Carrier transport across the heterojunction is by thermionic emission. Carrier
densities are discontinuous at the interface: at equilibrium they differ by
the Boltzmann factor of the band offsets, and under bias the interface current
is driven by the imbalance of the emission fluxes from the two sides.

The emission rates derive from the per-material Richardson constants
:attr:`SemiconductorMedium.richardson_electron` and
:attr:`SemiconductorMedium.richardson_hole`. When these are not set, a large
default is used that keeps the interface at its local-equilibrium band-offset
alignment; set them explicitly to model finite thermionic-emission rates.

.. note::

   Heterojunction support requires the accelerated solver (the default,
   ``HeatChargeSimulation.use_accelerated_solver=True``). The legacy solver
   ignores ``electron_affinity`` and treats all semiconductor interfaces as
   continuous.
