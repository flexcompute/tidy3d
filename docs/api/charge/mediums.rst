.. currentmodule:: tidy3d

Charge Mediums
-------------------------------

The electrical behaviour assigned to a structure's ``medium.charge``. Conductors
and insulators set contacts and dielectric regions; the semiconductor medium
carries the doping, mobility, recombination and band models that the
drift-diffusion solver uses.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ChargeConductorMedium
   ChargeInsulatorMedium
   SemiconductorMedium

Mobility
^^^^^^^^^^^^^^

Models contributing to carrier mobility, assigned to
``SemiconductorMedium.mobility_n`` and ``mobility_p`` or, for
``CanaliFieldDependence``, attached to those models through ``field_dependence``.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantMobilityModel
   CaugheyThomasMobility
   MasettiMobility
   CanaliFieldDependence

.. note::
   ``MasettiMobility`` is supported only on the accelerated solver, as is
   ``CanaliFieldDependence``, which adds high-field velocity saturation on top
   of any of the low-field mobility models through their ``field_dependence``
   parameter.

Generation Recombination
^^^^^^^^^^^^^^^^^^^^^^^^^

Carrier generation and recombination models contributing to the
drift-diffusion source terms.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   AugerRecombination
   RadiativeRecombination
   ShockleyReedHallRecombination
   FossumCarrierLifetime
   PalankovskiQuayApproxCarrierLifetime
   DistributedGeneration
   HurkxDirectBandToBandTunneling
   HurkxTrapAssistedTunneling
   SelberherrImpactIonization

.. note::
   At most one ``ShockleyReedHallRecombination`` model may be specified per
   medium; if several are provided only the last one is used.


Doping
^^^^^^

Spatial dopant distributions applied to a semiconductor medium.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantDoping
   GaussianDoping
   CustomDoping


Bandgap
^^^^^^^

Bandgap-narrowing models for heavily doped semiconductors.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   SlotboomBandGapNarrowing


Effective Density Of States (DOS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Effective density-of-states models for the conduction and valence bands.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantEffectiveDOS
   IsotropicEffectiveDOS
   MultiValleyEffectiveDOS
   DualValleyEffectiveDOS

Energy Bandgap
^^^^^^^^^^^^^^

Temperature dependence of the semiconductor energy bandgap.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantEnergyBandGap
   VarshniEnergyBandGap

Charge Carrier Properties
------------------------------------

Models describing how the optical properties of a medium change with the local
carrier concentration.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   LinearChargePerturbation
   CustomChargePerturbation
