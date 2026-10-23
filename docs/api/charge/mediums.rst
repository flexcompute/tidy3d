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

   AnisotropicPermittivity

.. note::
   ``permittivity`` takes a scalar or an ``AnisotropicPermittivity`` tensor,
   given by principal values and an optional rotation. The tensor is supported
   only in charge simulations on the accelerated solver, and reflection
   ``symmetry`` is allowed only along an axis the tensor does not couple to the
   others. For a permittivity that is strongly anisotropic within the simulated
   cross-section, ``linear_solver="direct"`` is recommended, since the
   iterative solver may not converge.

Mobility
^^^^^^^^^^^^^^

Models contributing to carrier mobility, assigned to
``SemiconductorMedium.mobility_n`` and ``mobility_p`` or, for
``CanaliFieldDependence``, ``TransferredElectronFieldDependence`` and
``LombardiSurfaceDependence``, attached to those models through
``field_dependence`` and ``surface_dependence``.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantMobilityModel
   CaugheyThomasMobility
   MasettiMobility
   AlloyMobility
   CanaliFieldDependence
   TransferredElectronFieldDependence
   LombardiSurfaceDependence

.. note::
   ``MasettiMobility``, ``AlloyMobility``, ``CanaliFieldDependence``,
   ``TransferredElectronFieldDependence`` and ``LombardiSurfaceDependence`` are
   supported only on the accelerated solver. ``CanaliFieldDependence`` (high-field
   velocity saturation) and ``TransferredElectronFieldDependence`` (the negative
   differential mobility of direct-gap III-V semiconductors) attach to any of the
   low-field mobility models through ``field_dependence``, and
   ``LombardiSurfaceDependence`` adds surface degradation near
   semiconductor/insulator interfaces to ``CaugheyThomasMobility``,
   ``MasettiMobility`` and ``AlloyMobility`` through ``surface_dependence``.

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
   Each mechanism (Shockley-Read-Hall, Auger, radiative, band-to-band tunneling
   and impact ionization) should appear at most once in a medium's ``R``: a
   mechanism listed twice warns, and only one of its entries is used. Several
   ``DistributedGeneration`` entries add up.


Doping
^^^^^^

Spatial dopant distributions applied to a semiconductor medium.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ConstantDoping
   GaussianDoping
   CustomDoping


Dopant Ionization
^^^^^^^^^^^^^^^^^

Incomplete ionization of the donors and acceptors, assigned to
``SemiconductorMedium.donor_ionization`` and ``acceptor_ionization``. Without
them the dopants are fully ionized.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   PearsonBardeenIonization
   AltermattSchenkIonization
   MultiSiteIonization
   IonizationSite

.. note::
   Incomplete ionization is supported only on the accelerated solver. The
   ionized concentrations enter the Poisson equation and the charge neutrality
   of ohmic contacts; mobility, lifetime and bandgap narrowing models keep
   reading the total doping, except that a ``SlotboomBandGapNarrowing`` is shifted
   so that neutral material keeps its fitted minority-carrier density.
   ``SteadyDopingMonitor`` records the total doping and the ionized
   concentrations.


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
