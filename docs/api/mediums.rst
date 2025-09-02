
.. currentmodule:: tidy3d

EM Mediums
==========

Overview
--------

The EM medium describes the electromagnetic properties of the material. In conjunction with Maxwell's equations and the constitutive equations, these properties determine how EM waves propagate through and interact with the medium. Tidy3D supports a diverse set of medium types to model various material characteristics.

In the Tidy3D python API, medium classes are split into six broad categories:

+ `Non-dispersive`_ medium: Constant relative permittivity :math:`\varepsilon_r` and conductivity :math:`\sigma`
+ `Dispersive`_ medium: Optical property varies with frequency
+ `Anisotropic`_ medium: Optical property varies with direction of light propagation
+ `Spatially-varying`_ medium: Optical property varies with spatial coordinates
+ `Perturbation`_ medium: Optical property is perturbed by the result of a multiphysics (heat/charge) simulation
+ `Metallic/PEC/PMC`_: Metals in the shallow skin depth regime

Combinations are possible, e.g. a spatially-varying, dispersive, anisotropic medium. In addition to these categories, users can define additional modifiers on some mediums using medium specifications:

+ `Nonlinearity`_: Optical property depends on electric field
+ `Spatio-temporal modulation`_: Optical property is modulated by a function of time and spatial coordinates

The material library also provides a growing list of commonly used material models.

+ `Material Library <material_library.html>`_
+ `RF Material Library <rf_material_library.html>`_

~~~~

Non-Dispersive
--------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Medium

A simple, non-dispersive optical medium can be described by constant relative permittivity :math:`\varepsilon_r` and conductivity :math:`\sigma` (units: S/um). Alternatively, it can also be defined using the real (:math:`n`) and imaginary (:math:`k`) components of the refractive index, but note that this is exact only at a single specified frequency (units: Hz). Thus the latter option is only appropriate for narrow-band simulations.

.. code-block:: python

   # A lossless medium with constant permittivity
   my_lossless_medium = Medium(permittivity=3.8)

   # A lossy medium with constant permittivity and conductivity
   my_lossy_medium = Medium(permittivity=4.0, conductivity=1.0)

   # Alternatively, use refractive index n and k at a specified frequency
   my_medium_nk = Medium.from_nk(n=1.5, k=1e-3, freq=150e12)

.. note::
   
   To specify a constant :math:`n` and :math:`k` over a wide frequency range, it is required to use a dispersive model. Please see the section on `Dispersive`_ mediums.

.. note::

   To specify a lossy dielectric using a constant loss tangent model, it is required to use a dispersive model. Please see the section on `Dispersive`_ mediums. 

~~~~

Dispersive
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PoleResidue
   tidy3d.Lorentz
   tidy3d.Sellmeier
   tidy3d.Drude
   tidy3d.Debye

There are many different models that can be used to describe dispersive mediums. For detailed usage instructions, please visit their respective documentation page.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.dispersion.FastDispersionFitter

Alternatively, the ``FastDispersionFitter`` plugin can be used to generate a dispersive medium from external data. The data can be provided as a local text file or a web URL from the materials database `refractiveindex.info <https://refractiveindex.info>`_.

.. code-block:: python

   # importing from a local file
   my_imported_medium = FastDispersionFitter.from_file("path/to/file.csv", skiprows=1, delimiter=",")

   # importing from a URL
   my_imported_medium_fromurl = FastDispersionFitter.from_url(
       url_file="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Johnson.yml",
       delimiter=",",
   )

To define a material with constant loss tangent, use the ``constant_loss_tangent_model()`` method.

.. code-block:: python

   # constant loss tangent
   my_losstan_medium = FastDispersionFitter.constant_loss_tangent_model(
       eps_real=2.2,
       loss_tangent=5e-4,
       frequency_range=(1e9, 5e9)
   )

Use the convenience methods ``plot()`` and ``eps_model()`` to access the optical properties of the dispersive material model.

.. code-block:: python

   # plotting within applicable frequency range
   my_freqs = np.linspace(3e14, 5e14, 201)
   _ = my_dispersive_medium.plot(freqs=my_freqs)
   
   # evaluate complex-valued permittivity at a frequency point
   eps = my_dispersive_medium.eps_model(3e14)

.. seealso::

   For more in-depth discussion and examples, please see the following learning center articles:

   + `Modeling dispersive materials <../notebooks/Dispersion.html>`_
   + `Fitting dispersive material models <../notebooks/Fitting.html>`_

   For a background introduction to modeling dispersive materials in FDTD, please see the following FDTD101 resource:

   + `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

~~~~

Anisotropic
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.AnisotropicMedium
   tidy3d.FullyAnisotropicMedium
   tidy3d.Medium2D

An anisotropic medium has different optical properties depending on the direction of light propagation. Its relative permittivity is thus specified in the form of a 3x3 tensor. For diagonally anisotropic mediums, use ``AnisotropicMedium``:

.. code-block:: python

   # specify mediums for the xx, yy, zz diagonal components 
   medium_xx = Medium(permittivity=4.0)
   medium_yy = Medium(permittivity=4.1)
   medium_zz = Medium(permittivity=3.9)
   my_anisotropic_medium = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

Note that ``xx``, ``yy``, and ``zz`` can support dispersive mediums as well. To specify all 9 components of the tensor, use ``FullyAnisotropicMedium``:

.. code-block:: python

   # specify the full 3x3 tensor
   perm = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
   cond = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
   my_anisotropic_medium = FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

Currently, ``FullyAnisotropicMedium`` only supports non-dispersive mediums.

The ``Medium2D`` class is used to simulate 2D materials without an out-of-plane response.

.. seealso::

   For more in-depth discussion with examples, please see the following learning center articles:

   + `Defining fully anisotropic materials <../notebooks/FullyAnisotropic.html>`_
   + `Defining gyrotropic materials <../notebooks/Gyrotropic.html>`_

   Example applications:

   + `TFLN Adiabatic Waveguide Coupler <../notebooks/AdiabaticCouplerLN.html>`_
   + `TFLN Polarization Splitter/Rotator <../notebooks/LNOIPolarizationSplitterRotator.html>`_
   + `Hyperbolic Polaritons in Nanostructured hBN <../notebooks/NanostructuredBoronNitride.html>`_

~~~~

Metallic/PEC/PMC 
----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PECMedium
   tidy3d.PMCMedium
   tidy3d.LossyMetalMedium
   tidy3d.SurfaceImpedanceFitterParam
   tidy3d.HammerstadSurfaceRoughness
   tidy3d.HuraySurfaceRoughness

At lower frequencies, the EM field typically does not penetrate very far into the metallic medium. In this regime, metallic structures are commonly modeled as boundary conditions. In Tidy3D, a metallic medium is assigned to a structure and the corresponding boundary conditions are automatically applied to its geometric boundaries.

.. code-block:: python

   # lossless metal
   my_pec = PECMedium()

   # lossy metal
   my_lossy_metal = LossyMetalMedium(conductivity=58, freq_range=(1e9, 10e9))

The ``LossyMetalMedium`` class implements the surface impedance boundary condition (SIBC). It can also accept surface roughness specifications using the Hammerstad or Huray models. Please refer to its documentation page for details.

.. note::
   
   For lossy metallic mediums, always be sure to check the skin depth --- if the skin depth is not negligible compared to the structure size, then ``LossyMetalMedium`` may be not accurate. In that case, use a regular dispersive medium instead.

~~~~

.. _Spatially-varying:

Spatial Variation
-----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SpatialDataArray
   tidy3d.CustomMedium

The ``CustomMedium`` class is used to define a spatially-varying non-dispersive medium. The spatial distribution of the optical property is defined using the ``SpatialDataArray`` class. Below, we define an example spherical profile :math:`n(r) = \sqrt{2-(r/R)^2}` as in a Luneburg lens:

.. code-block:: python

   # spatial grid
   R = 20
   Nx, Ny, Nz = 100, 100, 100
   X = np.linspace(-R, R, Nx) 
   Y = np.linspace(-R, R, Ny) 
   Z = np.linspace(-R, R, Nz) 
   Xg, Yg, Zg = np.meshgrid(X, Y, Z, indexing='ij')
   rg = np.sqrt(Xg**2 + Yg**2 + Zg**2)
   
   # n profile
   n_profile = np.ones((Nx, Ny, Nz))
   n_profile[rg <= R] = np.sqrt(2-(rg[rg<=R]/R)**2)

   # define SpatialDataArray
   n_data = SpatialDataArray(n_profile, coords=dict(x=X, y=Y, z=Z))

   # define CustomMedium
   my_custom_medium = CustomMedium.from_nk(n_data)

Analogous classes also exist for spatially-varying dispersive and anisotropic mediums.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomPoleResidue
   tidy3d.CustomLorentz
   tidy3d.CustomSellmeier
   tidy3d.CustomDrude
   tidy3d.CustomDebye
   tidy3d.CustomAnisotropicMedium

.. seealso::

   For in-depth discussion with examples, see the following learning center article:

   + `Defining spatially-varying dielectric structures <../notebooks/CustomMediumTutorial.html>`_

   Example applications:

   + `3D Optical Luneburg Lens <../notebooks/OpticalLuneburgLens.html>`_

~~~~

.. _Perturbation:

Perturbation (Multiphysics)
---------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PerturbationMedium
   tidy3d.PerturbationPoleResidue
   tidy3d.NedeljkovicSorefMashanovich
   tidy3d.ParameterPerturbation

When performing a multiphysics simulation, the temperature or carrier density field will result in a perturbation to the optical medium.

.. code-block:: python

   # define a linear heat perturbation 
   my_linear_heat_perturbation = LinearHeatPerturbation(coeff=1e-5, temperature_ref = 300)

   # define perturbation medium
   my_perturbed_medium = PerturbationMedium(
       permittivity = 2.0,
       permittivity_perturbation = ParameterPerturbation(heat=my_linear_heat_perturbation)
   )

Multiphysics simulations require additional set up in addition to defining a perturbation medium. To get started, we recommend that you check out the example models listed below. 

.. seealso::

   Example applications (thermal):

   + `Thermally Tuned Ring Resonator <../notebooks/ThermallyTunedRingResonator.html>`_
   + `Thermally Tuned Waveguide <../notebooks/MetalHeaterPhaseShifter.html>`_

   Example applications (charge):

   + `Carrier Injection Based Mach-Zehnder Modulator <../notebooks/MachZehnderModulator.html>`_
   + `Thermo-optic Modulator with a Doped Silicon Heater <../notebooks/ThermoOpticDopedModulator.html>`_


~~~~


Nonlinearity
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.NonlinearSpec
   tidy3d.NonlinearSusceptibility
   tidy3d.KerrNonlinearity
   tidy3d.TwoPhotonAbsorption

Nonlinear mediums are modeled as regular mediums with an added `NonlinearSpec` specification.

.. code-block:: python

   # firstly, define a nonlinear model of interest
   my_nonlinear_model = NonlinearSusceptibility(chi3=1.0)

   # then, define the nonlinear spec
   my_nonlinear_spec = NonlinearSpec(models=[my_nonlinear_model], num_iters=5)

   # finally, define the nonlinear medium
   my_nonlinear_medium = Medium(permittivity=2.0, nonlinear_spec=my_nonlinear_spec)

Nonlinear problems are solved iteratively; use the ``num_iters`` parameter in ``NonlinearSpec`` to control the number of iterations. Currently, three nonlinear models are supported, ``NonlinearSusceptibility``, ``KerrNonlinearity``, and ``TwoPhotonAbsorption``. For details, please refer to the corresponding reference article.

.. note::

   Nonlinear simulations are generally quite complex and the proper approach can vary greatly on a case-by-case basis. Other than defining a nonlinear medium, additional simulation subtleties are likely present when compared to their linear counterparts. 

.. seealso::

   Example applications:

   + `Generation of Kerr Sideband <../notebooks/KerrSidebands.html>`_
   + `Bistability in Photonic Crystal Microcavities <../notebooks/BistablePCCavity.html>`_

~~~~

Spatio-temporal Modulation
--------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.ModulationSpec
   tidy3d.SpaceTimeModulation
   tidy3d.ContinuousWaveTimeModulation
   tidy3d.SpaceModulation

In active nanophotonic devices, the medium can be modulated both spatially and temporally. The ``ModulationSpec`` class allows the user to specify such modulations. This modulation is then passed as a parameter into one of the standard medium classes.

.. code-block:: python

   # define time modulation
   my_time_modulation = ContinuousWaveTimeModulation(freq0=1e12, amplitude=0.1, phase=0)

   # define spatial modulation
   my_spatial_modulation = SpaceModulation(amplitude=1.0, phase=0)

   # define space-time modulation
   my_spacetime_modulation = SpaceTimeModulation(time_modulation=my_time_modulation, space_modulation=my_spatial_modulation)

   # define modulation spec
   my_modulation_spec = ModulationSpec(permittivity=my_spacetime_modulation, conductivity=None)

   # define modulated material
   my_modulated_medium = Medium(permittivity=2.0, modulation_spec=my_modulation_spec)

In the example above, the ``ContinuousWaveTimeModulation`` specifies a time-harmonic modulation with frequency, amplitude and phase given by the respective parameters. The ``SpaceModulation`` with ``amplitude=1.0`` results in no spatial modulation. The two modulation factors are multiplied together, and the result is added to the unmodulated permittivity value. In the modulated medium, the permittivity now varies between ``1.9`` and ``2.1``.

.. seealso::

   For in-depth discussion with examples, see the following learning center article:

   + `Applying time modulation to materials <../notebooks/TimeModulationTutorial.html>`_

~~~~

Multiphysics Medium
-------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.material.multi_physics.MultiPhysicsMedium

This section is still under construction. 

~~~~

Abstract Base Classes
---------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.medium.AbstractPerturbationMedium
   tidy3d.components.medium.NonlinearModel

These are abstract base classes that are inherited by some of the medium classes above. The user should not need to interact with these classes during regular usage. 
   
~~~~
