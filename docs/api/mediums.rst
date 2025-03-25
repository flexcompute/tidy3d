
.. currentmodule:: tidy3d

EM Mediums
==========

Basic Medium
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Medium

A simple optical medium can be described by its relative permittivity and conductivity (units S/um).

.. code-block:: python

   my_medium = Medium(permittivity=4.0, conductivity=1.0)
   my_medium_nk = Medium.from_nk(n=1.5, k=1e-3, freq=1e12)

Equivalently, we can also specify the real and imaginary components of the refractive index, n and k respectively. Note that this latter option requires us to specify the frequency (units Hz) at which to perform the conversion.

Metallic Structure
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PECMedium
   tidy3d.LossyMetalMedium
   tidy3d.SurfaceImpedanceFitterParam

At lower frequencies, the EM field does not penetrate very far into metallic structures. Hence, they are commonly modeled as boundary conditions. In Tidy3D, we assign a metallic medium to a structure and the corresponding boundary conditions are automatically applied to its geometric boundaries.

.. code-block:: python

   my_pec = PECMedium()
   my_lossy_metal = LossyMetalMedium(conductivity=58, freq_range=(1e9, 10e9))

For lossy metallic mediums, always be sure to check the skin depth --- if the skin depth is not negligible compared to the structure size, then ``LossyMetalMedium`` may be not accurate. In that case, we should model the metal as a regular dispersive medium. 

Dispersive Medium
-----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PoleResidue
   tidy3d.Lorentz
   tidy3d.Sellmeier
   tidy3d.Drude
   tidy3d.Debye

There are many different models that can be used to describe dispersive mediums. Please visit their respective documentation page for usage tips.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.dispersion.FastDispersionFitter

Alternatively, we can also fit a dispersion model from data using the ``FastDispersionFitter`` plugin.

.. code-block:: python

   pass

In general, we can visualize the optical property as a function of frequency.

.. code-block:: python

   pass

For more in-depth discussion and examples on dispersive material modeling in Tidy3D, please see the following learning center articles:

+ `Modeling dispersive materials<../notebooks/Dispersion.html>`_
+ `Fitting dispersive material models<../notebooks/Fitting.html>`_

For a background introduction to modeling dispersive materials in FDTD, please see the following FDTD101 resource:

+ `Modeling dispersive material in FDTD<https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_. 

Anisotropic Medium
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.AnisotropicMedium
   tidy3d.FullyAnisotropicMedium
   tidy3d.Medium2D

An anisotropic medium has different optical properties depending on the direction of light propagation. Its relative permittivity is thus specified in the form of a 3x3 tensor. For a non-dispersive anistropic material, use the ``FullyAnisotropicMedium`` like so:

.. code-block:: python

   perm = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
   cond = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
   my_anisotropic_medium = FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

For a dispersive anisotropic medium, use the ``AnisotropicMedium`` instead:

.. code-block:: python

   medium_xx = Medium(permittivity=4.0)
   medium_yy = Medium(permittivity=4.1)
   medium_zz = Medium(permittivity=3.9)
   my_anisotropic_medium = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

Note that ``xx``, ``yy``, and ``zz`` can accept any medium, including dispersive and metallic medium definitions. Currently, only diagonal anisotropy is supported for ``AnisotropicMedium``.

For more in-depth discussion with examples, please see the following learning center articles:

+ `Defining fully anisotropic materials<../notebooks/FullyAnisotropic.html>`_
+ `Defining gyrotropic materials<../notebooks/Gyrotropichtml>`_

Spatially Varying Medium
------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomMedium

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomPoleResidue
   tidy3d.CustomLorentz
   tidy3d.CustomSellmeier
   tidy3d.CustomDrude
   tidy3d.CustomDebye

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomAnisotropicMedium




Medium Perturbations
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PerturbationMedium
   tidy3d.PerturbationPoleResidue


Medium Specifications (add properties to existing Medium)
---------------------------------------------------------

Nonlinear
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.NonlinearSpec
   tidy3d.NonlinearSusceptibility
   tidy3d.KerrNonlinearity
   tidy3d.TwoPhotonAbsorption

Time Modulation
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/

   tidy3d.ModulationSpec
   tidy3d.SpaceTimeModulation
   tidy3d.ContinuousWaveTimeModulation
   tidy3d.SpaceModulation

Material Library
----------------

.. toctree::
   material_library
   rf_material_library


Abstract Classes
-----------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.medium.AbstractPerturbationMedium
   tidy3d.components.medium.NonlinearModel


Multi-Physics Medium
====================


.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.material.multi_physics.MultiPhysicsMedium



