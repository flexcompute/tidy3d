
.. currentmodule:: tidy3d

Mediums
=======

Non-Dispersive Medium
---------------------

Spatially uniform
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Medium
   tidy3d.PECMedium
   tidy3d.FullyAnisotropicMedium

Spatially varying
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomMedium

Dispersive Mediums
------------------

Spatially uniform
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PoleResidue
   tidy3d.Lorentz
   tidy3d.Sellmeier
   tidy3d.Drude
   tidy3d.Debye

Spatially varying
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomPoleResidue
   tidy3d.CustomLorentz
   tidy3d.CustomSellmeier
   tidy3d.CustomDrude
   tidy3d.CustomDebye

General Mediums (can be both dispersive and non-dispersive)
-----------------------------------------------------------

Spatially uniform
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.AnisotropicMedium
   tidy3d.Medium2D

Spatially varying
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomAnisotropicMedium

Medium Specifications (add properties to existing Medium)
---------------------------------------------------------

Nonlinear
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

    tidy3d.NonlinearSpec
    tidy3d.NonlinearSusceptibility

Material Library
----------------

.. toctree::
   material_library
