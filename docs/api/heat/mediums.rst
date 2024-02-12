.. currentmodule:: tidy3d

Material Thermal Specification
-------------------------------

To create simulation setups that can be used in different solvers we need to create material specifications that contain all relevant information for each of the solvers. Specifically, when performing coupled thermal and optic simulations, each material definition will contain up to three different characteristic:

1. Optic properties such as permittivity and conductivity
2. Thermal properties such as thermal conductivity
3. Response of optic properties to changes in temperature

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.FluidSpec
   tidy3d.SolidSpec


Thermal Perturbation Specification
------------------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.LinearHeatPerturbation
   tidy3d.CustomHeatPerturbation

