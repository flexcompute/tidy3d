.. currentmodule:: tidy3d

Boundary Conditions
===================

Overview
--------

~~~~

Boundary Specification
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.BoundarySpec
   tidy3d.Boundary

.. seealso::

   For more details and examples, please see the following article:

   + `Setting up boundary conditions <../notebooks/BoundaryConditions.html>`_
   
~~~~

PEC/PMC 
-------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PECBoundary
   tidy3d.PMCBoundary
   tidy3d.Boundary.pec
   tidy3d.Boundary.pmc

~~~~

Periodic 
--------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Periodic
   tidy3d.BlochBoundary
   tidy3d.Boundary.periodic
   tidy3d.Boundary.bloch
   tidy3d.Boundary.bloch_from_source

.. seealso::

   For more details and examples, please see the following notebooks:

   + `Multilevel blazed diffraction grating <../notebooks/GratingEfficiency.html>`_
   + `Defining a total-field scattered-field (TFSF) plane wave source <../notebooks/TFSF.html>`_

~~~~

Absorbing
---------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PML
   tidy3d.PMLParams
   tidy3d.Boundary.pml
   tidy3d.StablePML
   tidy3d.Boundary.stable_pml
   tidy3d.Absorber
   tidy3d.Boundary.absorber
   tidy3d.AbsorberParams


.. seealso::

   For more details and exeamples, please see the following article:

   + `Suppressing artificial reflections with absorber and PML boundaries <../notebooks/AbsorbingBoundaryReflection.html>`_

   For a general introduction to PMLs, please see the following FDTD101 resource:

   + `Introduction to perfectly matched layer (PML) <https://www.flexcompute.com/fdtd101/Lecture-6-Introduction-to-perfectly-matched-layer/>`_

~~~~
