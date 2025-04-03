.. currentmodule:: tidy3d

Grid Discretization
===================

Overview
--------

.. seealso::

   For an introduction to FDTD discretization and related topics, please see the following FDTD101 lectures:

   + `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/>`_
   + `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
   + `Numerical Dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_

~~~~

Grid Specification
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GridSpec
   tidy3d.AutoGrid
   tidy3d.UniformGrid
   tidy3d.QuasiUniformGrid
   tidy3d.CustomGrid
   tidy3d.CustomGridBoundaries


.. seealso::

   For more detail explanation and examples, please see the following learning center resources:

   + `Using automatic nonuniform meshing <../notebooks/AutoGrid.html>`_

~~~~

Refinement
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.MeshOverrideStructure
   tidy3d.LayerRefinementSpec
   tidy3d.GridRefinement
   tidy3d.CornerFinderSpec

.. seealso::

   For more detail explanation and examples, please see the following learning center resources:

   + `Using automatic nonuniform meshing: Mesh override structures <../notebooks/AutoGrid.html#Mesh-override-structures>`_
   + `Automatic mesh refinement in layered structures <../notebooks/LayerRefinement.html>`_

~~~~

Subpixel Averaging
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SubpixelSpec

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Staircasing
   tidy3d.VolumetricAveraging
   tidy3d.HeuristicPECStaircasing
   tidy3d.PolarizedAveraging
   tidy3d.ContourPathAveraging
   tidy3d.PECConformal
   tidy3d.SurfaceImpedance

.. seealso::

   For an introduction to subpixel averaging, please see the following FDTD101 lectures:

   + `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_
   + `Introduction to subpixel averaging <https://www.flexcompute.com/fdtd101/Lecture-10-Introduction-to-subpixel-averaging/>`_

~~~~

Utility Classes
---------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Coords
   tidy3d.FieldGrid
   tidy3d.YeeGrid
   tidy3d.Grid

~~~~



