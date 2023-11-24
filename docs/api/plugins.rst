
.. currentmodule:: tidy3d

Plugins
=======

Mode Solver
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.mode.ModeSolver
   tidy3d.plugins.mode.ModeSolverData

Dispersive Model Fitting
------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.dispersion.FastDispersionFitter
   tidy3d.plugins.dispersion.AdvancedFastFitterParam
   tidy3d.plugins.dispersion.DispersionFitter
   tidy3d.plugins.dispersion.AdvancedFitterParam
   tidy3d.plugins.dispersion.web.run
   tidy3d.plugins.dispersion.StableDispersionFitter

Self-intersecting Polyslab
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.polyslab.ComplexPolySlab

Scattering Matrix Calculator
----------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.ComponentModeler
   tidy3d.plugins.smatrix.Port
   tidy3d.plugins.smatrix.SMatrixDataArray

Resonance Finder
----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.resonance.ResonanceFinder

Adjoint
-------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.adjoint.web.run
   tidy3d.plugins.adjoint.web.run_async
   tidy3d.plugins.adjoint.JaxBox
   tidy3d.plugins.adjoint.JaxPolySlab
   tidy3d.plugins.adjoint.JaxMedium
   tidy3d.plugins.adjoint.JaxAnisotropicMedium
   tidy3d.plugins.adjoint.JaxCustomMedium
   tidy3d.plugins.adjoint.JaxStructure
   tidy3d.plugins.adjoint.JaxSimulation
   tidy3d.plugins.adjoint.JaxSimulationData
   tidy3d.plugins.adjoint.JaxModeData
   tidy3d.plugins.adjoint.JaxPermittivityDataset
   tidy3d.plugins.adjoint.JaxDataArray
   tidy3d.plugins.adjoint.utils.filter.ConicFilter
   tidy3d.plugins.adjoint.utils.filter.BinaryProjector
   tidy3d.plugins.adjoint.utils.penalty.RadiusPenalty

Waveguide
---------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.waveguide.RectangularDielectric
