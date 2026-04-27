.. currentmodule:: tidy3d

Grid Specification
--------------------

An EME grid divides the simulation along its propagation axis into cells where
modes are solved locally, then matched at the cell interfaces.  Pick
:class:`.EMEUniformGrid` for evenly spaced cells that share a single
:class:`.EMEModeSpec`, :class:`.EMEExplicitGrid` to place cell boundaries at
known structural features with a per-cell mode spec, or
:class:`.EMECompositeGrid` to combine both within one simulation.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMEUniformGrid
   EMECompositeGrid
   EMEExplicitGrid
   EMEGrid
   EMEModeSpec
