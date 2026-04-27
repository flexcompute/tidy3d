.. currentmodule:: tidy3d

Output Data
-------------

Monitor Data
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMECoefficientData
   EMEModeSolverData
   EMEFieldData


Simulation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMESimulationData


Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMESMatrixDataset
   EMECoefficientDataset
   EMEOverlapDataset
   EMEFieldDataset
   EMEModeSolverDataset


Pipeline Stages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intermediate artifacts produced by the local propagation pipeline (see
:meth:`.EMESimulation.propagate`). All five classes support HDF5 round-trip,
so they can be checkpointed to disk and reused across design iterations.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   EMEStageCellModes
   EMEStageCellOverlap
   EMEStageInterfaceOverlap
   EMEStageCellSMatrix
   EMEStageInterfaceSMatrix
