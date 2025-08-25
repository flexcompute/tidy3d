.. currentmodule:: tidy3d

Output Data
===========

Overview
--------

This page contains information on simulation and monitor data containers. 

.. TODO
   - Expand overview
   - Working with monitor data
     - Distinction between MonitorData and Dataset
     - Dataset structure: attributes, coordinates and slicing
     - Saving/reading from file
     - Plotting and Visualization

~~~~

The ``SimulationData`` Object
-----------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SimulationData

~~~~

The ``BatchData`` Object
------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.container.BatchData


~~~~

List of Monitor Data Types
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.FieldData
   tidy3d.FieldTimeData
   tidy3d.ModeSolverData
   tidy3d.PermittivityData
   tidy3d.FluxData
   tidy3d.FluxTimeData
   tidy3d.ModeData
   tidy3d.FieldProjectionAngleData
   tidy3d.FieldProjectionCartesianData
   tidy3d.FieldProjectionKSpaceData
   tidy3d.DiffractionData
   tidy3d.DirectivityData
   tidy3d.AuxFieldTimeData


~~~~

List of Dataset Types
---------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.SpatialDataArray
   tidy3d.PermittivityDataset
   tidy3d.ScalarFieldDataArray
   tidy3d.ScalarModeFieldDataArray
   tidy3d.ScalarFieldTimeDataArray
   tidy3d.ModeAmpsDataArray
   tidy3d.ModeIndexDataArray
   tidy3d.FluxDataArray
   tidy3d.FluxTimeDataArray
   tidy3d.FieldProjectionAngleDataArray
   tidy3d.FieldProjectionCartesianDataArray
   tidy3d.FieldProjectionKSpaceDataArray
   tidy3d.DiffractionDataArray
   tidy3d.DirectivityDataArray
   tidy3d.AxialRatioDataArray
