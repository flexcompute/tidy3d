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

   SimulationData

~~~~

The ``BatchData`` Object
------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.api.container.BatchData


~~~~

List of Monitor Data Types
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   FieldData
   PointCloudFieldData
   DipoleEmissionData
   FieldTimeData
   ModeSolverData
   PermittivityData
   FluxData
   FluxTimeData
   ModeData
   FieldProjectionAngleData
   FieldProjectionCartesianData
   FieldProjectionKSpaceData
   DiffractionData
   rf.DirectivityData
   AuxFieldTimeData
   SurfaceFieldData
   SurfaceFieldTimeData


~~~~

List of Dataset Types
---------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   SpatialDataArray
   PermittivityDataset
   ScalarFieldDataArray
   SphericalAngleDataArray
   DipoleEmissionDataArray
   DipoleEmissionPositionDataArray
   PointCloudFieldDataset
   ScalarModeFieldDataArray
   ScalarFieldTimeDataArray
   components.data.data_array.FreqDataArray
   components.data.data_array.FreqModeDataArray
   ModeAmpsDataArray
   ModeIndexDataArray
   FluxDataArray
   FluxTimeDataArray
   FieldProjectionAngleDataArray
   FieldProjectionCartesianDataArray
   FieldProjectionKSpaceDataArray
   DiffractionDataArray
   SteadyVoltageDataArray
   IndexedSurfaceFieldDataArray
   IndexedSurfaceFieldTimeDataArray
   IndexedSurfaceFreqDataArray
   IndexedSurfaceTimeDataArray
   TriangularSurfaceDataset
