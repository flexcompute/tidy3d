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
   PointCloudPermittivityData
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

Point-cloud monitor data containers are indexed by ``index`` and frequency ``f`` rather than by structured ``x`` / ``y`` / ``z`` grid coordinates. Access point-cloud fields or permittivity components directly, for example ``sim_data["point_cloud"].Ex`` or ``sim_data["point_cloud_eps"].eps_xx``, and use the corresponding ``.points`` array to inspect the requested point coordinates. For ``PointCloudPermittivityData``, component values are sampled from the nearest native Yee-grid locations; ``.points`` does not expose those snapped component-grid sampling locations.


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
   PointCloudPermittivityDataset
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
