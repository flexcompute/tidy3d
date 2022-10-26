*************
API Reference
*************

.. currentmodule:: tidy3d


Simulation
==========

Simulation
----------

.. autosummary::
   :toctree: _autosummary/

   Simulation


Boundary Conditions
===================

.. autosummary::
   :toctree: _autosummary/

   BoundarySpec
   Boundary
   BoundaryEdge


Types of Boundaries
-------------------

.. autosummary::
   :toctree: _autosummary/

   Periodic
   PECBoundary
   PMCBoundary
   BlochBoundary


Absorbing Boundaries
====================


Types of Absorbers
------------------

.. autosummary::
   :toctree: _autosummary/

   PML
   StablePML
   Absorber

Absorber Parameters
-------------------

.. autosummary::
   :toctree: _autosummary/

   AbsorberParams
   PMLParams

Geometry
========

.. autosummary::
   :toctree: _autosummary/

   Box
   Sphere
   Cylinder
   PolySlab
   GeometryGroup

Mediums
=======

Non-Dispersive Medium
---------------------

.. autosummary::
   :toctree: _autosummary/

   Medium
   AnisotropicMedium
   PECMedium

Dispersive Mediums
------------------

.. autosummary::
   :toctree: _autosummary/

   PoleResidue
   Lorentz
   Sellmeier
   Drude
   Debye


Material Library
----------------

.. toctree::
   material_library


Structures
==========

.. autosummary::
   :toctree: _autosummary/

   Structure
   MeshOverrideStructure


Sources
=======


Types of Sources
----------------

.. autosummary::
   :toctree: _autosummary/

   PointDipole
   UniformCurrentSource
   PlaneWave
   ModeSource
   GaussianBeam
   AstigmaticGaussianBeam


Source Time Dependence
----------------------

.. autosummary::
   :toctree: _autosummary/

   GaussianPulse
   .. ContinuousWave


Monitors
========

.. autosummary::
   :toctree: _autosummary/

   FieldMonitor
   FieldTimeMonitor
   FluxMonitor
   FluxTimeMonitor
   ModeMonitor
   ModeSolverMonitor
   PermittivityMonitor
   Near2FarCartesianMonitor
   Near2FarAngleMonitor
   Near2FarKSpaceMonitor
   DiffractionMonitor


Mode Specifications
===================

.. autosummary::
   :toctree: _autosummary/

   ModeSpec

Discretization
==============

.. autosummary::
   :toctree: _autosummary/

   GridSpec
   AutoGrid
   UniformGrid
   CustomGrid
   Coords
   FieldGrid
   YeeGrid
   Grid


Output Data
===========

All Data for a Simulation
-------------------------

.. autosummary::
   :toctree: _autosummary/

   SimulationData

Collections of Data from single monitor
---------------------------------------

.. autosummary::
   :toctree: _autosummary/

   FieldData
   FieldTimeData
   ModeSolverData
   PermittivityData
   FluxData
   FluxTimeData
   ModeData
   AbstractNear2FarData
   Near2FarAngleData
   Near2FarCartesianData
   Near2FarKSpaceData
   Near2FarSurface
   FarFields
   DiffractionData

Individual Datasets
-------------------

.. autosummary::
   :toctree: _autosummary/

   ScalarFieldDataArray
   ScalarModeFieldDataArray
   ScalarFieldTimeDataArray
   ModeAmpsDataArray
   ModeIndexDataArray
   FluxDataArray
   FluxTimeDataArray
   Near2FarAngleDataArray
   Near2FarCartesianDataArray
   Near2FarKSpaceDataArray
   DiffractionDataArray

Logging
=======

.. autosummary::
   :toctree: _autosummary/

   log
   set_logging_level
   set_logging_file


Submitting Simulations
======================

Through python API
------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.webapi.run
   tidy3d.web.webapi.upload
   tidy3d.web.webapi.estimate_cost
   tidy3d.web.webapi.get_info
   tidy3d.web.webapi.start
   tidy3d.web.webapi.monitor
   tidy3d.web.webapi.download
   tidy3d.web.webapi.load
   tidy3d.web.webapi.delete
   tidy3d.web.webapi.download_log
   tidy3d.web.webapi.download_json
   tidy3d.web.webapi.load_simulation

Convenience for Single and Batch
--------------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.container.Job
   tidy3d.web.container.Batch
   tidy3d.web.container.BatchData

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.task.TaskInfo
   tidy3d.web.task.TaskStatus


Plugins
=======

Mode Solver
-----------

.. autosummary::
   :toctree: _autosummary/

   plugins.ModeSolver

Dispersive Model Fitting
------------------------

.. autosummary::
   :toctree: _autosummary/

   plugins.DispersionFitter
   plugins.StableDispersionFitter
   plugins.AdvancedFitterParam

Scattering Matrix Calculator
----------------------------

.. autosummary::
   :toctree: _autosummary/

   plugins.ComponentModeler
   plugins.Port

Constants
=========

Physical Constants
------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.C_0
   tidy3d.HBAR
   tidy3d.Q_e
   tidy3d.ETA_0
   tidy3d.EPSILON_0
   tidy3d.MU_0


Tidy3D Special Constants
------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.inf
   tidy3d.PEC

Tidy3D Configuration
--------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.config.Tidy3dConfig

Default Absorber Parameters
---------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.DefaultPMLParameters
   tidy3d.DefaultStablePMLParameters
   tidy3d.DefaultAbsorberParameters

Abstract Models
===============

These are classes that are used to organize the tidy3d components, but aren't to be used directly in the code.  Documented here mainly for reference.


.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.base.Tidy3dBaseModel
   tidy3d.components.boundary.AbsorberSpec
   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.DispersiveMedium
   tidy3d.components.geometry.Geometry
   tidy3d.components.geometry.Planar
   tidy3d.components.geometry.Circular
   tidy3d.components.source.SourceTime
   tidy3d.components.source.Source
   tidy3d.components.source.FieldSource
   tidy3d.components.monitor.Monitor
