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
   TriangleMesh
   GeometryGroup
   ClipOperation

Transformations
----------------

.. autosummary::
   :toctree: _autosummary/

   RotationAroundAxis

Mediums
=======

Non-Dispersive Medium
---------------------

Spatially uniform
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/

   Medium
   PECMedium
   FullyAnisotropicMedium

Spatially varying
^^^^^^^^^^^^^^^^^   
.. autosummary::
   :toctree: _autosummary/

   CustomMedium   

Dispersive Mediums
------------------

Spatially uniform
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary/

   PoleResidue
   Lorentz
   Sellmeier
   Drude
   Debye

Spatially varying
^^^^^^^^^^^^^^^^^   
.. autosummary::
   :toctree: _autosummary/

   CustomPoleResidue
   CustomLorentz
   CustomSellmeier
   CustomDrude
   CustomDebye   

General Mediums (can be both dispersive and non-dispersive)
-----------------------------------------------------------

Spatially uniform
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/

   AnisotropicMedium
   Medium2D

Spatially varying
^^^^^^^^^^^^^^^^^   
.. autosummary::
   :toctree: _autosummary/   

   CustomAnisotropicMedium   

Medium Specifications (add properties to existing Medium)
---------------------------------------------------------

Nonlinear
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/
    
   NonlinearSpec
   NonlinearSusceptibility
   KerrNonlinearity
   TwoPhotonAbsorption

Time Modulation
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/

   ModulationSpec
   SpaceTimeModulation   
   ContinuousWaveTimeModulation
   SpaceModulation

Medium Perturbations
--------------------

Mediums with Heat and Charge Perturbation Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/

   PerturbationMedium
   PerturbationPoleResidue

Perturbation Specifications of Individual Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary/

   ParameterPerturbation
   LinearHeatPerturbation
   CustomHeatPerturbation
   LinearChargePerturbation
   CustomChargePerturbation

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
   CustomFieldSource
   CustomCurrentSource
   TFSF


Source Time Dependence
----------------------

.. autosummary::
   :toctree: _autosummary/

   GaussianPulse
   ContinuousWave
   CustomSourceTime


Monitors
========

Monitor Types
-------------

.. autosummary::
   :toctree: _autosummary/

   FieldMonitor
   FieldTimeMonitor
   FluxMonitor
   FluxTimeMonitor
   ModeMonitor
   ModeSolverMonitor
   PermittivityMonitor
   FieldProjectionCartesianMonitor
   FieldProjectionAngleMonitor
   FieldProjectionKSpaceMonitor
   DiffractionMonitor

Apodization Specification
-------------------------

.. autosummary::
   :toctree: _autosummary/

   ApodizationSpec


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


Field Projector
===============

.. autosummary::
   :toctree: _autosummary/

   FieldProjectionSurface
   FieldProjector


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
   FieldProjectionAngleData
   FieldProjectionCartesianData
   FieldProjectionKSpaceData
   DiffractionData

Individual Datasets
-------------------

.. autosummary::
   :toctree: _autosummary/

   SpatialDataArray
   PermittivityDataset
   ScalarFieldDataArray
   ScalarModeFieldDataArray
   ScalarFieldTimeDataArray
   ModeAmpsDataArray
   ModeIndexDataArray
   FluxDataArray
   FluxTimeDataArray
   FieldProjectionAngleDataArray
   FieldProjectionCartesianDataArray
   FieldProjectionKSpaceDataArray
   DiffractionDataArray

Scene
=====

Scene
-----

.. autosummary::
   :toctree: _autosummary/

   Scene


Heat Solver
===========

Heat Simulation
---------------

.. autosummary::
   :toctree: _autosummary/

   HeatSimulation

Material Thermal Specification
------------------------------

.. autosummary::
   :toctree: _autosummary/

   FluidSpec
   SolidSpec

Boundary Conditions Specification
---------------------------------

.. autosummary::
   :toctree: _autosummary/

   HeatBoundarySpec

Boundary Conditions Types
-------------------------

.. autosummary::
   :toctree: _autosummary/

   TemperatureBC
   ConvectionBC
   HeatFluxBC

Boundary Conditions Placement
-----------------------------

.. autosummary::
   :toctree: _autosummary/

   StructureStructureInterface
   StructureBoundary
   MediumMediumInterface
   StructureSimulationBoundary
   SimulationBoundary

Sources
-------

.. autosummary::
   :toctree: _autosummary/

   UniformHeatSource

Grid Specification
------------------

.. autosummary::
   :toctree: _autosummary/

   UniformUnstructuredGrid
   DistanceUnstructuredGrid

Monitors
--------

.. autosummary::
   :toctree: _autosummary/

   TemperatureMonitor

Monitor Data
------------

.. autosummary::
   :toctree: _autosummary/

   TemperatureData

Heat Simulation Data
--------------------

.. autosummary::
   :toctree: _autosummary/

   HeatSimulationData

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

   tidy3d.web.run
   tidy3d.web.upload
   tidy3d.web.estimate_cost
   tidy3d.web.real_cost
   tidy3d.web.get_info
   tidy3d.web.start
   tidy3d.web.monitor
   tidy3d.web.download
   tidy3d.web.load
   tidy3d.web.delete
   tidy3d.web.download_log
   tidy3d.web.download_json
   tidy3d.web.load_simulation
   tidy3d.web.run_async

Convenience for Single and Batch
--------------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.Job
   tidy3d.web.Batch
   tidy3d.web.BatchData

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.core.task_info.TaskInfo
   tidy3d.web.core.task_info.TaskStatus


Plugins
=======

Mode Solver
-----------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.plugins.mode.ModeSolver
   tidy3d.plugins.mode.ModeSolverData

Dispersive Model Fitting
------------------------

.. autosummary::
   :toctree: _autosummary/

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

   tidy3d.plugins.polyslab.ComplexPolySlab

Scattering Matrix Calculator
----------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.plugins.smatrix.ComponentModeler
   tidy3d.plugins.smatrix.Port
   tidy3d.plugins.smatrix.SMatrixDataArray

Resonance Finder
----------------

.. autosummary::
   :toctree: _autosummary/
        
   tidy3d.plugins.resonance.ResonanceFinder

Adjoint
-------

.. autosummary::
   :toctree: _autosummary/

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

   tidy3d.plugins.waveguide.RectangularDielectric

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

These are some classes that are used to organize the tidy3d components, but aren't to be used directly in the code.  Documented here mainly for reference.


.. autosummary::
   :toctree: _autosummary/

   tidy3d.components.base.Tidy3dBaseModel
   tidy3d.components.boundary.AbsorberSpec
   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.DispersiveMedium
   tidy3d.NonlinearModel
   tidy3d.Geometry
   tidy3d.components.geometry.base.Planar
   tidy3d.components.geometry.base.Circular
   tidy3d.components.source.SourceTime
   tidy3d.components.source.Source
   tidy3d.components.source.FieldSource
   tidy3d.components.monitor.Monitor
