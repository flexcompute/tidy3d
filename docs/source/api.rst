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

Grid
====

3D Coordinates
--------------

.. autosummary::
   :toctree: _autosummary/

   Coords

Field Grid
----------

.. autosummary::
   :toctree: _autosummary/

   FieldGrid


Yee Lattice Grid
----------------

.. autosummary::
   :toctree: _autosummary/

   YeeGrid

Simulation Grid
---------------

.. autosummary::
   :toctree: _autosummary/

   Grid


Absorbing Boundaries
====================

Perfectly Matched Layer (PML)
-----------------------------

.. autosummary::
   :toctree: _autosummary/

   PML

Stable PML
----------

.. autosummary::
   :toctree: _autosummary/

   StablePML

Adiabatic Absorber
------------------

.. autosummary::
   :toctree: _autosummary/

   Absorber


Geometry
========

Box
---

.. autosummary::
   :toctree: _autosummary/

   Box

Sphere
------

.. autosummary::
   :toctree: _autosummary/

   Sphere

Cylinder
--------

.. autosummary::
   :toctree: _autosummary/

   Cylinder

PolySlab
--------

.. autosummary::
   :toctree: _autosummary/

   PolySlab


Mediums
=======

Medium
------

.. autosummary::
   :toctree: _autosummary/

   Medium

Anisotropic Medium
------------------

.. autosummary::
   :toctree: _autosummary/

   AnisotropicMedium

Perfect Electrical Conductor (PEC)
----------------------------------

.. autosummary::
   :toctree: _autosummary/

   PECMedium

Dispersive Pole Residue Medium
------------------------------

.. autosummary::
   :toctree: _autosummary/

   PoleResidue

Dispersive Lorentz Medium
-------------------------

.. autosummary::
   :toctree: _autosummary/

   Lorentz

Dispersive Sellmeier Medium
---------------------------

.. autosummary::
   :toctree: _autosummary/

   Sellmeier


Dispersive Drude Medium
-----------------------

.. autosummary::
   :toctree: _autosummary/

   Drude

Dispersive Debye Medium
-----------------------

.. autosummary::
   :toctree: _autosummary/

   Debye


Material Library
----------------

.. toctree::
   material_library


Structures
==========

Structure
---------

.. autosummary::
   :toctree: _autosummary/

   Structure


Sources
=======


Volume Source
-------------

.. autosummary::
   :toctree: _autosummary/

   VolumeSource


Plane Wave
----------

.. autosummary::
   :toctree: _autosummary/

   PlaneWave


Gaussian Beam
-------------

.. autosummary::
   :toctree: _autosummary/

   GaussianBeam


ModeSource 
----------

.. autosummary::
   :toctree: _autosummary/

   ModeSource


Source Time Dependence
======================

Gaussian Pulse
--------------

.. autosummary::
   :toctree: _autosummary/

   GaussianPulse

Continuous Wave (CW)
--------------------

.. autosummary::
   :toctree: _autosummary/

   ContinuousWave


Monitors
========

FieldMonitor
------------

.. autosummary::
   :toctree: _autosummary/

   FieldMonitor

FieldTimeMonitor
----------------

.. autosummary::
   :toctree: _autosummary/

   FieldTimeMonitor

FluxMonitor
-----------

.. autosummary::
   :toctree: _autosummary/

   FluxMonitor

FluxTimeMonitor
---------------

.. autosummary::
   :toctree: _autosummary/

   FluxTimeMonitor

ModeMonitor
-----------

.. autosummary::
   :toctree: _autosummary/

   ModeMonitor


Modes
=====

Mode Specification
------------------

.. autosummary::
   :toctree: _autosummary/

   ModeSpec


Output Data
===========

Simulation Data
---------------
.. autosummary::
   :toctree: _autosummary/

   SimulationData

Vector Field Data
-----------------

.. autosummary::
   :toctree: _autosummary/

   FieldData

Scalar Field Data
-----------------

.. autosummary::
   :toctree: _autosummary/

   ScalarFieldData

Scalar Field Data (Time-domain)
-------------------------------

.. autosummary::
   :toctree: _autosummary/

   ScalarFieldTimeData

Flux Data
---------

.. autosummary::
   :toctree: _autosummary/

   FluxData

Flux Data (Time-domain)
-----------------------

.. autosummary::
   :toctree: _autosummary/

   FluxTimeData

Mode Amplitude Data
-------------------

.. autosummary::
   :toctree: _autosummary/
   ModeData

Log
===

.. autosummary::
   :toctree: _autosummary/

   log
   set_logging_level
   set_logging_file


Submitting Simulations
======================

Web API
-------

.. autosummary::
   :toctree: _autosummary/

   web.run
   web.upload
   web.get_info
   web.start
   web.monitor
   web.download
   web.load
   web.delete

Job Interface
-------------

.. autosummary::
   :toctree: _autosummary/

   web.Job
   .. web.Job.run
   .. web.Job.upload
   .. web.Job.get_info
   .. web.Job.start
   .. web.Job.monitor
   .. web.Job.download
   .. web.Job.load
   .. web.Job.delete   

Batch Processing
----------------

.. autosummary::
   :toctree: _autosummary/

   web.Batch
   .. web.Batch.run
   .. web.Batch.upload
   .. web.Batch.get_info
   .. web.Batch.start
   .. web.Batch.monitor
   .. web.Batch.download
   .. web.Batch.load
   .. web.Batch.delete

Info Containers
---------------

.. autosummary::
   :toctree: _autosummary/

   web.task.Task
   web.task.TaskInfo
   web.task.TaskStatus


Plugins
=======

Dispersive Model Fitting Tool
-----------------------------

.. autosummary::
   :toctree: _autosummary/

   plugins.DispersionFitter
   plugins.DispersionFitter.from_file
   plugins.DispersionFitter.fit
   plugins.DispersionFitter.plot

Mode Solver
-----------

.. autosummary::
   :toctree: _autosummary/

   plugins.ModeSolver
   plugins.ModeSolver.solve
   plugins.mode.mode_solver.ModeInfo

Near Field to Far Field Transformation
--------------------------------------

.. autosummary::
   :toctree: _autosummary/

   plugins.Near2Far
   plugins.Near2Far.fields_cartesian
   plugins.Near2Far.fields_spherical
   plugins.Near2Far.power_cartesian
   plugins.Near2Far.power_spherical
   plugins.Near2Far.radar_cross_section

Abstract Models
===============

.. currentmodule:: tidy3d.components.base


Tidy3D Base Model
-----------------

.. autopydantic_model:: Tidy3dBaseModel
   :inherited-members: Tidy3dBaseModel


.. currentmodule:: tidy3d.components.pml

Absorber Specification
----------------------

.. autopydantic_model:: AbsorberSpec
   :inherited-members: Tidy3dBaseModel

.. currentmodule:: tidy3d.components.medium


Abstract Medium
---------------

.. autopydantic_model:: AbstractMedium
   :inherited-members: Tidy3dBaseModel

.. currentmodule:: tidy3d.components.source


Source Time
-----------

.. autopydantic_model:: SourceTime
   :inherited-members: Tidy3dBaseModel

.. currentmodule:: tidy3d.components.monitor

Monitor
-------

.. autopydantic_model:: Monitor
   :inherited-members: Tidy3dBaseModel

.. currentmodule:: tidy3d

Types
-----

.. autosummary::
   :toctree: _autosummary/

   components.types.Size
   components.types.Coordinate
   components.types.Inf
   components.types.NegInf

