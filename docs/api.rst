*************
API Reference
*************

.. currentmodule:: tidy3d


Simulation
==========

Simulation
----------

.. autopydantic_model:: Simulation
   :inherited-members: Tidy3dBaseModel

Size
----

.. autopydantic_field:: components.types.Size

Grid
====

3D Coordinates
--------------

.. autopydantic_model:: Coords
   :inherited-members: Tidy3dBaseModel

Field Grid
----------

.. autopydantic_model:: FieldGrid
   :inherited-members: Tidy3dBaseModel

Yee Lattice Grid
----------------

.. autopydantic_model:: YeeGrid
   :inherited-members: Tidy3dBaseModel

Simulation Grid
---------------

.. autopydantic_model:: Grid
   :inherited-members: Tidy3dBaseModel


Absorbing Boundaries
====================

Perfectly Matched Layer (PML)
-----------------------------

.. autopydantic_model:: PML
   :inherited-members: Tidy3dBaseModel

Stable PML
----------

.. autopydantic_model:: StablePML
   :inherited-members: Tidy3dBaseModel

Adiabatic Absorber
------------------

.. autopydantic_model:: Absorber
   :inherited-members: Tidy3dBaseModel


Geometry
========

Box
---

.. autopydantic_model:: Box
   :inherited-members: Tidy3dBaseModel

Sphere
------

.. autopydantic_model:: Sphere
   :inherited-members: Tidy3dBaseModel

Cylinder
--------

.. autopydantic_model:: Cylinder
   :inherited-members: Tidy3dBaseModel

PolySlab
--------

.. autopydantic_model:: PolySlab
   :inherited-members: Tidy3dBaseModel

Mediums
=======

Medium
------

.. autopydantic_model:: Medium
   :inherited-members: Tidy3dBaseModel

Anisotropic Medium
------------------

.. autopydantic_model:: AnisotropicMedium
   :inherited-members: Tidy3dBaseModel

Perfect Electrical Conductor (PEC)
----------------------------------

.. autopydantic_model:: PECMedium
   :inherited-members: Tidy3dBaseModel

Dispersive Pole Residue Medium
------------------------------

.. autopydantic_model:: PoleResidue
   :inherited-members: Tidy3dBaseModel

Dispersive Lorentz Medium
-------------------------

.. autopydantic_model:: Lorentz
   :inherited-members: Tidy3dBaseModel

Dispersive Sellmeier Medium
---------------------------

.. autopydantic_model:: Sellmeier
   :inherited-members: Tidy3dBaseModel


Dispersive Drude Medium
-----------------------

.. autopydantic_model:: Drude
   :inherited-members: Tidy3dBaseModel

Dispersive Debye Medium
-----------------------

.. autopydantic_model:: Debye
   :inherited-members: Tidy3dBaseModel


Material Library
----------------

.. toctree::

   material_library


Structures
==========

Structure
---------

.. autopydantic_model:: Structure
   :inherited-members: Tidy3dBaseModel


Sources
=======


Volume Source
-------------

.. autopydantic_model:: VolumeSource
   :inherited-members: Tidy3dBaseModel


Plane Wave
----------

.. autopydantic_model:: PlaneWave
   :inherited-members: Tidy3dBaseModel


Gaussian Beam
-------------

.. autopydantic_model:: GaussianBeam
   :inherited-members: Tidy3dBaseModel


ModeSource 
----------

.. autopydantic_model:: ModeSource
   :inherited-members: Tidy3dBaseModel


Source Time Dependence
======================

Gaussian Pulse
--------------

.. autopydantic_model:: GaussianPulse
   :inherited-members: Tidy3dBaseModel

Continuous Wave (CW)
--------------------

.. autopydantic_model:: ContinuousWave
   :inherited-members: Tidy3dBaseModel


Monitors
========

FieldMonitor
------------

.. autopydantic_model:: FieldMonitor
   :inherited-members: Tidy3dBaseModel

FieldTimeMonitor
----------------

.. autopydantic_model:: FieldTimeMonitor
   :inherited-members: Tidy3dBaseModel

FluxMonitor
-----------

.. autopydantic_model:: FluxMonitor
   :inherited-members: Tidy3dBaseModel

FluxTimeMonitor
---------------

.. autopydantic_model:: FluxTimeMonitor
   :inherited-members: Tidy3dBaseModel

ModeMonitor
-----------

.. autopydantic_model:: ModeMonitor
   :inherited-members: Tidy3dBaseModel


Modes
=====

Mode Specification
------------------

.. autopydantic_model:: ModeSpec
   :inherited-members: Tidy3dBaseModel


Output Data
===========


.. autosummary::
   :toctree: _autosummary/

   SimulationData
   FieldData
   FluxData
   FluxTimeData
   ModeData

.. Output Data
.. ===========

.. Simulation Data
.. ---------------

.. .. autopydantic_model:: SimulationData
..    :inherited-members: Tidy3dBaseModel

.. Vector Field Data
.. -----------------

.. .. autopydantic_model:: FieldData
..    :inherited-members: Tidy3dBaseModel

.. Scalar Field Data
.. -----------------


.. .. autopydantic_model:: ScalarFieldData
..    :inherited-members: Tidy3dBaseModel

.. Scalar Field Data (Time Domain)
.. -------------------------------

.. .. autopydantic_model:: ScalarFieldTimeData
..    :inherited-members: Tidy3dBaseModel


.. Flux Data
.. ---------

.. .. autopydantic_model:: FluxData
..    :inherited-members: Tidy3dBaseModel

.. Flux Data (Time Domain)
.. -----------------------

.. .. autopydantic_model:: FluxTimeData
..    :inherited-members: Tidy3dBaseModel

.. Mode Data
.. ---------

.. .. autopydantic_model:: ModeData
..    :inherited-members: Tidy3dBaseModel


.. Constants
.. =========
.. .. automodule:: tidy3d.constants
..    :members:

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
   web.Job.run
   web.Job.upload
   web.Job.get_info
   web.Job.start
   web.Job.monitor
   web.Job.download
   web.Job.load
   web.Job.delete   

Batch Processing
----------------

.. autosummary::
   :toctree: _autosummary/

   web.Batch
   web.Batch.run
   web.Batch.upload
   web.Batch.get_info
   web.Batch.start
   web.Batch.monitor
   web.Batch.download
   web.Batch.load
   web.Batch.delete

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
   components.types.Coords

