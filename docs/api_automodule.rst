*************
API Reference
*************

Base
====

.. automodule:: tidy3d.components.base
   :members:
   :inherited-members: Tidy3dBaseModel


Simulation
==========

.. automodule:: tidy3d.components.base
   :members:
   :inherited-members: Tidy3dBaseModel

Grid
====

.. automodule:: tidy3d.components.grid
   :members:
   :inherited-members: Tidy3dBaseModel


Absorbing Boundaries
====================


.. automodule:: tidy3d.components.pml
   :members:
   :inherited-members: Tidy3dBaseModel

Geometry
========

.. automodule:: tidy3d.components.geometry
   :members:
   :inherited-members: Tidy3dBaseModel

Mediums
=======

.. automodule:: tidy3d.components.medium
   :members:
   :inherited-members: Tidy3dBaseModel

Structures
==========

.. automodule:: tidy3d.components.structure
   :members:
   :inherited-members: Tidy3dBaseModel

Sources
=======

.. automodule:: tidy3d.components.source
   :members:
   :inherited-members: Tidy3dBaseModel


Monitors
========

.. automodule:: tidy3d.components.monitor
   :members:
   :inherited-members: Tidy3dBaseModel


Modes
=====

Mode Specification
------------------

.. automodule:: tidy3d.components.mode
   :members:
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


Simulation
==========

.. automodule:: tidy3d.components.simulation
   :members:

Sources
=======

.. automodule:: tidy3d.components.source
   :members:
