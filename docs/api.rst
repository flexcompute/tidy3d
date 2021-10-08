*************
API Reference
*************

.. currentmodule:: tidy3d

Components
==========

Simulation Specification
------------------------

.. autosummary::
   :toctree: _autosummary/

   Simulation
   PMLLayer

Monitors
--------

.. autosummary::
   :toctree: _autosummary/

   FieldMonitor
   FieldTimeMonitor
   PermittivityMonitor
   FluxMonitor
   FluxTimeMonitor
   ModeMonitor
   Mode


Simulation Output Data
----------------------

.. autosummary::
   :toctree: _autosummary/

   SimulationData
   FieldData
   FieldTimeData
   PermittivityData
   FluxData
   FluxTimeData
   ModeData


Submitting Simulations
======================

Web API
-------

.. autosummary::
   :toctree: _autosummary/

   web.upload
   web.get_info
   web.get_run_info
   web.run
   web.monitor
   web.download
   web.load_results
   web.delete

Job Interface
-------------

.. autosummary::
   :toctree: _autosummary/

   web.Job
   web.Job.upload
   web.Job.get_info
   web.Job.get_run_info
   web.Job.run
   web.Job.monitor
   web.Job.download
   web.Job.load_results
   web.Job.delete   

Batch Processing
----------------

.. autosummary::
   :toctree: _autosummary/

   web.Batch
   web.Batch.upload
   web.Batch.get_info
   web.Batch.get_run_info
   web.Batch.run
   web.Batch.monitor
   web.Batch.download
   web.Batch.load_results
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
   plugins.DispersionFitter.load
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


.. Simulation
.. ==========

.. .. autosummary::
..    :toctree: _autosummary/

..    Simulation
..    PMLLayer


.. Geometry
.. ========

.. .. autosummary::
..    :toctree: _autosummary/

..    Box
..    Sphere
..    Cylinder
..    PolySlab


.. Medium
.. ======

.. .. autosummary::
..    :toctree: _autosummary/

..    Medium

.. Dispersive Media
.. ----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    PoleResidue
..    Sellmeier
..    Lorentz
..    Debye

.. Material Library
.. ----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    material_library


.. Structure
.. =========

.. .. autosummary::
..    :toctree: _autosummary/

..    Structure


.. Source
.. ======

.. .. autosummary::
..    :toctree: _autosummary/

..    VolumeSource
..    ModeSource
..    PlaneWave
..    ..GaussianBeam

.. Source Time Dependence
.. ----------------------

.. .. autosummary::
..    :toctree: _autosummary/

..    GaussianPulse
..    ..CW


.. Monitor
.. =======

.. .. autosummary::
..    :toctree: _autosummary/

..    FluxMonitor
..    FieldMonitor
..    ModeMonitor

.. Monitor Samplers
.. ----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    TimeSampler
..    FreqSampler

..    uniform_times
..    uniform_freqs


.. Modes
.. =====

.. .. autosummary::
..    :toctree: _autosummary/

..    Mode

