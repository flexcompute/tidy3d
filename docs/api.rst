*************
API Reference
*************

.. currentmodule:: tidy3d


Simulation
==========

.. autosummary::
   :toctree: _autosummary/

   Simulation

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   Simulation.plot
   Simulation.plot_eps
   Simulation.plot_structures
   Simulation.plot_structures_eps
   Simulation.plot_sources
   Simulation.plot_monitors
   Simulation.plot_symmetries
   Simulation.plot_pml
   Simulation.plot_grid
   Simulation.grid
   Simulation.dt
   Simulation.tmesh
   Simulation.wvl_mat_min
   Simulation.frequency_range
   Simulation.pml_thicknesses
   Simulation.num_pml_layers
   Simulation.discretize
   Simulation.epsilon   


Grid
====

.. autosummary::
   :toctree: _autosummary/

   Coords
   FieldGrid
   YeeGrid
   Coords1D
   Grid
   Grid.centers
   Grid.sizes
   Grid.yee


Absorbing Boundaries
====================

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
   Box.from_bounds
   Sphere
   Cylinder
   PolySlab
   PolySlab.from_gds

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   Geometry.plot
   Geometry.inside
   Geometry.intersections
   Geometry.intersects
   Geometry.intersects_plane
   Geometry.bounds
   Geometry.bounding_box
   Geometry.pop_axis
   Geometry.unpop_axis


Mediums
=======

.. autosummary::
   :toctree: _autosummary/

   Medium
   Medium.from_nk
   AnisotropicMedium
   PEC
   PoleResidue
   Sellmeier
   Debye
   Lorentz
   Drude

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   AbstractMedium.plot
   AbstractMedium.eps_model
   AbstractMedium.nk_to_eps_sigma
   AbstractMedium.nk_to_eps_complex
   AbstractMedium.eps_sigma_to_eps_complex
   AbstractMedium.eps_complex_to_nk


Material Library
----------------

.. toctree::

   material_library


Structures
==========

.. autosummary::
   :toctree: _autosummary/

   Structure

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   Structure.plot


Modes
=====

.. autosummary::
   :toctree: _autosummary/

   Mode


Sources
=======

.. autosummary::
   :toctree: _autosummary/

   VolumeSource
   PlaneWave
   ModeSource
   GaussianPulse

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   Source.geometry
   Source.plot
   Source.inside
   Source.intersections
   Source.intersects
   Source.intersects_plane
   Source.bounds
   Source.bounding_box
   Source.pop_axis
   Source.unpop_axis

Source Time Dependence
----------------------

.. autosummary::
   :toctree: _autosummary/

   GaussianPulse
   ContinuousWave
   SourceTime.amp_time
   SourceTime.plot
   SourceTime.frequency_range


Monitors
========

.. autosummary::
   :toctree: _autosummary/

   FieldMonitor
   FieldTimeMonitor
   FluxMonitor
   FluxTimeMonitor
   ModeMonitor

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   Monitor.geometry
   Monitor.plot
   Monitor.inside
   Monitor.intersections
   Monitor.intersects
   Monitor.intersects_plane
   Monitor.bounds
   Monitor.bounding_box
   Monitor.pop_axis
   Monitor.unpop_axis



Output Data
===========

.. autosummary::
   :toctree: _autosummary/

   SimulationData
   FieldData
   FluxData
   FluxTimeData
   ModeData

Methods
-------

.. autosummary::
   :toctree: _autosummary/

   SimulationData.plot_field
   Monitor.plot
   Monitor.inside
   Monitor.intersections
   Monitor.intersects
   Monitor.intersects_plane
   Monitor.bounds
   Monitor.bounding_box
   Monitor.pop_axis
   Monitor.unpop_axis

Tidy3dBaseModel
===============

.. autosummary::
   :toctree: _autosummary/

   components.base.Tidy3dBaseModel
   components.base.Tidy3dBaseModel.to_file
   components.base.Tidy3dBaseModel.from_file
   components.base.Tidy3dBaseModel.help

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

