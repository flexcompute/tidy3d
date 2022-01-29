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


Sources
=======


Types of Sources
----------------

.. autosummary::
   :toctree: _autosummary/

   VolumeSource
   PlaneWave
   GaussianBeam
   ModeSource


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


Mode Specifications
===================

.. autosummary::
   :toctree: _autosummary/

   ModeSpec

Discretization
==============

.. autosummary::
   :toctree: _autosummary/

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
   ModeData

Individual Dataset
------------------

.. autosummary::
   :toctree: _autosummary/

   ScalarFieldData
   ScalarFieldTimeData
   FluxData
   FluxTimeData
   ModeAmpsData
   ModeIndexData

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
   tidy3d.web.get_info
   tidy3d.web.start
   tidy3d.web.monitor
   tidy3d.web.download
   tidy3d.web.load
   tidy3d.web.delete

Convenience for Single and Batch
--------------------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.Job
   tidy3d.web.Batch

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/

   tidy3d.web.task.Task
   tidy3d.web.task.TaskInfo
   tidy3d.web.task.TaskStatus


Plugins
=======

.. Dispersive Model Fitting Tool
.. -----------------------------

.. autosummary::
   :toctree: _autosummary/

   plugins.DispersionFitter
   plugins.ModeSolver
   plugins.Near2Far
   .. plugins.DispersionFitter.from_file
   .. plugins.DispersionFitter.fit
   .. plugins.DispersionFitter.plot

.. Mode Solver
.. -----------

.. .. autosummary::
..    :toctree: _autosummary/

..    plugins.ModeSolver
..    .. plugins.ModeSolver.solve
..    .. plugins.mode.mode_solver.ModeInfo

.. Near Field to Far Field Transformation
.. --------------------------------------

.. .. autosummary::
..    :toctree: _autosummary/

..    plugins.Near2Far
..    plugins.Near2Far.fields_cartesian
..    plugins.Near2Far.fields_spherical
..    plugins.Near2Far.power_cartesian
..    plugins.Near2Far.power_spherical
..    plugins.Near2Far.radar_cross_section

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
   tidy3d.components.pml.AbsorberSpec
   tidy3d.components.medium.AbstractMedium
   tidy3d.components.medium.DispersiveMedium
   tidy3d.components.geometry.Geometry
   tidy3d.components.geometry.Planar
   tidy3d.components.geometry.Circular
   tidy3d.components.source.SourceTime
   tidy3d.components.source.Source
   tidy3d.components.source.FieldSource
   tidy3d.components.source.AngledFieldSource
   tidy3d.components.monitor.Monitor
   tidy3d.components.data.Tidy3dData
   tidy3d.components.data.CollectionData
   tidy3d.components.data.AbstractFieldData
   tidy3d.components.data.MonitorData
   tidy3d.components.data.FreqData
   tidy3d.components.data.TimeData
   tidy3d.components.data.PlanarData
   tidy3d.components.data.AbstractFluxData
   tidy3d.components.data.AbstractScalarFieldData


   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime
   .. tidy3d.components.base.SourceTime




.. Tidy3D Base Model
.. -----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    plugins.DispersionFitter
..    plugins.ModeSolver
..    plugins.Near2Far


.. .. currentmodule:: tidy3d.components.pml

.. Absorber Specification
.. ----------------------

.. .. autopydantic_model:: AbsorberSpec
..    :inherited-members: Tidy3dBaseModel

.. .. currentmodule:: tidy3d.components.medium


.. Mediums
.. -------

.. .. autopydantic_model:: AbstractMedium
..    :inherited-members: Tidy3dBaseModel

.. .. currentmodule:: tidy3d.components.source


.. Source Time
.. -----------

.. .. autopydantic_model:: SourceTime
..    :inherited-members: Tidy3dBaseModel

.. .. currentmodule:: tidy3d.components.monitor

.. Monitor
.. -------

.. .. autopydantic_model:: Monitor
..    :inherited-members: Tidy3dBaseModel

.. .. currentmodule:: tidy3d

.. Types
.. -----

.. .. autosummary::
..    :toctree: _autosummary/

..    components.types.Size
..    components.types.Coordinate
..    components.types.Inf
..    components.types.NegInf

