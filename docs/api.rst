*************
API Reference
*************

.. currentmodule:: tidy3d

Defining Simulations
====================


Simulation Definition
---------------------

.. currentmodule:: tidy3d

.. autosummary::
   :toctree: _autosummary/

   Simulation
   simulation.grid

Structures
----------

.. autosummary::
   :toctree: _autosummary/

   Structure
   Structure.plot


Absorbing Boundaries
--------------------

.. autosummary::
   :toctree: _autosummary/

   PML
   StablePML
   Absorber
   AbsorberParams
   PMLParams

Mode Specification
------------------

.. autosummary::
   :toctree: _autosummary/

   Mode


Core Components
===============


Tidy3d Base Model
-----------------

.. currentmodule:: tidy3d.components.base

.. autosummary::
   :toctree: _autosummary/

   Tidy3dBaseModel
   Tidy3dBaseModel.export
   Tidy3dBaseModel.load
   Tidy3dBaseModel.help

FDTD Grid
---------

.. currentmodule:: tidy3d.components.grid

.. autosummary::
   :toctree: _autosummary/

   Coords
   FieldGrid
   YeeGrid
   Grid
   Grid.centers
   Grid.sizes
   Grid.yee


.. Absorbing Boundaries
.. --------------------

.. .. autosummary::
..    :toctree: _autosummary/

..    PML
..    StablePML
..    Absorber


.. Geometry
.. --------

.. .. autosummary::
..    :toctree: _autosummary/

..    Box
..    Sphere
..    Cylinder
..    PolySlab


.. Physical Objects
.. ----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    Structure
..    Medium
..    PoleResidue
..    Sellmeier
..    Debye
..    Lorentz
..    plugins.DispersionFitter
..    .. material_library


.. Monitors
.. --------

.. .. autosummary::
..    :toctree: _autosummary/

..    FieldMonitor
..    FieldTimeMonitor
..    FluxMonitor
..    FluxTimeMonitor
..    ModeMonitor
..    Mode


.. Simulation Output Data
.. ----------------------

.. .. autosummary::
..    :toctree: _autosummary/

..    SimulationData
..    SimulationData.export
..    SimulationData.load
..    FieldData
..    FluxData
..    FluxTimeData
..    ModeData


.. Submitting Simulations
.. ======================

.. .. currentmodule:: tidy3d

.. Web API
.. -------

.. .. autosummary::
..    :toctree: _autosummary/

..    web.upload
..    web.get_info
..    web.get_run_info
..    web.run
..    web.monitor
..    web.download
..    web.load_data
..    web.delete

.. Job Interface
.. -------------

.. .. autosummary::
..    :toctree: _autosummary/

..    web.Job

.. Batch Processing
.. ----------------

.. .. autosummary::
..    :toctree: _autosummary/

..    web.Batch

.. Info Containers
.. ---------------

.. .. autosummary::
..    :toctree: _autosummary/

..    web.task.Task
..    web.task.TaskInfo
..    web.task.TaskStatus


.. Plugins
.. =======

.. Dispersive Model Fitting Tool
.. -----------------------------

.. .. autosummary::
..    :toctree: _autosummary/

..    plugins.DispersionFitter
..    plugins.DispersionFitter.load
..    plugins.DispersionFitter.fit
..    plugins.DispersionFitter.plot

.. Mode Solver
.. -----------

.. .. autosummary::
..    :toctree: _autosummary/

..    plugins.ModeSolver
..    plugins.ModeSolver.solve
..    plugins.mode.mode_solver.ModeInfo

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

