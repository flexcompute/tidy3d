.. currentmodule:: tidy3d

Simulation
==========

Overview
--------

This page contains information on the different simulation types available in Tidy3D.

~~~~

Defining a FDTD Simulation
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.Simulation

~~~~

Submitting a Simulation
-----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.container.Job
   tidy3d.web.api.container.Job.run
   tidy3d.web.api.container.Job.estimate_cost
   tidy3d.web.api.container.Job.real_cost
   tidy3d.web.api.container.Job.upload
   tidy3d.web.api.container.Job.start
   tidy3d.web.api.container.Job.monitor

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.container.Job.download
   tidy3d.web.api.container.Job.load
   tidy3d.web.api.container.Job.delete
   tidy3d.web.api.container.Job.get_info
   tidy3d.web.api.container.Job.get_run_info
   tidy3d.web.api.container.Job.to_file


~~~~

Batch Jobs
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.container.Batch

~~~~

Other Simulation Types
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.mode.ModeSolver
   tidy3d.EMESimulation
   tidy3d.HeatSimulation
   tidy3d.HeatChargeSimulation

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.ComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModeler

~~~~

Logging
-------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.config.logging_level
   tidy3d.set_logging_file

~~~~

Additional Methods
------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.estimate_cost
   tidy3d.web.api.webapi.run
   tidy3d.web.api.webapi.upload
   tidy3d.web.api.webapi.start
   tidy3d.web.api.webapi.real_cost
   tidy3d.web.api.webapi.get_info
   tidy3d.web.api.webapi.monitor
   tidy3d.web.api.webapi.download
   tidy3d.web.api.webapi.load
   tidy3d.web.api.webapi.delete
   tidy3d.web.api.webapi.download_log
   tidy3d.web.api.webapi.download_json
   tidy3d.web.api.webapi.load_simulation
   tidy3d.web.api.asynchronous.run_async
   tidy3d.web.core.task_info.TaskInfo
   tidy3d.web.core.task_info.TaskStatus
