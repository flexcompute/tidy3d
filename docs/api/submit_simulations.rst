
.. currentmodule:: tidy3d

Submitting Simulations
======================

Generic Web API
----------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.run
   tidy3d.web.api.webapi.upload
   tidy3d.web.api.webapi.estimate_cost
   tidy3d.web.api.webapi.real_cost
   tidy3d.web.api.webapi.get_info
   tidy3d.web.api.webapi.start
   tidy3d.web.api.webapi.monitor
   tidy3d.web.api.webapi.download
   tidy3d.web.api.webapi.load
   tidy3d.web.api.webapi.delete
   tidy3d.web.api.webapi.download_log
   tidy3d.web.api.webapi.download_json
   tidy3d.web.api.webapi.load_simulation
   tidy3d.web.api.asynchronous.run_async

Job and Batch Containers
-------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.container.Job
   tidy3d.web.api.container.Batch
   tidy3d.web.api.container.BatchData

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.core.task_info.TaskInfo
   tidy3d.web.core.task_info.TaskStatus


Mode Solver Web API
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.mode.run
   tidy3d.web.api.mode.run_batch
   tidy3d.web.api.mode.ModeSolverTask