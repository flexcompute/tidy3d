.. currentmodule:: tidy3d

Submitting Simulations
======================

Generic Web API
----------------

Core Workflow
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.run
   tidy3d.web.api.webapi.upload
   tidy3d.web.api.webapi.start
   tidy3d.web.api.webapi.monitor
   tidy3d.web.api.webapi.download
   tidy3d.web.api.webapi.load
   tidy3d.web.api.asynchronous.run_async

Download Utilities
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.download_json
   tidy3d.web.api.webapi.download_hdf5
   tidy3d.web.api.webapi.download_log
   tidy3d.web.api.webapi.load_simulation

Task Information
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.get_info
   tidy3d.web.api.webapi.get_run_info
   tidy3d.web.api.webapi.get_tasks

Cost Estimation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.estimate_cost
   tidy3d.web.api.webapi.real_cost

Task Management
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.delete
   tidy3d.web.api.webapi.delete_old
   tidy3d.web.api.webapi.abort

Account and System
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.webapi.account
   tidy3d.web.api.webapi.test

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
