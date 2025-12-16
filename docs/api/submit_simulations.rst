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

   tidy3d.web.run
   tidy3d.web.upload
   tidy3d.web.start
   tidy3d.web.monitor
   tidy3d.web.download
   tidy3d.web.load
   tidy3d.web.api.asynchronous.run_async

Download Utilities
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.download_json
   tidy3d.web.download_hdf5
   tidy3d.web.download_log
   tidy3d.web.load_simulation

Task Information
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.get_info
   tidy3d.web.get_run_info
   tidy3d.web.get_tasks

Cost Estimation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.estimate_cost
   tidy3d.web.real_cost

Task Management
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.delete
   tidy3d.web.delete_old
   tidy3d.web.abort

Account and System
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.account
   tidy3d.web.test

Job and Batch Containers
-------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.Job
   tidy3d.web.Batch
   tidy3d.web.BatchData

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.core.task_info.TaskInfo


Mode Solver Web API
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.web.api.mode.run
   tidy3d.web.api.mode.run_batch
   tidy3d.web.api.mode.ModeSolverTask
