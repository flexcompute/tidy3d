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

   web.run
   web.upload
   web.start
   web.monitor
   web.download
   web.load
   web.api.asynchronous.run_async

Download Utilities
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.download_json
   web.download_log
   web.load_simulation

Task Information
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.get_info
   web.get_run_info
   web.get_tasks

Cost Estimation
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.estimate_cost
   web.real_cost

Task Management
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.delete
   web.delete_old
   web.abort

Account and System
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.account
   web.test
   web.refresh_licenses

Job and Batch Containers
-------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.Job
   web.Batch
   web.BatchData

Information Containers
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.core.task_info.TaskInfo


Mode Solver Web API
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.api.mode.run
   web.api.mode.run_batch
   web.api.mode.ModeSolverTask
