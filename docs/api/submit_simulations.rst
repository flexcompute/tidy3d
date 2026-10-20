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
   web.run_async

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
   web.execution_flexcredit_status
   exceptions.FlexCreditLimitExceededError

Task Management
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.delete
   web.delete_old
   web.abort
   web.move
   web.duplicate
   web.list_workspaces
   web.list_users
   web.get_activity

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
   web.WorkspaceInfo
   web.UserInfo
   web.ActivityEntry

Mode Solver Web API
--------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   web.api.mode.run
   web.api.mode.run_batch
   web.api.mode.ModeSolverTask

Shared workspace
----------------

Shared Workspace is available to members of a company with sharing enabled.
Contact Flexcompute support to set up access for your company or account.
Use ``web.list_workspaces()`` to check which workspaces are available to you.

New cloud tasks use your private workspace by default. To submit to your
company's shared workspace, set ``td.config.workspace.default`` to ``"shared"``::

   import tidy3d as td
   from tidy3d import web

   with td.config as scoped_config:
       scoped_config.workspace.default = "shared"
       data = web.run(sim, task_name="shared-example", folder_name="Team")

This setting also applies to ``web.upload``, ``Job``, ``Batch``, ``run_async``,
remote mode solving, the CLI, and plugins that submit cloud tasks. The previous
default is restored when the ``with`` block ends. Private and shared folders
with the same name are independent.

To save the default workspace, add the following to ``config.toml``::

   [workspace]
   default = "shared"

The equivalent environment variable is ``TIDY3D_WORKSPACE__DEFAULT=shared``.

Selecting an unavailable shared workspace raises an error; it never uses your
private workspace instead.

Move or duplicate tasks by specifying their destination::

   web.move([task_id_a, task_id_b], folder_name="Team", workspace="shared")
   copies = web.duplicate(task_id_a, folder_name="Experiments", workspace="private")

``folder_name`` and ``workspace`` select the destination. The destination folder
is created if needed. A list can contain tasks from either workspace; their
current locations are resolved automatically. If ``workspace`` is omitted,
these helpers use ``td.config.workspace.default``.

Each task belongs to a project, which can contain multiple saved revisions:

* ``move`` moves the entire project, including all revisions. A project is moved
  once even if several input task IDs belong to it. Projects already in the
  destination folder are left alone. Moving between workspaces logs a warning
  because it changes who can see the project and removes existing share links.
* ``duplicate`` copies the revision selected by each task ID and returns new
  task IDs in input order. Copying within the same folder is supported, and
  repeated input IDs create separate copies. Copies have the destination
  workspace's visibility; the original tasks stay in place.

Source tasks are checked before moving or copying. Operations are processed
sequentially: a failure can leave earlier moves or copies completed, and the
error reports those results. A lost connection can also leave the last
operation completed. These operations are not automatically retried; check
the error details and destination before retrying.

Cross-workspace moves and copies run in the background. Returning from the call
does not mean the files have arrived. Transfers may take several minutes and
time out after 30 minutes. Use ``web.get_info(task_id).operationStatus`` to inspect
``moving`` / ``move_completed`` / ``move_timeout`` or
``copying`` / ``copy_completed`` / ``copy_timeout`` when reported by the service.
Moves are rejected during preflight when a source reports an operation in progress.
If optional project metadata is unavailable, ``get_info`` warns and returns the
task details with the available fields; move, copy, and membership operations
still require their source metadata.

The Python client never takes an edit lock. Moving or copying a project being
edited may fail with an error identifying the editor. A copy may temporarily
show a copying status on the source project; its simulation data is preserved.

Inspect workspaces, members, and activity::

   workspaces = web.list_workspaces()
   users = web.list_users(task_id_a)
   folder_activity = web.get_activity(folder_name="Team", workspace="shared")
   task_activity = web.get_activity(task_id=task_id_a)

``get_activity`` requires exactly one of ``folder_name`` or ``task_id``. Its
workspace argument applies only to folders and defaults to ``"shared"``.
With ``task_id``, it returns activity for the containing project across all
revisions. Activity is available for shared workspaces; unsupported requests
raise ``NotImplementedError``.
