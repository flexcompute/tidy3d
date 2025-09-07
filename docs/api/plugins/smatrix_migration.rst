.. _smatrix_migration:

v2.10 Refactor Migration
----------------------

In version ``v2.10.0rc1``, ``smatrix`` plugin classes were refactored to improve web and GUI support for RF capabilities. This guide helps you update your scripts to the new, more robust API.

Key Changes
~~~~~~~~~~~

*   **Rename**: ``ComponentModeler`` has been renamed ``ModalComponentModeler``.
*   **Web Interface**: Web parameters like ``verbose`` and ``path_dir`` are now passed to ``tidy3d.web.run()`` instead of the modeler class constructor.
*   **Immutable Data Structures**: Modeler classes are now immutable. ``ComponentModelerData`` classes hold simulation results, separating data from the modeler definition.
*   **Explicit Method Calls**: Running simulations and fetching results are now done through explicit function calls like ``tidy3d.web.run()`` and ``tidy3d.web.load()``.

Updated Workflow
~~~~~~~~~~~~~~~~~

The new workflow is more explicit and aligns with the general ``tidy3d`` API.

**Before (Old API):**

.. code-block:: python

    import tidy3d.plugins.smatrix as sm

    # Modeler class was mutable and included web parameters
    tcm = sm.TerminalComponentModeler(
        simulation=sim,
        ports=[LP1, LP2],
        freqs=freqs,
        verbose=True,
        path_dir="data",
    )
    # The run method was part of the modeler class
    s_matrix = tcm.run()

**After (New API):**

.. code-block:: python

    import tidy3d.web as web
    import tidy3d.plugins.smatrix as sm

    # Modeler class is now immutable and cleaner
    tcm = sm.TerminalComponentModeler(
        simulation=sim,
        ports=[LP1, LP2],
        freqs=my_freqs,
    )
    # Use web.run to execute the simulation
    modeler_data = web.run(tcm, verbose=True, path="data/modeler_data.hdf5")
    s_matrix = modeler_data.smatrix()

.. note::
    The S-matrix is now computed from the ``ComponentModelerData`` objects, so it is accessed via the ``.smatrix()`` method.

Cost Estimation
~~~~~~~~~~~~~~~

Cost estimation is now done by uploading the modeler to the web API.

**Before:**

.. code-block:: python

    est_flex_credits = tcm.estimate_cost()
    real_flex_credits = tcm.real_cost()

**After:**

.. code-block:: python

    task_id = web.upload(tcm)
    est_flex_credits = web.estimate_cost(task_id)
    # After the run is complete
    real_flex_credits = web.real_cost(task_id)

Data Handling
~~~~~~~~~~~~~

The new API introduces immutable data containers for simulation results, ensuring that your data is more predictable and easier to manage.

*   ``TerminalComponentModeler`` returns a ``TerminalComponentModelerData`` object.
*   ``ModalComponentModeler`` returns a ``ModalComponentModelerData`` object.

These data objects contain the S-matrix, port impedance, and other relevant results.

**Before:**

.. code-block:: python

    # batch_data was a mutable property of the modeler
    tcm_batch = tcm.batch_data
    sim_data = tcm_batch["smatrix_LP1"]

**After:**

.. code-block:: python

   # web.run returns an immutable data object
   modeler_data = web.run(tcm)
   # Access simulation data for each port
   sim_data = modeler_data.data["smatrix_LP1"]


Migration Utilities
~~~~~~~~~~~~~~~~~~~

To ease the transition, we provide utilities that mimic the old workflow.

.. warning::
    These migration helpers are temporary and will be deprecated in a future release.

- **Run a batch of simulations**:
  If you prefer to manage the batch run yourself, you can create and run a batch explicitly.

  .. code-block:: python

      from tidy3d.plugins.smatrix.run import create_batch

      batch = create_batch(modeler=my_modeler)
      batch_data = batch.run()

- **Compose data from a batch**:
  If you have a :class:`tidy3d.web.BatchData` object from a manual run, you can still create the corresponding `ModalComponentModelerData` or `TerminalComponentModelerData` object.

  .. code-block:: python

      from tidy3d.plugins.smatrix.run import compose_modeler_data_from_batch_data

      modeler_data = compose_modeler_data_from_batch_data(
          modeler=my_modeler, batch_data=batch_data
      )

- **Legacy run command**:
  The `run` command from the old API is available as a legacy helper.

  .. code-block:: python

     from tidy3d.plugins.smatrix.run import run

     modeler_data = run(my_modeler)


API Reference
~~~~~~~~~~~~~

For more details, see the API documentation for the new classes and functions:

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.ModalComponentModeler
   tidy3d.plugins.smatrix.ModalComponentModelerData
   tidy3d.plugins.smatrix.TerminalComponentModeler
   tidy3d.plugins.smatrix.TerminalComponentModelerData
   tidy3d.SimulationMap
   tidy3d.SimulationDataMap
   tidy3d.plugins.smatrix.run.run
   tidy3d.plugins.smatrix.run.create_batch
   tidy3d.plugins.smatrix.run.compose_modeler_data_from_batch_data
   tidy3d.plugins.smatrix.run.compose_modeler_data
