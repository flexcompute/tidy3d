.. currentmodule:: tidy3d

Accelerated solver and convergence
----------------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   ChargeToleranceSpec

The **accelerated solver** is the default for every simulation
(``HeatChargeSimulation.use_accelerated_solver=True``). Setting
``use_accelerated_solver=False`` selects the **legacy solver**, and is only
permitted for charge simulations; heat and conduction always run on the
accelerated solver.

The two solvers measure convergence differently, so
**tolerance values are not transferable between them**. On the accelerated
solver the ``ChargeToleranceSpec`` field defaults are tight and **sufficient
for the majority of problems**. Start there and only adjust if a run does not
converge.

Per-solver tolerance guidance:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Setting
     - Accelerated (default)
     - Legacy
   * - ``rel_tol``
     - tight, at the defaults (``~1e-9`` to ``1e-10``)
     - loose (``~1e-5``)
   * - ``abs_tol``
     - *ignored on this path*
     - large (``~1e7`` to ``1e8``)

.. note::
   ``abs_tol`` only affects the legacy solver. On the accelerated (default)
   solver, ``rel_tol`` is the effective convergence knob.

Convergence parameters
^^^^^^^^^^^^^^^^^^^^^^^

Convergence on the accelerated solver is governed by a few fields on
``ChargeToleranceSpec`` and ``SteadyChargeDCAnalysis``. On this solver
``rel_tol`` is the effective tolerance (``abs_tol`` is ignored); the remaining
fields control how the solver steps toward the DC solution.

Parameters you can adjust:

- ``rel_tol`` (``ChargeToleranceSpec``, default ``1e-10``): relative convergence
  tolerance and the effective convergence criterion on the accelerated solver.
  The tight default is sufficient for most problems. Loosening it is **not** a
  way to fix non-convergence (see Troubleshooting below): it only stops the solve
  at a higher residual, i.e. a less accurate solution.
- ``max_iters`` (``ChargeToleranceSpec``, default ``120``): maximum number of
  nonlinear (Newton) iterations per bias step. If the convergence history shows
  the residual still decreasing when a bias reaches this cap, raise it.
- ``convergence_dv`` (``SteadyChargeDCAnalysis``, default ``1.0`` V): the bias
  step. The solver starts at zero bias and ramps to the requested voltage in
  ``convergence_dv`` increments, re-solving at each step. Lowering it takes
  smaller, more robust steps toward the target bias, at the cost of more steps
  and longer runtime.
- ``ramp_up_iters`` (``ChargeToleranceSpec``, default ``1``): number of
  iterations over which quantities such as doping are ramped up to their full
  values during start-up. Increasing it introduces doping more gradually, which
  can stabilize start-up on heavily doped devices.

Advanced controls (leave at their defaults):

These rarely need changing. Non-default values emit a warning and can cause long
runtimes or non-convergence:

- ``cfl_number`` (default ``1e9``)
- ``cfl_min`` (default ``None``)
- ``preconditioner_iterations`` (default ``50``)

Troubleshooting convergence (accelerated solver)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The defaults converge for most problems. If a charge run on the accelerated
solver does not converge, work through these in order:

#. **Look at the convergence history first.** Inspect
   ``sim_data.device_characteristics.dc_convergence`` -- its ``converged``,
   ``n_iters`` and ``residual_history`` fields (see
   :class:`SteadyConvergenceData`)::

       conv = sim_data.device_characteristics.dc_convergence
       conv.converged, conv.n_iters, conv.residual_history

   This tells you whether the residual was still decreasing (the solver simply
   needs more iterations) or had stagnated above the tolerance (it will not
   converge further without other changes).
#. **Residual still decreasing -- give it more iterations.** Raise ``max_iters``
   first. If the residual is still dropping when ``n_iters`` reaches the cap,
   also raise ``max_pseudo_steps``. A non-default ``max_pseudo_steps`` emits a
   warning about long runtimes and convergence; that warning is conservative and
   is expected in this case.
#. **A high-bias point fails or oscillates.** Lower ``convergence_dv`` so the
   solver approaches the target voltage in smaller bias steps.
#. **Start-up is unstable on a heavily doped device.** Increase ``ramp_up_iters``
   so doping is introduced more gradually.
#. **Only then, trade accuracy for runtime.** If the residual has stagnated above
   the tolerance and a less accurate solution is acceptable, loosen ``rel_tol``
   to stop at that residual -- but never down to legacy-style values such as
   ``1e-5``. Loosening ``rel_tol`` is not a convergence fix; it only stops the
   solve at a higher residual.

Migrating from the legacy solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the two solvers measure convergence differently, tolerance values are
not transferable: a configuration tuned for the legacy solver won't necessarily
work for the accelerated solver. In most cases ``td.ChargeToleranceSpec()`` (the
defaults) is enough.

- **Tighten** ``rel_tol`` back toward the defaults (``~1e-9`` to ``1e-10``). The
  loose ``rel_tol`` (``~1e-5``) used on the legacy solver is satisfied almost
  immediately on the accelerated solver and returns an unconverged result.
- **Drop** the inflated ``abs_tol``. It is ignored on the accelerated solver, so
  leaving it at the field default is fine.

The ``cfl_number``, ``cfl_min``, ``max_pseudo_steps`` and
``preconditioner_iterations`` controls apply only to the accelerated solver and
have no legacy counterpart, so there is nothing to migrate there; leave them at
their defaults.

Accelerated-only features
^^^^^^^^^^^^^^^^^^^^^^^^^

A few features are supported only on the accelerated solver, and requesting
``use_accelerated_solver=False`` while they are configured raises an error that
names the offending feature:

- ``MasettiMobility``
- SSAC ``at_voltages`` bias-point selection
