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
- ``convergence_dv`` (``SteadyChargeDCAnalysis``, default ``1.0`` V): the
  maximum bias step within a voltage sweep. On the accelerated solver it
  applies only to multi-voltage sweeps: where the gap between consecutive
  sweep voltages exceeds ``convergence_dv``, intermediate warm-start bias
  points are inserted (and excluded from the output); a single requested
  voltage is solved directly, so this field has no effect. Lowering it takes
  smaller, more robust steps between sweep points, at the cost of more solves
  and longer runtime. The legacy solver instead ramps every requested bias
  from zero in ``convergence_dv`` increments.
- ``ramp_up_iters`` (``ChargeToleranceSpec``, default ``1``): number of
  iterations over which quantities such as doping are ramped up to their full
  values during start-up. Increasing it introduces doping more gradually, which
  can stabilize start-up on heavily doped devices.
- ``linear_rel_tol`` (``ChargeToleranceSpec``, default ``1e-2``): the factor by
  which the iterative linear solver reduces the residual of the linear system
  at each nonlinear iteration. A smaller value makes each iteration more
  accurate and more expensive; see Troubleshooting for when to lower it. It has
  no effect with ``linear_solver='direct'``.
- ``max_iters`` (``ChargeToleranceSpec``, default ``120``): the iteration
  budget of the iterative linear solver at each nonlinear iteration. It does
  not limit the nonlinear iterations, which ``max_pseudo_steps`` caps. It has
  no effect with ``linear_solver='direct'``.

Advanced controls (leave at their defaults):

These rarely need changing. Non-default values emit a warning and can cause long
runtimes or non-convergence:

- ``cfl_number`` (default ``1e9``) -- the upper bound of the adaptive CFL
  controller.
- ``cfl_min`` (default ``None``, meaning ``1``) -- its lower bound.
- ``max_pseudo_steps`` (default ``60``) -- the maximum number of nonlinear
  iterations per bias; see Troubleshooting step 2.
- ``preconditioner_iterations`` (default ``50``)

Troubleshooting convergence (accelerated solver)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The defaults converge for most problems. If a charge run on the accelerated
solver does not converge, work through these in order:

#. **Look at the convergence history first.** Inspect
   ``sim_data.device_characteristics.dc_convergence`` -- its ``converged``,
   ``n_iters``, ``residual_history`` and ``cfl_history`` fields (see
   :class:`SteadyConvergenceData`)::

       conv = sim_data.device_characteristics.dc_convergence
       conv.converged, conv.n_iters, conv.residual_history, conv.cfl_history

   ``residual_history`` tells you whether the residual was still decreasing (the
   solver simply needs more iterations) or had stagnated above the tolerance (it
   will not converge further without other changes).

   ``cfl_history`` tells you how the pseudo-time damping evolved. The solver
   damps the Newton iteration by adding a pseudo-time term that scales as
   ``1 / CFL``; the controller adapts the CFL between ``cfl_min`` and
   ``cfl_number`` as the solve proceeds. At a non-converged bias::

       stuck = conv.cfl_history.sel(v=0.7)

   * **CFL at its lower bound** (``cfl_min``, default ``1``): the controller
     cut the CFL as far as it is allowed. It lowers the CFL for several
     reasons, so this says where the controller ended, not why.
   * **CFL at its upper bound** (``cfl_number``, default ``1e9``): the
     pseudo-time term was at its smallest allowed value. If the residual was
     still decreasing, go to step 2; if it had stagnated, go to step 3, and if
     that does not help, look at the setup -- mesh, doping, operating point.
   * **CFL still moving between the two bounds**: the controller was still
     adapting. Read ``residual_history`` and follow steps 2 to 4 as usual.

   With ``cfl_min == cfl_number`` the CFL is held constant and the trace says
   nothing about the bounds.

   A drop between consecutive samples means the controller backed off; only
   one sample per ``pseudo_step`` is recorded. Traces are not independent
   across biases: the first bias solved starts at ``1`` before any bound is
   applied, so its first sample can lie outside ``[cfl_min, cfl_number]`` on
   either side; each later bias starts from the CFL carried over from the
   previous one, brought within range and lowered after a bias that had to
   back off.

   When the trace for a non-converged bias conclusively ends on one bound, the
   non-convergence warning issued when the simulation finishes names it.
#. **Residual still decreasing -- give it more iterations.** If the residual is
   still dropping when ``n_iters`` reaches ``max_pseudo_steps``, raise
   ``max_pseudo_steps``. A non-default ``max_pseudo_steps`` emits a warning
   about long runtimes and convergence; that warning is conservative and is
   expected in this case.
#. **The residual oscillates or stagnates on the iterative linear solver.**
   Lower ``linear_rel_tol`` to ``1e-4`` or ``1e-5`` so that each nonlinear
   iteration is solved more accurately. This helps most where one
   ``component`` of ``residual_history`` stays orders of magnitude below the
   others, as in undoped or lightly doped regions, heterojunctions and
   degenerately doped junctions with band-to-band tunneling.
   ``linear_solver='direct'`` is the alternative when memory allows.
#. **A high-bias point in a sweep fails or oscillates.** Lower
   ``convergence_dv`` so the solver approaches it through smaller intermediate
   bias steps. For a single requested voltage ``convergence_dv`` has no
   effect; turn the simulation into a sweep by adding intermediate voltages
   instead.
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

The ``cfl_number``, ``cfl_min``, ``max_pseudo_steps``,
``preconditioner_iterations`` and ``linear_rel_tol`` controls apply only to the
accelerated solver and have no legacy counterpart, so there is nothing to
migrate there; start from their defaults.

Accelerated-only features
^^^^^^^^^^^^^^^^^^^^^^^^^

A few features are supported only on the accelerated solver, and requesting
``use_accelerated_solver=False`` while they are configured raises an error that
names the offending feature:

- ``MasettiMobility``
- ``AlloyMobility``
- ``CanaliFieldDependence`` (high-field velocity saturation, attached to any
  mobility model through ``field_dependence``)
- ``TransferredElectronFieldDependence`` (high-field negative differential
  mobility, attached to any mobility model through ``field_dependence``)
- ``LombardiSurfaceDependence`` (surface mobility degradation, attached to
  ``CaugheyThomasMobility``, ``MasettiMobility`` and ``AlloyMobility`` through
  ``surface_dependence``)
- SSAC ``at_voltages`` bias-point selection
- ``SteadyChargeResidualMonitor`` (per-node residual debug monitor)
- ``SteadyGenerationRecombinationMonitor`` (per-node generation-recombination
  rates)
- Schottky contacts (``VoltageBC`` with ``model="schottky_mott"``); supported
  by the charge solver only -- in conduction simulations the ``model`` flag
  has no effect and the boundary applies the plain potential
