Simulation Tips for AI Agents
=============================

*Workflow patterns and runtime pitfalls for AI-assisted photonics simulation.*

This page collects practical workflow guidance that spans multiple Tidy3D classes and doesn't belong in any single docstring. For API-level guidance (which source to use, how to set up materials, grid resolution), see the class docstrings — they contain selection guides and tips under **Practical Advice** headings.

**Units**: length in micrometers, time in seconds, frequency in Hz.

.. contents:: On this page
   :local:
   :depth: 2


Documentation Lookups with MCP
------------------------------

The `FlexAgent MCP <flex_agent.html>`_ server is your primary source for API docs and examples. Always search it before writing simulation code.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Tool
     - Purpose
   * - ``search_flexcompute_docs``
     - Search all Flexcompute docs and examples
   * - ``fetch_flexcompute_doc``
     - Fetch a specific documentation page by URL

**Search patterns:**

- Device type: ``"ring resonator tidy3d"``, ``"grating coupler SOI"``
- API class: ``"ModeSource tidy3d"``, ``"FluxMonitor tidy3d"``
- Workflow: ``"parameter sweep batch tidy3d"``, ``"adjoint inverse design tidy3d"``

**Decompose the request before searching:**

1. **Device** — what structure? Search ``"{device_type} tidy3d"``
2. **FOM** — what to measure? Search ``"{monitor_type} tidy3d"``
3. **Workflow** — single run, sweep, EME, or inverse design?


Visualization-First Workflow
----------------------------

Never run a simulation you haven't visually inspected. Generate plots liberally and inspect them — this is the cheapest way to catch errors.

1. **Plot geometry** — ``sim.plot_eps(x=0)``, ``sim.plot_eps(y=0)``, ``sim.plot_eps(z=0)``. Check: waveguides extend through PML, sources in straight sections, monitors correctly placed, dimensions look right.
2. **Fix and re-plot** — if anything looks wrong, fix it and plot again. Do not skip to running the simulation.
3. **After running, inspect results** — are T values physical (at most 1.0)? Do field profiles look reasonable? Any shutoff warnings? Do sweep trends make sense?
4. **Stop if wrong** — do not continue to optimization or parameter sweeps on top of a broken setup.


Parameter Sweeps
----------------

Use ``web.run_async()`` to run multiple simulations in parallel:

.. code-block:: python

   sims = {f"width_{w:.3f}": make_sim(w) for w in np.linspace(0.4, 0.6, 11)}
   batch_data = web.run_async(sims)
   for name, sim_data in batch_data.items():
       T = sim_data["transmission"].flux.values


S-Parameter Extraction
----------------------

Use the ``ModalComponentModeler`` plugin for multi-port S-parameters:

.. code-block:: python

   from tidy3d.plugins.smatrix import ModalComponentModeler, Port

   import tidy3d.web as web

   ports = [
       Port(center=(-5, 1, 0), size=(0, 2, 2), mode_spec=td.ModeSpec(), direction="+", name="in1"),
       Port(center=(5, 1, 0), size=(0, 2, 2), mode_spec=td.ModeSpec(), direction="-", name="out1"),
   ]
   modeler = ModalComponentModeler(simulation=sim, ports=ports, freqs=freqs)
   modeler_data = web.run(modeler)
   smatrix = modeler_data.smatrix()
   S21 = smatrix.loc[dict(port_in="in1", mode_index_in=0, port_out="out1", mode_index_out=0)]


Resonator Q Factor
------------------

Three methods:

1. **Spectral fitting** — fit a Lorentzian to the transmission dip near resonance.
2. **Time-domain decay** — place a ``FieldTimeMonitor`` at the cavity center, fit exponential decay to extract ``Q = omega * tau / 2``.
3. **ResonanceFinder plugin** — ``from tidy3d.plugins.resonance import ResonanceFinder``.


EME Simulations
---------------

EME (Eigenmode Expansion) is efficient for devices with slowly varying cross-sections — tapers, transitions, periodic structures.

.. code-block:: python

   eme_sim = td.EMESimulation(
       size=(10, 4, 4),
       structures=[...], monitors=[...],
       grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=wavelength),
       axis=0,
       eme_grid_spec=td.EMEUniformGrid(
           num_cells=20, mode_spec=td.EMEModeSpec(num_modes=10)
       ),
       freqs=[freq0],
   )
   eme_data = web.run(eme_sim, task_name="eme_taper")

EME can sweep device length without re-solving modes using ``sweep_spec=td.EMELengthSweep(scale_factors=np.linspace(0.5, 2.0, 20))``.
For broadband EME, prefer listing the target frequencies directly in ``EMESimulation.freqs`` and using ``td.EMEModeSpec(interp_spec=...)`` to tune the interpolation cost/accuracy tradeoff; ``EMEFreqSweep`` is deprecated.
If EME drops weakly increasing modes because of tiny negative imaginary effective indices, set ``td.EMEModeSpec(increasing_mode_tolerance=...)``. A mode is dropped only when ``Im(n_eff)`` is more negative than that tolerance, so starting with a small value such as ``1e-6`` is usually appropriate; the default ``0.0`` preserves the previous behavior.
For bent EME cells, choose ``td.EMEModeSpec(bend_medium_frame="global")`` when material axes are fixed in physical space, and ``"co_rotating"`` when the material profile should bend with the local cross-section. Bent custom media, including ``td.CustomAnisotropicMedium``, currently require ``"co_rotating"``, and global-frame anisotropic bends, including reciprocal ``td.FullyAnisotropicMedium``, may need multiple cells instead of reusing one bent cell, so check convergence versus the number of EME cells.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Use EME when...
     - Use FDTD when...
   * - Device is long (>20 wavelengths)
     - Complex 3D scattering
   * - Cross-section changes slowly
     - Broadband response needed
   * - Need length optimization
     - Time-domain effects matter


Inverse Design
--------------

Tidy3D uses **HIPS autograd**, not JAX. The old ``tidy3d.plugins.adjoint`` plugin has been removed. Regular ``td.Simulation`` and ``web.run()`` are differentiable — no special classes needed.

.. code-block:: python

   import autograd
   import autograd.numpy as anp
   # NOT: import jax
   # NOT: from tidy3d.plugins.adjoint import ...

For detailed patterns (filter and project, beta scheduling, learning rates, gradient sign conventions, traceable components, gotchas), see the autograd README in the Tidy3D source: ``tidy3d/plugins/autograd/README.md``.


Pre-Flight Checklist
--------------------

Run through before every simulation:

1. Waveguides extend through PML — use ``td.inf``
2. Source/monitor in straight sections — not at bends or junctions
3. Mode monitor sized 3-4x waveguide width
4. Grid resolves features — at least 15-20 cells across smallest feature
5. Transmission is physical — T at most 1.0
6. ``RunTimeSpec`` or sufficient ``run_time`` — no shutoff warnings


Common Failure Modes
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Symptom
     - Likely Cause
     - Fix
   * - T = 0
     - Source/monitor misaligned
     - Check geometry plot
   * - T > 1.0
     - Monitor direction wrong
     - Check ``direction`` parameter
   * - NaN
     - Simulation diverged
     - Check PML, resolution, dispersive materials
   * - Shutoff warning
     - ``run_time`` too short
     - Use ``RunTimeSpec`` with higher ``quality_factor``
   * - Wrong wavelength
     - ``n_eff`` guessed
     - Use ModeSolver to compute it
   * - Autograd TypeError
     - ``float()`` on traced param
     - Never cast traced params
   * - 0 gradient
     - Sim domain depends on params
     - Only geometry should be traced


Coding Discipline
-----------------

- **Never suppress warnings** — every Tidy3D warning tells you something is wrong. Read it, fix the root cause. Do not use ``warnings.filterwarnings("ignore")``.
- **Never use try/except to mask failures** — if ``web.run()`` fails, the error tells you what's wrong. Catching and ignoring it means proceeding with garbage.
- **Never guess physical parameters** — always compute ``n_eff``, grating periods, etc. using ModeSolver or equivalent. This is the number one source of wrong results.
- **Fix warnings, don't work around them** — search the MCP for the warning text, understand the cause, fix it, verify it's gone.
