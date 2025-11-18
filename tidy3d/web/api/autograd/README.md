# Web Autograd Orchestration Guide

This document is scoped to the modules under `tidy3d/web/api/autograd/`. It explains how the Python web client wraps Tidy3D simulations with autograd primitives, when solver interactions are delegated to the cloud vs. run locally, and how artifacts move through the pipeline. For the geometry/material derivative contract see `tidy3d/components/autograd/README.md`.

## Goals & Guiding Principles
- Keep the user-facing `web.run` / `web.run_async` signatures stable while seamlessly switching to autograd-aware logic when tracers exist.
- Make primitive boundaries (`@primitive` functions) explicit so autograd can re-use them across graph replays while we retain full control of solver submissions.
- Centralize every network side-effect (uploads, downloads, parent task wiring) in one place so it is simple to audit or mock.
- Ensure local-gradient mode honors every override in `config.adjoint.*` without leaking those knobs to the remote execution path.

## Module Map
| Module | Responsibility |
| --- | --- |
| `autograd.py` | User entry points (`run`, `run_async`), autograd primitives/VJPs, glue code that caches forward data and dispatches adjoint batches. |
| `forward.py` | `setup_fwd` (monitor injection using `Simulation._with_adjoint_monitors`) and `postprocess_fwd` (splits solver output, caches originals, returns tracer-shaped data). |
| `backward.py` | `setup_adj` (build adjoint sims) and `postprocess_adj` (assemble `DerivativeInfo` chunks, call `_compute_derivatives`). |
| `engine.py` | Abstractions over `tidy3d.web.api.container.Job`/`Batch` that know how to set `simulation_type`, upload tracer keys, and rewire task paths. |
| `io_utils.py` | Upload/download helpers. Handles conversion between `AutogradFieldMap` (dict) and `FieldMap` (Pydantic model) for serialization. |
| `constants.py` | Shared aux-data keys and filenames; must stay aligned with backend expectations. |
| `utils.py` | Math helpers for field products (E·E†, D·D†, etc.). |

## Execution Flow

The following steps outline how a user's request transforms into autograd primitives:

1.  **Parameter Tracing**: When user parameters flow through `autograd.numpy`, they are wrapped in `Box` objects (specifically `TidyArrayBox`). These boxes propagate into structure fields (permittivity, geometry, etc.).
2.  **Simulation Creation**: The user creates a `Simulation` containing these traced structures.
3.  **Run Invocation**: `web.run()` validates traced fields via `is_valid_for_autograd()` (checking for `_freqs_adjoint` and structure limits) and delegates to `autograd.py:_run`, which calls `_run_primitive`.
4.  **Forward Pass (`_run_primitive`)**:
    *   Extracts traced fields into an `AutogradFieldMap`.
    *   Runs the forward simulation (locally or remotely) with added gradient monitors.
    *   Returns the `AutogradFieldMap` to the autograd graph, caching the full simulation data in `aux_data`.
5.  **Backward Pass (`_run_bwd`)**:
    *   Triggered by `ag.grad`.
    *   Uses `setup_adj` to create adjoint sources from the incoming gradient (VJP).
    *   Runs adjoint simulations (reversed source injection).
    *   Computes derivatives via `postprocess_adj` and returns them to the autograd graph.

## Primitive & VJP Strategy
- `_run_primitive` / `_run_async_primitive` are the only autograd primitives. They always take the stripped `AutogradFieldMap` first (positional arg) so autograd registers a dependency.
- `defvjp(...)` registers `_run_bwd` / `_run_async_bwd`. The closures capture `aux_data` containing the serialized forward simulation, tracer keys, and optionally monitor data.
- Primitives are side-effectful: they upload simulations (remote mode) or run them directly (local mode). VJPs reverse that process by constructing adjoint simulations based on upstream gradients.
- Never call solver APIs inside `setup_run`/`postprocess_run`. Those helpers must stay pure so they can run inside autograd tracing without triggering uploads.

## Local vs. Remote Gradients

| Mode | Trigger | Forward path | Adjoint path | Config handling |
| --- | --- | --- | --- | --- |
| `local_gradient=True` | Explicit `web.run(..., local_gradient=True)` or `config.adjoint.local_gradient=True`. | Run combined sim locally (with adjoint monitors) via `_run_tidy3d`. | Build adjoint sims locally, batch-run via `_run_async_tidy3d`, read `sim_data_fwd` from `aux_data`. | All `config.adjoint.*` overrides apply (monitor spacing, chunk sizes, dtype, directories). |
| Remote (default) | `local_gradient=False`. | Upload `simulation_type="autograd_fwd"`. `engine.upload_sim_fields_keys` pushes the tracer key file. | Build adjoint sims with `simulation_type="autograd_bwd"`, link the forward task ID as a parent, download `autograd_sim_vjp.hdf5` via `_get_vjp_traced_fields`. | Backend enforces its defaults; only `max_num_adjoint_per_fwd` (argument or config default) and tracer-count limits are honored client-side. |

### Backend Execution Details (Remote Mode)

When running remotely, the backend handles the heavy lifting:

*   **Forward (`autograd_fwd`)**:
    *   Backend loads `TracerKeys` and adds gradient monitors.
    *   Simulation runs; forward fields are saved to `autograd_fwd_data_file`.
    *   Original user data is separated and returned normally.

*   **Backward (`autograd_bwd`)**:
    *   Adjoint simulation runs with reversed sources.
    *   Backend loads original data, forward fields, and tracer keys.
    *   `postprocess_adj` computes gradients (combining forward and adjoint fields).
    *   Only the VJP result (`autograd_sim_vjp.hdf5`) is saved and made available for download.

## Artifact Lifecycle & Data Flow

```
Frontend (web.run with traced params):
1. setup_run() extracts AutogradFieldMap
2. Upload TracerKeys → backend
3. Run forward simulation (type="autograd_fwd")
   
Backend (forward):
1. Read TracerKeys -> Add gradient monitors
2. Run FDTD simulation -> Save forward fields

Frontend (VJP computation):
1. setup_adj() creates adjoint sources from output gradients
2. Run adjoint simulations (type="autograd_bwd")

Backend (backward):
1. Load forward data + original data + TracerKeys
2. Run adjoint FDTD simulation
3. postprocess_adj() computes gradients -> Save VJP data

Frontend (completion):
1. Download VJP file -> Return gradients to autograd
```

1. **Tracer keys** (`autograd_sim_fields_keys.hdf5`): produced from the stripped field map; uploaded before the forward run when remote.
2. **Forward data** (`aux_data[AUX_KEY_SIM_DATA_ORIGINAL]`, `AUX_KEY_SIM_DATA_FWD`): cached in memory so VJPs can rehydrate `SimulationData` without hitting disk. Note: `forward.py` populates these using internal string literals that match `constants.py` values (`"sim_data"`, `"sim_data_fwd_adjoint"`). Remote mode stores only the original data; local mode keeps both.
3. **Adjoint outputs** (`output/autograd_sim_vjp.hdf5`): downloaded and translated back into `AutogradFieldMap` instances. `io_utils.get_vjp_traced_fields` caches the result in `web.cache` to avoid re-downloading on repeated VJP calls.

## Error Handling & Limits
- `is_valid_for_autograd` enforces: simulation instance is `td.Simulation`, traced structures exist, `simulation._freqs_adjoint` is non-empty, and `config.adjoint.max_traced_structures` is not exceeded.
- `_setup_adj_impl` raises `AdjointError` when requested adjoint batches exceed `max_num_adjoint_per_fwd`. Surface errors (e.g., NaNs in VJP inputs) are caught before solver submission to avoid burning unnecessary cloud time.
- All uploads reuse verbose flags passed to `web.run`, so CLI users can watch transfer progress.

## Testing Hooks
- The suite under `tests/test_components/autograd/` emulates the web layer by patching `_run_tidy3d`, `_run_async_tidy3d`, `upload_sim_fields_keys`, and `_get_vjp_traced_fields`. Keep those functions thin and import-safe so tests can monkeypatch them without circular imports.
- When exposing new artifacts or aux-data keys, add fixtures to `tests/utils.py` so the emulation harness remains authoritative.

## Keeping Docs in Sync
- Update this README whenever `tidy3d/web/api/autograd/` gains new modules or changes how it talks to the solver.
- Mirror conceptual changes (e.g., new artifact types, batching semantics) in `tidy3d/components/autograd/README.md` so contributors see consistent narratives regardless of entry point.
