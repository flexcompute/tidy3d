# Plan: Split modal S-matrix, RF terminal S-matrix, and microwave math

This document captures the concrete, step-by-step plan to refactor the existing `plugins/smatrix` and `plugins/microwave` into a cleaner separation of concerns, and to introduce a new `plugins/rf` plugin for RF/terminal workflows.

## Goals
- Keep `plugins/microwave/` math-only (no web, no cross-plugin deps; can depend on `components/*` and math libs).
- Make `plugins/smatrix/` generic and modal-only.
- Add `plugins/rf/` for RF/terminal S-matrix workflows (lumped ports, wave ports, antenna analysis, impedance via path integrals).

## User-facing high-level
- Modal workflows remain under `tidy3d.plugins.smatrix`.
- RF/terminal workflows move to `tidy3d.plugins.rf`.
- Microwave math stays under `tidy3d.plugins.microwave`.

## Step-by-step tasks

1) Create new package `plugins/rf/` [DONE]
- Create directories: `tidy3d/plugins/rf/{component_modelers,ports,data,analysis,utils}`
- Add `tidy3d/plugins/rf/__init__.py` exporting:
  - Modeler: `TerminalComponentModeler`
  - Ports: `LumpedPort`, `CoaxialLumpedPort`, `WavePort`
  - Data: `TerminalComponentModelerData`, `MicrowaveSMatrixData`, `PortDataArray`, `TerminalPortDataArray`
  - Utils: `ab_to_s`, `s_to_z`, `compute_power_wave_amplitudes`, `compute_power_delivered_by_port`, `compute_port_VI`, `compute_F`, `check_port_impedance_sign`
  - Run helpers: `create_batch`, `compose_terminal_modeler_data`, `run`

2) Copy RF/terminal code from `plugins/smatrix` → `plugins/rf` [DONE]
- Move files (copy then later delete originals):
  - `component_modelers/terminal.py` → `rf/component_modelers/terminal.py`
  - `ports/{base_terminal.py,base_lumped.py,rectangular_lumped.py,coaxial_lumped.py,wave.py,types.py}` → `rf/ports/`
  - `data/terminal.py` → `rf/data/terminal.py`
  - `analysis/{terminal.py,antenna.py}` → `rf/analysis/`
  - `data/data_array.py`: copy `PortDataArray` and `TerminalPortDataArray` → `rf/data/data_array.py` (keep `ModalPortDataArray` in smatrix)
  - `utils.py` → `rf/utils.py`
- Update imports inside moved files to `tidy3d.plugins.rf.*`. Keep microwave imports unchanged.
- Add `rf/run.py` with terminal-only batch composition and run helpers.
- Create `rf/data/modal.py` with `PortSimulationData` to avoid rf→smatrix dependency.

3) Purify `plugins/smatrix` (modal-only) [IN PROGRESS]
- `__init__.py`: remove RF exports. Optionally re-export RF items from `plugins.rf` with deprecation warnings for one minor release.
- `run.py`: remove terminal-related code; keep modal-only `create_batch`, `compose_component_modeler_data`, `run`.
- `component_modelers/types.py`: change to `ComponentModelerType = ComponentModeler`.
- `data/data_array.py`: keep only `ModalPortDataArray`.
- `component_modelers/base.py`: remove RF license validator and any terminal/wave types from annotations; keep modal helpers only.
- Delete `utils.py` or split into `utils_generic.py` with purely algebraic helpers if still needed by modal analysis.

4) Wire up RF package APIs [DONE]
- Ensure rf imports are consistent:
  - `rf/data/terminal.py` imports `rf.utils`, `rf.data.data_array`, `rf.analysis.*`, `rf.ports.types`, and `rf.component_modelers.terminal`.
  - `rf/component_modelers/terminal.py` imports `smatrix.component_modelers.base.AbstractComponentModeler` and `rf.ports.*`.
  - `rf/ports/wave.py` continues importing microwave math from `tidy3d.plugins.microwave`.
- `rf/__init__.py` exports all RF public surface.

5) Optional facade (ergonomics; no cross-coupling)
- Add `tidy3d/plugins/__init__.py:run_smatrix(modeler, path_dir=".")` that dispatches to either modal or RF run depending on the modeler type. Resolve imports inside the function.

6) Backward compatibility and deprecation [PENDING]
- For one minor release:
  - In `plugins/smatrix/__init__.py`, re-export RF items from `plugins.rf` and warn on access.
  - If `plugins.smatrix.run.run()` receives a terminal modeler, forward to `plugins.rf.run.run()` with a warning or raise a `TypeError` with a migration message.
- Update notebooks/examples with new RF imports; keep modal examples unchanged.

7) Testing [IN PROGRESS]
- Add/adjust unit tests for:
  - RF run path: batch composition, impedance, power-wave amplitudes, S↔Z conversion, antenna analysis.
  - Modal run path: existing tests remain valid.
- Verify `plugins/smatrix` has no imports from `plugins/microwave` or `plugins/rf`.
- CI: run modal and RF test suites independently.

8) Documentation
- Add “Plugins overview” page explaining the split.
- Migration guide with old→new import map and code diffs.
- Update docstrings to reference new paths.

9) Search/replace checklist
- Update references in codebase:
  - `tidy3d.plugins.smatrix.ports.(rectangular_lumped|coaxial_lumped|wave|base_*)` → `tidy3d.plugins.rf.ports.*`
  - `tidy3d.plugins.smatrix.component_modelers.terminal` → `tidy3d.plugins.rf.component_modelers.terminal`
  - `tidy3d.plugins.smatrix.data.terminal` → `tidy3d.plugins.rf.data.terminal`
  - `tidy3d.plugins.smatrix.data.data_array.(PortDataArray|TerminalPortDataArray)` → `tidy3d.plugins.rf.data.data_array.(...)`
  - `tidy3d.plugins.smatrix.utils` → `tidy3d.plugins.rf.utils`

10) Optional renames (follow-up)
- Consider `ComponentModeler` → `ModalComponentModeler`, `Port` → `ModalPort` with deprecations to improve clarity.

11) Rollout
- PR 1: Add `plugins/rf` package (copied code), import updates, RF tests.
- PR 2: Purify `plugins/smatrix` and add deprecation shims; update notebooks/docs.
- PR 3: Remove shims in next minor.