Inverse Design Scaffold (invdes2)
=================================

Overview
--------
This module provides a lightweight, composable scaffold for photonic inverse design built on Tidy3D. It separates concerns into:

- Design regions: turn parameter vectors into `td.Structure`s.
- Device specs: combine a base `td.Simulation` with parameterized structures and score it with metrics.
- Inverse design orchestrator: build, run, and aggregate multiple device scenarios.
- Metrics: compute scalar objectives from monitor data.
- Optimizer spec: a hyperparameter container to integrate with your optimizer of choice.

High-level flow
---------------
```
  parameters (list[list[np.ndarray]])
           │
           ▼
  InverseDesign.get_simulations(params)
           │          ┌─────────────────────────────────────────────────────┐
           ├─ for each device_spec:                                         │
           │     DeviceSpec.get_simulation(device_params)                   │
           │       ├─ for each region:                                      │
           │       │     DesignRegion.to_structure(region_params)           │
           │       └─ append structures to base simulation                  │
           └────────────────────────────────────────────────────────────────┘
           │
           ▼
  InverseDesign.run_simulations({name: Simulation})  ──> Tidy3D Web (async)
           │
           ▼
  InverseDesign.get_metric(batch_data)  ── sum over devices
           │
           ▼
  scalar objective (float)
```

Key concepts and APIs
---------------------
Design regions (`design_region.py`)
- `DesignRegion` (ABC): contract for parameterized geometry providers.
  - `parameter_shape: int`: total number of parameters (flattened).
  - `to_structure(params: np.ndarray) -> td.Structure`: maps a 1D parameter vector to a `td.Structure`.
  - `ones(**kwargs) -> np.ndarray`: helper to create a correctly sized parameter vector.
- `TopologyDesignRegion(DesignRegion)`: pixellated permittivity grid inside a `td.Box`.
  - `size, center, pixel_size`: define a voxelized grid; `shape_3d` is computed from `(size / pixel_size)`.
  - `eps_bounds`: available for future projection/constraints (currently not applied).

Device specs (`device_spec.py`)
- `DeviceSpec`: describes a single device scenario.
  - `simulation: td.Simulation`: base sim.
  - `design_regions: list[DesignRegion]`: ordered list; each consumes a parameter array.
  - `metrics: list[Metric]`: weighted scalar contributions.
  - `name: str`: unique task name for batch runs.
  - `get_simulation(params)`: appends one `td.Structure` per region to the base sim, returns a new `td.Simulation`.
  - `run_simulation(sim)`: executes the simulation via `tidy3d.web.run(sim, task_name=name)`.
  - `get_metric(sim_data)`: for each metric, pulls `sim_data[metric.monitor_name]`, applies `metric.evaluate`, multiplies by `metric.weight`, and sums.
  - `parameter_shape`: list of per-region sizes.
  - `ones(**kwargs)`: list of correctly sized 1D arrays for each region.

Inverse design orchestrator (`inverse_design.py`)
- `InverseDesign` coordinates multiple `DeviceSpec`s.
  - Validates unique device names in `__post_init__`.
  - `get_simulations(params) -> dict[str, td.Simulation]`: builds one sim per device.
  - `run_simulations(sims) -> BatchData`: runs all sims concurrently via `tidy3d.web.run_async`.
  - `get_metric(batch_data) -> float`: sums `DeviceSpec.get_metric(batch_data[name])` across devices.
  - `get_objective(params) -> float`: convenience: build → run → aggregate.
  - `parameter_shape`: list of per-device lists of per-region sizes.
  - `_flatten_params(params) -> np.ndarray` and `_unflatten_params(vec) -> params`: helpers for optimizer integrations.
  - `ones(**kwargs)`: list of list of 1D arrays, matching devices→regions.

Metrics (`metric.py`)
- `Metric` (ABC): contract for computing scalars from monitor data.
  - `monitor_name: str`, `weight: float = 1.0`.
  - `evaluate(mnt_data) -> float`: returns scalar.
- `FluxMetric(Metric)`: sums `mnt_data.flux.values`.

Optimizer spec (`optimizer_spec.py`)
- `OptimizerSpec`: hyperparameters for an external optimization loop.
  - `num_steps: int`, `learning_rate: float`.
  - Not used directly by this scaffold; provided for consistency and integration.

Parameter shapes and nesting
---------------------------
- Per-region parameter is a 1D vector with length `region.parameter_shape`.
- Per-device parameters are `list[np.ndarray]`, aligned with `DeviceSpec.design_regions`.
- All-devices parameters are `list[list[np.ndarray]]`, aligned with `InverseDesign.device_specs`.
- `_unflatten_params` returns 1D segments for each region; `TopologyDesignRegion.to_structure` internally reshapes to 3D via `shape_3d`.

Minimal example
---------------
```python
import autograd.numpy as np
import tidy3d as td
from tidy3d.plugins.invdes2 import (
    DeviceSpec,
    FluxMetric,
    InverseDesign,
    OptimizerSpec,
    TopologyDesignRegion,
)

# 1) Base simulation with a Flux monitor
sim_base = td.Simulation(
    size=(10.0, 10.0, 10.0),
    grid_spec=td.GridSpec.auto(wavelength=1.0, min_steps_per_wvl=10),
    run_time=1.0,
    structures=(),
    monitors=[
        td.FluxMonitor(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), freqs=[2e14], name="flux")
    ],
    sources=(),
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    medium=td.Medium(permittivity=1.0),
)

# 2) Design regions and metrics
regions = [
    TopologyDesignRegion(
        size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), eps_bounds=(1.0, 4.0), pixel_size=0.02
    )
]
metrics = [FluxMetric(monitor_name="flux", weight=0.5)]

# 3) Two device scenarios (could vary monitors, sources, or regions per device)
dev1 = DeviceSpec(simulation=sim_base, design_regions=regions, metrics=metrics, name="d1")
dev2 = DeviceSpec(simulation=sim_base, design_regions=regions, metrics=metrics, name="d2")

inv = InverseDesign(optimizer_spec=OptimizerSpec(learning_rate=0.1, num_steps=10), device_specs=[dev1, dev2])

# 4) Build params, run, and score
params = inv.ones()  # nested list: devices → regions → 1D arrays
objective_value = inv.get_objective(params)
print("Objective:", objective_value)
```

Batch build/run/aggregate explicitly
------------------------------------
```python
params = inv.ones()
sims = inv.get_simulations(params)               # {"d1": Simulation, "d2": Simulation}
batch_data = inv.run_simulations(sims)           # runs async via Tidy3D Web
value = inv.get_metric(batch_data)               # aggregates across devices
```

Flatten/unflatten for optimizers
--------------------------------
```python
flat = inv._flatten_params(params)
restored = inv._unflatten_params(flat)
assert np.allclose(flat, inv._flatten_params(restored))

# Example: manual gradient loop (requires autograd-enabled backend for Tidy3D execution)
from autograd import grad

def objective_from_flat(vec):
    p = inv._unflatten_params(vec)
    return inv.get_objective(p)

g = grad(objective_from_flat)
vec = flat
for step in range(inv.optimizer_spec.num_steps):
    vec = vec - inv.optimizer_spec.learning_rate * g(vec)
final_params = inv._unflatten_params(vec)
```

Testing and emulation seam
--------------------------
The tests demonstrate two seam points for swapping execution backends via monkeypatching:

```python
# Swap device run for emulation
monkeypatch.setattr(
    DeviceSpec,
    "run_simulation",
    lambda self, simulation: run_emulated(simulation, task_name="test"),
)

# Swap batch run for emulation
monkeypatch.setattr(
    InverseDesign,
    "run_simulations",
    lambda self, sims: {name: run_emulated(sim, task_name=name) for name, sim in sims.items()},
)
```

Extending the system
--------------------
Add a new design region
```python
from dataclasses import dataclass
import autograd.numpy as np
import tidy3d as td
from tidy3d.plugins.invdes2 import DeviceSpec, InverseDesign, TopologyDesignRegion

@dataclass
class ParamBox(td.typing.NoAttrs):  # or just a plain dataclass
    center: tuple[float, float, float]
    size: tuple[float, float, float]

class CustomRegion(DesignRegion):
    @property
    def parameter_shape(self) -> int:
        return 3  # example: 3 parameters

    def to_structure(self, params: np.ndarray) -> td.Structure:
        # use params to drive geometry/material
        geometry = td.Box(center=(0,0,0), size=(1,1,1))
        return td.Structure(geometry=geometry, medium=td.Medium(permittivity=1.0))
```

Add a new metric
```python
from dataclasses import dataclass
import autograd.numpy as np
import tidy3d as td

@dataclass
class PowerAtFreq(Metric):
    monitor_name: str
    target_idx: int
    weight: float = 1.0

    def evaluate(self, mnt_data: td.FluxData) -> float:
        return float(mnt_data.flux.values[self.target_idx])
```

Design decisions and rationale
------------------------------
- Separation of concerns: geometry (regions), simulation assembly (device), orchestration (inverse design), scoring (metrics).
- Immutability: `DeviceSpec.get_simulation` uses `updated_copy` to avoid mutating the base.
- Batched orchestration: device names index batch results; uniqueness enforced at construction.
- Numeric compatibility: all math uses `autograd.numpy` for easy integration with gradient methods when supported by the backend.
- Simple helpers: `ones()` prevent shape mistakes when creating parameter vectors.

Limitations and future work
---------------------------
- `eps_bounds` not applied yet. Consider projections or relaxed binarization during optimization.
- `_unflatten_params` returns 1D segments; reshape happens inside region implementations. This is intentional but should be kept in mind when writing new regions.
- Optimizer loop is intentionally not baked-in; add a small `optimize()` convenience wrapper if a standard optimizer is adopted.
- Consider Protocol-based structural typing for plugin ergonomics once the contract stabilizes.

Quick reference
---------------
```python
# Create params
region_params = region.ones()
device_params = device_spec.ones()
all_params = inv.ones()

# Build and run
sim = device_spec.get_simulation(device_params)
sim_data = device_spec.run_simulation(sim)
score = device_spec.get_metric(sim_data)

# Multi-device
sims = inv.get_simulations(all_params)
batch = inv.run_simulations(sims)
objective = inv.get_metric(batch)
```


How invdes2 differs from invdes (design philosophy)
---------------------------------------------------

At a glance
- **invdes2 (this module)**: minimal, composable, and backend-agnostic scaffold. Focuses on clear seams: regions → device build → batch run → metric aggregation. No optimizer baked-in by design.
- **invdes (existing plugin)**: feature-rich, opinionated, and end-to-end. Includes parameter transforms, penalties, initialization specs, optimizer with history/caching, and analysis helpers.

Side‑by‑side comparison

- **Core abstraction**
  - invdes2: `DesignRegion` + `DeviceSpec` + `Metric` + `InverseDesign` (orchestration only).
  - invdes: `DesignRegion` with transformations/penalties, `InverseDesign`/`InverseDesignMulti`, plus an `Optimizer` producing an `InverseDesignResult`.

- **Parameter mapping to permittivity**
  - invdes2: `TopologyDesignRegion.to_structure(params)` reshapes a 1D vector into a voxel grid; `eps_bounds` reserved for future use; no built-in transforms/penalties.
  - invdes: rich pipeline: transformations (e.g., filter + projection) → material density → penalties (e.g., erosion/dilation) → permittivity mapping; validated gradients.

- **Objective composition**
  - invdes2: explicit `Metric` objects per monitor; `DeviceSpec.get_metric` does weighted sum; `InverseDesign.get_metric` aggregates across devices.
  - invdes: user supplies a `post_process_fn(sim_data)` (or batch dict) that defines the scalar objective; penalties subtract from the objective; helpers/utilities available.

- **Execution model**
  - invdes2: `web.run` for single and `web.run_async` for batches; easy monkeypatch seams in tests for emulation.
  - invdes: same web backends, but wrapped inside an optimizer loop with result persistence.

- **Optimization**
  - invdes2: no optimizer included; provides `_flatten_params/_unflatten_params` and simple helpers like `ones()` to integrate with external optimizers.
  - invdes: built‑in Adam optimizer, gradient computation via autograd, progress display, callbacks, result caching to disk, continue/resume, plotting, and export helpers.

- **State and results**
  - invdes2: stateless orchestration; returns scalars and leaves bookkeeping to the caller.
  - invdes: stateful `InverseDesignResult` with history of params, grads, states, and convenience accessors.

- **Validation and typing**
  - invdes2: plain dataclasses + simple ABCs; minimal validation; lightweight surface for rapid iteration.
  - invdes: heavy validation with Pydantic, explicit error messaging, gradient checks for transforms/penalties.

Why invdes2 exists
- **Iteration speed**: smaller conceptual surface for experimentation and research prototypes.
- **Composability**: metrics are small objects; devices are thin wrappers around base simulations; orchestration is explicit.
- **Testability**: seams for swapping execution (`run_simulation`, `run_simulations`) enable local/emulated unit tests without network calls.

Trade‑offs
- Fewer batteries included: no transformations, penalties, optimizer history, or built‑in continuation.
- Less guardrail validation: places more responsibility on the optimization script for bounds, projections, and schedules.

Migration guidance (invdes → invdes2)
- Map `post_process_fn` to one or more `Metric` implementations. If your post‑process touched multiple monitors, create a metric per monitor and combine via weights, or implement a custom metric that reads multiple monitors from `sim_data`.
- Convert transformations/penalties into your optimization loop: pre‑process the parameter vectors (e.g., filtering + projection) before passing to `to_structure`, and subtract penalty values inside your objective (or define a negative‑weight metric analogue).
- Replace `InverseDesignMulti` with multiple `DeviceSpec`s (one per simulation scenario); `InverseDesign` will build a batch and sum device objectives.
- Replace `Optimizer.run(...)` with an external loop that calls `get_objective(params)` or wraps it with autograd for gradients; use `_flatten_params/_unflatten_params` to interoperate with libraries.

When to choose which
- Choose invdes2 when you want a lean, hackable kernel for orchestration and prefer to own the optimization and parameter processing.
- Choose invdes when you want a turnkey, validated pipeline with transformations, penalties, built‑in optimizers, and analysis utilities.


