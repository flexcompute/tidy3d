# Simulation package architecture

This package contains the FDTD simulation model and its Yee-grid behavior. It replaces the
former monolithic `components/simulation.py` module without changing the public
`tidy3d.components.simulation` import surface.

## Model layering

`AbstractYeeGridSimulation` in `yee.py` owns fields and validation shared by simulations
defined on a Yee grid. `Simulation` in `model.py` adds the concrete FDTD fields, the ordered
post-init validation pipeline, and pre-upload validation.

Behavior remains grouped in focused area modules. Their module-level method descriptors are
bound directly in the model class bodies, so the concrete classes own the documented methods
without adding behavioral bases to their MRO:

| Area | Modules | Layer |
| --- | --- | --- |
| Shared Yee behavior | `boundaries`, `construction`, `grid`, `materials`, `mode_integration`, `monitors`, `visualization` | `AbstractYeeGridSimulation` |
| Concrete FDTD behavior | `adjoint`, `export`, `rf_integration`, `runtime`, `sources`, `tfsf` | `Simulation` |
| Model definitions | `yee`, `model` | Fields, Pydantic validators, and lifecycle entry points |
| Shared limits | `constants` | Canonical owners for validation limits and warning thresholds |

Some area modules contribute behavior to both layers. `yee.py` and `model.py` make that
ownership explicit through their binding sections.

## Validation ownership

Pydantic model validators remain on the class that owns the fields. Area modules expose ordinary
validation helpers, which are bound to `AbstractYeeGridSimulation` or `Simulation`.
`AbstractYeeGridSimulation._run_after_validators()` and
`Simulation._run_after_validators()` call them in explicit dependency order. Do not turn these
helpers into independently registered model validators: their ordering is part of the model
contract.

When adding a check:

1. Put the implementation in the module for the relevant domain.
2. Bind it in the relevant area block on the owning model.
3. Call it from the owning model's ordered validation entry point.
4. Preserve the existing call order unless the dependency change is intentional and tested.
5. Use location-aware validation errors for failures attributable to a model field.

## Import and compatibility contract

`simulation/__init__.py` is an eager compatibility facade. Existing imports such as
`from tidy3d.components.simulation import Simulation` and former module-level constants must
continue to work there.

Internal dependencies must remain static and acyclic. Area modules may depend on leaf helpers,
but they must not import `model.py` or use function-local imports to hide cycles. The intended
direction is:

```text
constants and external leaf helpers
    -> area implementations
    -> yee.py
    -> model.py
    -> simulation/__init__.py
```

Optional plotting dependencies may remain function-local so importing `tidy3d` does not import
Matplotlib eagerly; they are not a mechanism for resolving internal cycles.
