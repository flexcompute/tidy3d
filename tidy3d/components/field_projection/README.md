# Field Projection Package

This package contains the local field-projection implementation behind `FieldProjector`.

The refactor splits the implementation by responsibility:

- `projector.py` contains the public entry point and shared current preparation.
- `common.py` contains shared constants, helper functions, reusable metadata, and result assembly.
- `approximate_angle.py` contains the angle-monitor path.
- `approximate_paired.py` contains the shared paired-point engine used by Cartesian and k-space monitors.
- `exact.py` contains the exact projection kernel that is reused by the monitor-specific modules when `far_field_approx=False`.
- `__init__.py` exports `FieldProjector`.

## Public Surface

The intended public interface of this package is `FieldProjector`.

Typical lifecycle:

1. Build a projector with `FieldProjector.from_near_field_monitors(...)` when the source fields
   come from simulation monitor data.
   Alternatively, use `FieldProjector.from_near_field_data(...)` to project custom or modified
   `FieldData` independently, using only the inputs needed for the projection.
2. Reuse the projector for one or more calls to `project_fields(...)`.
3. Receive one of the projection monitor data models:
   - `FieldProjectionAngleData`
   - `FieldProjectionCartesianData`
   - `FieldProjectionKSpaceData`

All other names in this package are private implementation details.

## Module Responsibilities

### `projector.py`

`projector.py` is the orchestration layer.

It owns:

- the `FieldProjector` model
- default projector configuration such as `pts_per_wavelength` and `origin`
- source-surface current extraction from simulation monitor data or raw `FieldData`
- current resampling / colocation
- apodization window application
- dispatch from `project_fields(...)` to the monitor-specific implementations

This is the module that knows how near-field monitor data is converted into equivalent surface
currents, whether the source comes from a full `SimulationData` object or from raw `FieldData`.

### `common.py`

`common.py` holds logic that is shared by more than one projection path.

This includes:

- package-level constants
- progress tracking helpers
- reusable approximate far-field preparation
- result wrapping into projection data models
- trapezoidal integration helpers
- the two low-level far-field integral kernels
- shared conversion from projected current components to spherical field components

It also defines the two internal metadata containers used by the approximate paths:

- `_FarFieldIntegralSpec`
- `_PreparedFarFieldProjection`

These structures capture the static information needed to evaluate a separable far-field integral without rebuilding the same indexing and weighting metadata at every call.

### `approximate_angle.py`

`approximate_angle.py` owns the angle-monitor implementation.

Why it stands alone:

- angle monitors naturally evaluate fields on a tensor-product `(theta, phi)` grid
- the phase construction follows that grid structure directly
- the result layout is different from the flattened paired-point paths used by Cartesian and k-space monitors

This module handles both:

- approximate angular projection when `far_field_approx=True`
- exact angular projection by flattening the observation grid and delegating to the shared exact point projector when `far_field_approx=False`

### `approximate_paired.py`

`approximate_paired.py` contains the shared implementation for Cartesian and k-space projection.

These two monitor types share the same high-level algorithm:

- construct observation points
- convert them to spherical observation angles
- flatten them into paired point lists
- evaluate the paired-point projection engine
- reshape the result back onto the requested monitor grid

This module therefore owns:

- approximate Cartesian projection
- approximate k-space projection
- frequency chunking for approximate paired paths
- the shared paired-point batching logic
- exact Cartesian and exact k-space fallback orchestration by calling the shared exact point projector and reshaping the result

### `exact.py`

`exact.py` holds the exact projection kernel that is shared by the monitor-specific modules.

Its job is to evaluate the exact homogeneous-medium Green-function expression for a flat list of observation points and a list of source surfaces.

This module is intentionally focused on array-level math:

- exact point-by-point evaluation
- per-surface field assembly
- Cartesian-to-spherical field conversion
- exact surface integration

It does not know how to build projection monitor result models. That remains in the monitor-specific modules and shared wrappers.

## Internal Data Structures

The package uses two different styles of internal data depending on the path.

### Approximate path data

Approximate projection relies on prepared metadata objects defined in `common.py`:

- `_FarFieldIntegralSpec`
  - integration weights
  - tangential integration axes
  - 2D vs 3D integral mode
  - optional 1D integration axis for 2D simulations
- `_PreparedFarFieldProjection`
  - sliced field components for one frequency
  - the corresponding integral spec
  - source-grid coordinates
  - propagation and impedance factors

These prepared objects are lightweight summaries of the data needed by the approximate kernels.

### Exact path data

On this branch, the exact path still operates mostly on raw arrays and datasets rather than dedicated exact-path metadata classes.

The main pieces passed around are:

- `surface_currents`: a list of `(FieldProjectionSurface, xr.Dataset)` pairs
- flattened observation coordinate arrays `x`, `y`, `z`
- stacked field arrays with leading component dimension

The exact module is therefore simpler structurally than the approximate path, but it is also less explicit about intermediate state.

## Configurability

The main configuration points are on `FieldProjector` and `project_fields(...)`.

### Projector construction

`FieldProjector` stores:

- `sim_data`
  - the simulation-backed data source when projection starts from monitor results
- `surfaces`
  - the source surfaces used for projection
- `pts_per_wavelength`
  - resampling density for source currents
  - `None` means use the simulation grid directly after colocation
- `origin`
  - local coordinate origin for observation points
  - if omitted, the average of all source-surface centers is used

When using `FieldProjector.from_near_field_data(...)`, the projector is configured from the
provided `FieldData`, homogeneous `medium`, source normal direction, and projection-point
arguments, without requiring a full simulation object.

### Per-call options

`project_fields(...)` accepts:

- `proj_monitor`
  - defines the target projection geometry and whether the path is approximate or exact
- `verbose`
  - enables local progress bars
- `freq_chunk_size`
  - only affects approximate Cartesian and k-space projection
  - controls how many frequencies are prepared together
  - `None` means prepare all frequencies together

The monitor itself also controls important behavior:

- monitor type selects angle vs Cartesian vs k-space output
- `far_field_approx` selects approximate vs exact evaluation
- window settings control current apodization
- monitor medium, if provided, overrides the projector background medium for the projection

## End-to-End Flow

At a high level, a local projection call works as follows.

1. `FieldProjector` validates and stores source-surface information.
2. Source fields are converted into equivalent electric and magnetic surface currents.
3. Currents are colocated onto the projection surface grid and optionally resampled.
4. Coordinates are shifted into the projector-local origin.
5. `project_fields(...)` dispatches by monitor type.
6. The selected module chooses approximate or exact evaluation based on `proj_monitor.far_field_approx`.
7. The projection kernel returns raw field-component arrays.
8. `common.py` wraps those arrays into the correct projection monitor data model.

The module split is designed so that:

- `projector.py` owns source-facing preparation
- the monitor-specific modules own output-shape logic
- `common.py` owns shared approximate helpers and wrappers
- `exact.py` owns exact kernel math

## 2D and 3D Behavior

The package supports both 2D and 3D simulations.

Important differences:

- approximate kernels use different integration metadata in 2D and 3D
- 2D validation happens before projection dispatch
- resampling skips collapsed dimensions with only one coordinate
- exact and approximate monitor-specific code still produce the same public projection data models

## Extension Guidance

When adding or changing functionality:

- keep public API changes in `projector.py` and `__init__.py`
- place monitor-type-specific orchestration in `approximate_angle.py` or `approximate_paired.py`
- place shared helpers in `common.py` only when more than one path uses them
- keep exact kernel math in `exact.py`
- prefer adding private helper structures only when they clarify repeated state or repeated contracts

As a rule, split by algorithmic responsibility rather than by monitor label alone.
