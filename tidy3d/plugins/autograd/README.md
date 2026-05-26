# Automatic Differentiation in Tidy3D

As of version 2.7.0, Tidy3D provides native support for automatic differentiation (AD), empowering you to perform gradient-based optimization and sensitivity analysis of photonic devices directly within your simulation workflow.

The gradient calculation is performed efficiently using the **adjoint method**, which keeps the cost nearly independent of the number of design parameters. In most objectives, only one adjoint simulation is needed because adjoint sources can be grouped by frequency or spatial port. If an objective uses multiple frequencies and multiple spatial ports, the standard adjoint path groups by whichever dimension is smaller, so the number of adjoint simulations is the smaller of the unique frequency count and the spatial-port count.

This implementation is powered by the `autograd` library and replaces the previous `jax`-based `adjoint` plugin, offering several key benefits:

*   **Simplicity**: Use standard Tidy3D components like `td.Structure` and `td.Simulation` directly in your differentiable functions.
*   **Ease of Use**: The regular `web.run` entry point is directly differentiable.
*   **Painless Installation**: The core AD framework, `autograd`, is a direct dependency of Tidy3D, removing the installation challenges associated with `jax`.

## Legacy Adjoint Plugin Reminder

The former `tidy3d.plugins.adjoint` (JAX) plugin was deprecated in 2.7 and is fully removed in 2.10. If you are updating older notebooks:

1. Replace `tidy3d.plugins.adjoint` imports (`tda.JaxSimulation`, `tda.JaxStructure`, etc.) with the standard `tidy3d` classes.
2. Switch `jax.grad` / `jax.numpy` to `autograd.grad` / `autograd.numpy`.
3. If you need PyTorch-centric tensors, use the lightweight wrapper in `tidy3d.plugins.pytorch` so you can keep your optimizer stack without touching `jax`.

For new projects, start directly with the native workflow described below.

## How It Works: `autograd` and the Adjoint Method

Tidy3D's AD capability combines two core technologies:

1.  **The `autograd` Framework**: This library automatically tracks all numerical operations in your Python objective function, building a computational graph to calculate derivatives using the chain rule.
2.  **The Adjoint Method**: Tidy3D provides the derivative of the FDTD simulation step (`web.run`) using adjoint simulations and the stored forward fields required for the backward pass.

When you request a gradient, Tidy3D and `autograd` work together behind the scenes:
1.  **Forward Pass**: Your code executes, running a standard FDTD simulation and calculating your scalar objective value. Tidy3D automatically stores the fields required for the subsequent gradient calculation.
2.  **Backward Pass**: `autograd` propagates gradients backward. When it reaches the simulation step, Tidy3D sets up the required adjoint simulations and uses both forward and adjoint fields to efficiently compute the gradients with respect to the traced simulation parameters.

### Forward/Adjoint Flow

The forward and backward passes follow this data flow:

* Forward: design parameters -> build `td.Simulation` -> `td.web.run()` -> `SimulationData` plus traced fields -> scalar objective.
* Backward: `autograd.value_and_grad()` propagates the objective gradient, Tidy3D launches the required adjoint simulations, and the result is returned as gradients for the design parameters.

## Basic Workflow

An inverse design optimization loop in Tidy3D generally follows these steps:

1.  **Define a function** that creates your `td.Simulation` based on a set of design parameters.
2.  **Define an objective function** that:
    *   Takes the design parameters as input.
    *   Calls the simulation-creation function.
    *   Runs the simulation via `web.run()`.
    *   Post-processes the results from the `SimulationData` object to return a single, real scalar value (the figure of merit).
3.  **Get the gradient function** using `autograd.value_and_grad()`.
4.  **Run an optimization loop** that iteratively calls the value-and-gradient function and updates the parameters using the computed gradient.

## Key Features at a Glance

* **Geometry + Material coverage**: Optimize most geometries (including `PolySlab` sidewall angles or `TriangleMesh` vertices) and dispersive media without custom wrappers.
* **Topology-friendly workflows**: `CustomMedium` plus filters/projections in `tidy3d.plugins.autograd` let you impose fabrication constraints while staying differentiable.
* **Broadband + adjoint throttling**: Adjoint jobs are auto-grouped by frequency or spatial port and limited by `max_num_adjoint_per_fwd`.
* **S-matrix gradients**: Differentiate objective functions involving supported scattering-matrix modelers when the underlying simulations are autograd-ready.
* **Far-field aware**: Near-field monitors can feed local `FieldProjector` steps, so you can optimize flux or far-field metrics. Objective functions cannot currently differentiate through server-side projection monitor data directly.

## Adjoint Job Count and Parallel Adjoint

The number of adjoint simulations depends on which simulation outputs the objective uses, not on the number of design parameters. The standard adjoint path first builds adjoint sources from the objective's vector-Jacobian product (VJP), then groups them by frequency or spatial port, whichever yields fewer simulations:

* If several objective terms use the same spatial port at multiple frequencies, they can usually be combined into one broadband adjoint simulation.
* If several objective terms use the same frequency at multiple spatial ports, they can usually be combined into one single-frequency adjoint simulation.
* If an objective uses multiple frequencies and multiple spatial ports, the standard path uses the smaller of the unique frequency count and the spatial-port count.

Unused frequencies in monitors increase forward-run field and permittivity data size, but they do not by themselves create adjoint simulations. Only frequencies that participate in the objective contribute adjoint sources.

`max_num_adjoint_per_fwd` caps the number of adjoint solves spawned by each forward simulation. Increase it intentionally for objectives that touch many frequencies, components, or spatial ports.

For local gradients, Tidy3D can also launch eligible canonical adjoint simulations in parallel with the forward solve:

* Enable it with `config.adjoint.parallel_run = True`.
* It is only effective when `local_gradient=True`; remote gradients ignore this flag.
* The initial supported outputs are mode monitor amplitudes, diffraction monitor amplitudes, and single-point field sampling.
* Unsupported monitor outputs fall back to the standard sequential adjoint path.
* Canonical parallel adjoint bases are grouped by spatial port, so this mode can trade extra simulations and credit usage for lower wall-clock time.

For mode monitors, `config.adjoint.parallel_adjoint_mode_direction_policy` controls whether Tidy3D assumes the outgoing direction or launches both `+` and `-` directions.

**Example: A Simple Optimization**
```python
import autograd
import autograd.numpy as anp
import tidy3d as td
from tidy3d import web

# 1. Function to create the simulation from parameters
def make_simulation(width):
    # ... (define sources, monitors, etc.)
    geometry = td.Box(size=(width, 0.5, 0.22))
    structure = td.Structure(geometry=geometry, medium=td.Medium(permittivity=12.0))
    sim = td.Simulation(
        # ... (simulation parameters)
        structures=[structure],
        # ...
    )
    return sim

# 2. Objective function returning a scalar
def objective_fn(width):
    sim = make_simulation(width)
    sim_data = web.run(sim, task_name="optimization_step")
    # Objective: maximize power in the fundamental mode
    mode_amps = sim_data["monitor_name"].amps.sel(direction="+", mode_index=0)
    return anp.sum(anp.abs(mode_amps.data)**2)

# 3. Get the value and gradient function
value_and_grad_fn = autograd.value_and_grad(objective_fn)

# 4. Optimization loop (naive gradient ascent)
width = 2.0  # Initial width
learning_rate = 0.05

for i in range(20):
    value, gradient = value_and_grad_fn(width)
    width = width + learning_rate * gradient  # move uphill to maximize
    print(f"Step {i+1}: Value = {value:.4f}, Width = {width:.3f}")
```

> **Frequency-domain monitor required**: Any simulation that carries traced structures or media must include at least one frequency-domain monitor (`FieldMonitor`, `ModeMonitor`, `DiffractionMonitor`, etc.). If a traced simulation has no frequency-domain monitor, `web.run` raises an `AdjointError` instead of falling back to a non-differentiable run. Keep at least one spectral sample active on every monitor that participates in the objective.

### Common Pitfalls

* Use `autograd.numpy` for every array operation in your objective; mixing standard NumPy silently drops gradients.
* Keep monitor frequencies focused on the objective to avoid unnecessary forward data size.
* Keep an eye on the traced-structure budget (default 500). Group repeated tiles or motifs into a `GeometryGroup` before differentiating large layouts.

## Capabilities and Supported Components

Tidy3D's AD framework supports a wide range of design scenarios.

### Differentiable Parameters (Simulation Inputs)

#### Geometry

| Component       | Traceable Attributes                                       | Example Use Case                  |
|:----------------|:-----------------------------------------------------------|:----------------------------------|
| `Box`           | `.center`, `.size`                                         | Shape Optimization                |
| `Sphere`        | `.center`, `.radius`                                       | Shape Optimization                |
| `Cylinder`      | `.center`, `.radius`, `.length`, `.sidewall_angle`         | Shape Optimization                |
| `PolySlab`      | `.vertices`, `.slab_bounds`, `.sidewall_angle`             | Shape Optimization & taper tuning |
| `GeometryGroup` | `.geometries`                                              | Grouping for performance          |
| `ClipOperation` | traced parameters in underlying geometries                 | Boolean shape optimization        |
| `TriangleMesh`  | `.mesh_dataset.surface_mesh`                               | 3D Shape Optimization             |

#### Base Materials

| Component                 | Traceable Attributes | Example Use Case                           |
|:--------------------------| :--- |:-------------------------------------------|
| `Medium`                  | `.permittivity`, `.conductivity` | Material Optimization                      |
| `CustomMedium`            | Permittivity data array | Topology Optimization                      |
| `AnisotropicMedium`       | nested `xx`, `yy`, `zz` component medium fields | Anisotropic material optimization |
| `CustomAnisotropicMedium` | nested custom `xx`, `yy`, `zz` component fields | Anisotropic topology optimization |

#### Dispersive Models

| Component | Traceable Attributes | Example Use Case |
| :--- | :--- | :--- |
| `PoleResidue` | `.eps_inf`, `.poles` | General dispersive fit |
| `CustomPoleResidue` | `.eps_inf`, `.poles` (spatial data) | Spatially varying dispersive fit |
| `Sellmeier` / `CustomSellmeier` | `coeffs[i][0]` (B) and `coeffs[i][1]` (C) | Refractive-index dispersion control |
| `Lorentz` / `CustomLorentz` | `eps_inf`, `(Δε_i, f_i, δ_i)` | Resonant material modeling |
| `Drude` / `CustomDrude` | `eps_inf`, `(f_{p,i}, δ_i)` | Free-carrier / plasmonic tuning |
| `Debye` / `CustomDebye` | `eps_inf`, `(Δε_i, τ_i)` | Relaxation media / polymers |

#### Sources

| Component | Traceable Attributes |
| :--- | :--- |
| `CustomCurrentSource` | `.center`, `.current_dataset` |
| `CustomFieldSource` | `.center`, `.field_dataset` |
| `GaussianBeam` | `.center`, `.angle_theta`, `.angle_phi`, `.pol_angle`, `.waist_radius`, `.waist_distance` |
| `AstigmaticGaussianBeam` | `.center`, `.angle_theta`, `.angle_phi`, `.pol_angle`, `.waist_sizes`, `.waist_distances` |

### Differentiable Results (Simulation Outputs)

| Source monitor → data object | Traceable attributes & methods | Notes |
| :--- | :--- | :--- |
| `ModeMonitor` → `ModeData` | `.amps` | Differentiate modal amplitudes and powers directly. |
| `GaussianOverlapMonitor` / `AstigmaticGaussianOverlapMonitor` → `FieldOverlapData` | `.amps` | Differentiate overlap amplitudes used by Gaussian ports. |
| `DiffractionMonitor` → `DiffractionData` | `.amps` | Capture gradients of diffraction efficiencies / orders. |
| `FieldMonitor` / `PermittivityMonitor` → `FieldData`, `PermittivityData` | field components, permittivity components, `FieldData.flux` | Use these to build custom objectives (power, overlap, material penalties). |
| `SimulationData` helpers | `get_intensity(field_monitor_name)`, `get_poynting_vector(field_monitor_name)` | Convenience wrappers remain differentiable because they operate on traced monitor data. |

#### Requires Local Post-processing

| Data target | Status |
| :--- | :--- |
| `FluxMonitor` (`FluxData`) | Not directly differentiable. Record the enclosing `FieldMonitor` and integrate the Poynting vector yourself. |
| Field projection monitors (`FieldProjectionAngleData`, `FieldProjectionCartesianData`, `FieldProjectionKSpaceData`) | Not supported for adjoint. Store the near fields and run `FieldProjector.from_near_field_monitors` locally to form far-field gradients. |

## Runtime Controls and Gradient Flow

*   **`local_gradient`**: Pass `local_gradient=True` to `web.run` (or set `config.adjoint.local_gradient`) to download the forward and adjoint field data. This is required if you rely on local-only `config.adjoint.*` overrides such as grid spacing, gradient precision, or frequency chunking, because remote/server-side gradients ignore those settings.
    When enabled, Tidy3D attaches the adjoint monitors up front (via `_with_adjoint_monitors`) so the forward run exports all fields needed for the backward pass, increasing monitor count, runtime, and download size. Ensure the directory pointed to by `config.adjoint.local_adjoint_dir` has sufficient space.
*   **Adjoint batch safety (`max_num_adjoint_per_fwd`)**: Each forward simulation can spawn at most `max_num_adjoint_per_fwd` adjoint solves (defaults to `config.adjoint.max_adjoint_per_fwd = 10`). Increase the argument if your objective touches many monitors or broadband field data; otherwise the run will raise an error before launching excessive jobs.
*   **Tracer budget (`max_traced_structures`)**: Autograd accepts up to `config.adjoint.max_traced_structures` traced geometries (default 500). Use `GeometryGroup` to consolidate repeated materials or prune unused tracers before submission.
*   **Adjoint data location**: When `local_gradient=True`, intermediate data are stored under `config.adjoint.local_adjoint_dir` (defaults to `adjoint_data/`). Make sure the directory has enough space if you are differentiating large field monitors.
*   **Parallel local adjoint (`parallel_run`)**: When `config.adjoint.parallel_run=True`, eligible canonical adjoint simulations can be submitted with the forward solve for local-gradient workflows.

For every other switch (e.g., `gradient_precision`, `solver_freq_chunk_size`, custom monitor spacing), refer to the [configuration reference](https://docs.flexcompute.com/projects/tidy3d/en/latest/configuration/reference.html) under the `autograd` section.

## The Autograd Plugin: Advanced Design Functions

Beyond the core differentiation of components, Tidy3D includes a powerful set of tools in the `tidy3d.plugins.autograd` module designed to facilitate advanced optimization tasks. This toolkit provides differentiable building blocks for common inverse design techniques like topology optimization, shape parameterization, and enforcing fabrication constraints.
All of the utilities described here live directly under `tidy3d.plugins.autograd` (see the `invdes`, `functions`, `primitives`, `optimizers`, and `utilities` submodules for the actual call signatures).

### Topology Optimization and Fabrication-Aware Design

Many of the tools are geared towards topology optimization, where the goal is to find the optimal distribution of materials in a design region.

*   **Filtering**: Functions like `make_circular_filter`, `make_conic_filter`, `make_gaussian_filter`, and `make_filter` apply a convolution to the raw design parameters. This is a standard technique to enforce a minimum length scale and create smooth, manufacturable features.
*   **Projection**: To ensure the final design consists of distinct materials (e.g., silicon or air), projection functions like `tanh_projection`, `ramp_projection`, and `smoothed_projection` are used. They smoothly binarize the continuous design parameters to values like 0 and 1.
*   **Penalties**: To further guide the optimization, you can add penalty terms to your objective function. The toolkit includes `make_curvature_penalty` to control the curvature of boundaries and `make_erosion_dilation_penalty` to enforce minimum feature sizes.

These operations can be easily connected using the `chain` utility to create a standard data processing pipeline for your parameters.

```python
from tidy3d.plugins.autograd import (
    chain,
    make_conic_filter,
    tanh_projection,
)
from functools import partial

# Define a filter to enforce a 20 nm minimum feature size on a 5 nm grid.
conic_filter = make_conic_filter(radius=0.02, dl=0.005)

# Define a projection function to binarize the design
project = partial(tanh_projection, beta=8.0, eta=0.5)

# Chain them together to create a single processing function
process_params = chain(conic_filter, project)

# In the objective function, apply this to the raw parameters
def objective_fn(raw_params):
    processed_params = process_params(raw_params)
    # ... create CustomMedium and Simulation from processed_params ...
    # ... run simulation and compute objective ...
    return objective_value
```

### Differentiable Primitives and Utilities

The plugin also offers several general-purpose differentiable functions:

*   `interpolate_spline`: A powerful tool for parameterizing device geometries. You can define a shape using a small number of control points and use this function to generate a smooth, differentiable spline. Optimizing the control points allows for flexible shape optimization.
*   **Morphological Operations**: Differentiable versions of standard image processing functions like `grey_dilation`, `grey_erosion`, `grey_opening`, `grey_closing`, and `convolve` are available for parameter processing.
*   `least_squares`: A differentiable least-squares optimizer for fitting models to data within your objective function.
*   `smooth_max` / `smooth_min`: Differentiable approximations of `max()` and `min()`, useful for creating objectives that depend on the maximum or minimum value in a set of results.
*   `scalar_objective`: A helper for enforcing scalar objective returns compatible with `grad` and `value_and_grad`.
*   `Adam`, `adam`, `apply_updates`, and `optimize`: Lightweight optimization helpers for plugin-native optimization loops.

## Best Practices and Limitations

To ensure robust and efficient optimizations, please consider the following guidelines. For more details, refer to the official [autograd tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md).

### Do's

*   **Use `autograd.numpy`**: Always import `autograd.numpy as anp` and use it for all numerical operations within your objective function.
*   **Return a scalar numeric objective**: For a selected single-value `DataArray`, use `.item()`; for a single-element array, returning `objective_value.data` is also acceptable.
    ```python
    objective_value = mode_power.sel(mode_index=0, f=freq0)
    return objective_value.item()  # alternatively, for single-element arrays: objective_value.data
    ```
*   **Extract data before complex post-processing**: For more complex objective functions, extract the `.data` attribute from the `DataArray` before performing any `autograd.numpy` operations.
*   **Use `GeometryGroup`**: To optimize more than 500 structures, group them into a single `GeometryGroup` if they share the same medium.
*   **Set `background_medium` when needed**: When optimizing a shape embedded in a material that differs from the simulation background, set `Structure.background_medium` to describe the material outside the traced structure.
*   **Manage Monitor Frequencies**: During optimization, monitor frequencies that do not enter the objective still increase forward data size. Keep monitor frequency lists focused on what you need.

### Don'ts

*   **Don't Use In-place Operations**: Avoid in-place assignment (`x[i] = val`) or operators (`x += 1`) on arrays tracked by `autograd`.
*   **Don't Differentiate `FluxMonitor`**: `FluxMonitor` data is not directly differentiable. To optimize flux, you must use a `FieldMonitor` and compute the flux from the field data.
*   **Don't Differentiate Server-Side Projections**: Far-field gradients must be computed locally using `FieldProjector` on downloaded `FieldMonitor` data.

### Current Limitations

*   **Traced Structures Limit**: A maximum of 500 structures containing tracers can be added to a `Simulation`. Use `GeometryGroup` to bypass this.
*   **Adjoint solve budget**: Objectives that use many field-monitor frequencies, components, or spatial ports can require multiple adjoint simulations.
*   **Forward data size**: The forward simulation records fields and permittivities within the bounding box of any traced object at each unique frequency in the simulation. This can increase data usage when monitors include frequencies that are not relevant to the objective.

## Migrating from the `adjoint` Plugin

Updating your code from the old `adjoint` plugin is straightforward:

1.  **Replace `Jax` Components**: Replace `tidy3d.plugins.adjoint` (`tda`) imports with standard `tidy3d` (`td`) imports. For example, `tda.JaxStructure` becomes `td.Structure`, and `tda.JaxMedium` becomes `td.Medium`.
2.  **Use Standard `td.Simulation`**: The `JaxSimulation` class is no longer needed. You can now use a standard `td.Simulation`. Tidy3D automatically detects which components are being traced for differentiation.
3.  **Use Standard `web.run`**: Use the standard `web.run` function. No special wrappers are required.

If you have feature requests or questions, please feel free to file an issue or start a discussion on the [Tidy3D GitHub repository](https://github.com/flexcompute/tidy3d).

Happy autogradding!
