# Automatic Differentiation in Tidy3D

As of version 2.7.0, `tidy3d` supports the ability to differentiate functions involving a `web.run` of a `tidy3d` simulation.
This allows users to optimize objective functions involving `tidy3d` simulations using gradient descent.
This gradient calculation is done under the hood using the adjoint method, which requires just one additional simulation, no matter how many design parameters are involved.

This functionality was previously available using the `adjoint` plugin, which used `jax`. There were a few issues with this approach:

1. `jax` can be quite difficult to install on many systems and often conflicted with other packages.
2. Because we wanted `jax` to be an optional dependency, the `adjoint` plugin was separated from the regular `tidy3d` components, requiring a new set of `Jax_` classes.
3. Because we inherited these classes from their `tidy3d` components, for technical reasons, we needed to separate the `jax`-traced fields from the regular fields.
   For example, `JaxSimulation.input_structures` and `.output_monitors` were needed.

All of these limitations (among others) motivated us to come up with a new approach to automatic differentiation, which was introduced as an experimental feature in `2.7`.
The `adjoint` plugin will continue to be supported indefinitely, but no new features will be developed for it.
We also believe the new approach offers a far better user experience, so we encourage users to switch whenever is convenient.
This guide will give some instructions on how to do so.

## New implementation using `autograd`

Automatic differentiation in `2.7` is built directly into `tidy3d`.
One can perform objective function differentiation similarly to what was possible in the `adjoint` plugin.
However, this can be done using regular `td.` components, such as `td.Simulation`, `td.Structure`, and `td.Medium`.
Also, the regular `web.run()` function is now differentiable, so there is no need to import a wrapper.
In short, users can take existing functional code and differentiate it without changing much:

```py
def objective(eps: float) -> float:
    structure = td.Structure(
        medium=td.Medium(permittivity=eps),
        geometry=td.Box(...),
    )

    sim = td.Simulation(
        structures=[structure],
        ...
    )

    data = td.web.run(sim)

    return np.sum(np.abs(data["mode"].amps.sel(mode_index=0))).item()

# compute derivative of objective(1.0) with respect to input
autograd.grad(objective)(1.0)
```

Instead of using `jax`, we now use the [autograd](https://github.com/HIPS/autograd) package for our "core" automatic differentiation.
Many `tidy3d` components now accept and are compatible with `autograd` arrays.
Due to its lightweight nature and minimal dependencies, `autograd` has been made a core dependency of `tidy3d`.

Although `autograd` is used internally, we provide wrappers for other automatic differentiation frameworks, allowing you to use your preferred AD framework (e.g., `jax`, `pytorch`) with minimal syntax changes. For instance, you can refer to our PyTorch wrapper [here](../pytorch/).

The usability of `autograd` is extremely similar to `jax` but with a couple of modifications, which we'll outline below.

### Migrating from jax to autograd

Like in `jax`, the gradient functions can be imported directly from `autograd`:

```py
import jax
jax.grad(f)
```

becomes

```py
import autograd
autograd.grad(f)
```

There is also a `numpy` wrapper that can be similarly imported from `autograd.numpy`

```py
import jax.numpy as jnp
jnp.sum(...)
```

becomes

```py
import autograd.numpy as anp
anp.sum(...)
```

`Autograd` supports fewer features than `jax`.
For example, the `has_aux` option is not supported in the default `autograd.grad()` function, but one can write their own utilities to implement these features, as we show in the notebook examples.
We also have a `value_and_grad` function in `tidy3d.plugins.autograd.differential_operators` that is similar to `jax.value_and_grad` and supports `has_aux`.
Additionally, `autograd` has a `grad_with_aux` function that can be used to compute gradients while returning auxiliary values, similar to `jax.grad` with `has_aux`.

Otherwise, `jax` and `autograd` are very similar to each other in practice.

### Migrating from `adjoint` plugin

Converting code from the `adjoint` plugin to the native autograd support is straightforward.

Instead of importing classes from the `tda` namespace, with names like `tda.Jax_`, we can just use regular `td.` classes.

```py
import tidy3d.plugins.adjoint as tda
tda.JaxStructure(...)
```

becomes

```py
import tidy3d as td
td.Structure(...)
```

These `td.` classes can be used directly in the differentiable objective functions.
Like before, only some fields are traceable for differentiation, and we outline the full list of supported fields in the feature roadmap below.

Furthermore, there is no need for separated fields in the `JaxSimulation`, so one can eliminate `output_monitors` and `input_structures` and put everything in `monitors` and `structures`, respectively.
`tidy3d` will automatically determine which structure and monitor is traced for differentiation.

Finally, the regular `web.run()` and `web.run_async()` functions have their derivatives registered with `autograd`, so there is no need to use special web API functions.
If there are no tracers found in `web.run()` or `web.run_async()` simulations, the original (non-`autograd`) code will be called.

## Common Gotchas

Autograd has some limitations and quirks.
A good starting point to get familiar with them is the [autograd tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md).

Some of the most important autograd "Don'ts" are:

- Do not use in-place assignment on numpy arrays, e.g., `x[i] = something`.
  Often, you can formulate the assignment in terms of `np.where()`.
- Similarly, do not use in-place operators such as `+=`, `*=`, etc.
- Prefer numpy functions over array methods, e.g., use `np.sum(x)` over `x.sum()`.

It is important to note that any function you use with autograd differential operators like `grad`, `value_and_grad`, `elementwise_grad`, etc., must return real values in the form of a float, a tuple of floats, or a numpy array.
Specifically, for `grad` and `value_and_grad`, the output must be either a scalar or a one-element array.

When extracting values from `SimulationData`, ensure that any output value is converted to a float or numpy array before returning.
This is because numpy operations on `DataArray` objects will yield other `DataArray` objects, which are not compatible with autograd's automatic differentiation when returned from the function.

For example:

```py
def objective(params: np.ndarray) -> float:
    sim = make_simulation(params)
    sim_data = td.web.run(sim)

    amps = sim_data["mode_monitor"].amps
    mode_power = np.abs(amps)**2  # mode_power is still a DataArray!

    # either select out a specific value
    objective_value = mode_power.sel(mode_index=0, f=freq0)
    # or, for example, sum over all frequencies
    objective_value = mode_power.sel(mode_index=0).sum()

    # just make sure that whatever you return is scalar and a numeric type by extracting the scalar value with item()
    return objective_value.item()  # alternatively, for single-element arrays: flux.data or flux.values (deprecated)
```

For more complex objective functions, it is advisable to extract the `.data` attribute from the `DataArray` _before_ performing any numpy operations.
Although most autograd numpy functions are compatible with `DataArray` objects, there can be instances of unexpected behavior.
Therefore, working directly with the underlying data of the `DataArray` is generally a more robust approach.

For example:

```py
def objective(params: np.ndarray) -> float:
    sim = make_simulation(params)
    sim_data = td.web.run(sim)

    fields = sim_data["field_monitor"]

    # extract the data from the DataArray
    Ex = fields.Ex.data
    Ey = fields.Ey.data
    Ez = fields.Ez.data

    # we can now use these just like regular numpy arrays
    intensity = anp.abs(Ex) ** 2 + anp.abs(Ey) ** 2 + anp.abs(Ez) ** 2  # sim_data.get_intensity("field_monitor") would also work of course
    norm_intensity = anp.linalg.norm(intensity)

    return norm_intensity  # no .item() needed
```

## Feature Roadmap

Please check out our [Adjoint Master Plan](https://github.com/flexcompute/tidy3d/issues/1548) on GitHub if you want to stay updated on the progress of planned features and contribute to the discussion.

### Currently Supported

The following components are traceable as inputs to the `td.Simulation`

| Component Type                                                    | Traceable Attributes                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------- |
| rectangular prisms                                                | `Box.center`, `Box.size`                                |
| polyslab (including those with dilation or slanted sidewalls)     | `PolySlab.vertices`                                     |
| regular mediums                                                   | `Medium.permittivity`, `Medium.conductivity`            |
| spatially varying mediums (for topology optimization mainly)      | `CustomMedium.permittivity`, `CustomMedium.eps_dataset` |
| groups of geometries with the same medium (for faster processing) | `GeometryGroup.geometries`                              |
| complex and self-intersecting polyslabs                           | `ComplexPolySlab.vertices`                              |
| dispersive materials                                              | `PoleResidue.eps_inf`, `PoleResidue.poles`              |
| spatially dependent dispersive materials                          | `CustomPoleResidue.eps_inf`, `CustomPoleResidue.poles`  |
| cylinders                                                         | `Cylinder.radius`, `Cylinder.center`                    |

The following components are traceable as outputs of the `td.SimulationData`

| Data Type         | Traceable Attributes & Methods                                |
| ----------------- | ------------------------------------------------------------- |
| `ModeData`        | `amps`                                                        |
| `DiffractionData` | `amps`                                                        |
| `FieldData`       | `field_components`, `flux`                                    |
| `SimulationData`  | `get_intensity(field_monitor)`, `get_poynting(field_monitor)` |

We also support the following high-level features:

- To manually set the background permittivity of a structure for purposes of shape optimization, one can set `Structure.background_permittivity`.
  This is useful when there is a substrate or multiple overlapping structures as some geometries, such as `PolySlab`, do not automatically detect background permittivity and instead use the `Simulation.medium` by default.
- Compute gradients for objective functions that rely on multi-frequency data using a single broadband adjoint source.
- Enable server-side gradient processing by setting `local_gradient=False` in the web functions.
  This can significantly reduce data storage time.
  However, exercise caution when using this feature with multi-frequency monitors and large design regions, as it may result in substantial data storage on our servers.

We currently have the following restrictions:

- Only 500 max structures containing tracers can be added to the `Simulation` to cut down on processing time.
  To bypass this restriction, use `GeometryGroup` to group structures with the same medium.
- `web.run_async` for simulations with tracers does not return a `BatchData` but rather a `dict` mapping task name to `SimulationData`.
  There may be high memory usage with many simulations or a lot of data for each.
- Tidy3D can handle objective functions over a single simulation under any of the following conditions for the monitors that the objective function output depends on:
  - Several monitors, all with the same frequency.
  - One monitor with many frequencies where the data is being extracted out of a single coordinate (e.g., single mode_index or direction will work, multiple will not).
  If your optimization does not fall into one of these categories, you must split it into separate simulations and run them with `web.run_async`. In all cases, the adjoint simulation bandwidth will be the same as the forward simulation. These limitations allow us to avoid the need to use methods that combine all adjoint sources into one simulation, which have the potential to degrade accuracy and increase the run time and cost significantly. That being said, we plan to offer support for more flexible and general broadband adjoint in the future.

### To be supported soon

Next on our roadmap (targeting 2.8 and 2.9, fall 2024) is to support:

- `TriangleMesh`.
- `GUI` integration of invdes plugin.

### Finally

If you have feature requests or questions, please feel free to file an issue or discussion on this `tidy3d` front-end repository.

Happy autogradding!

## Developer Notes

To convert existing tidy3d front end code to be autograd compatible, will need to be aware of

- `numpy` -> `autograd.numpy`
- Casting to `float()` is not supported for autograd `ArrayBox` objects.
- `isclose()` -> `np.isclose()`
- `array[i] = something` needs a different approach (happens in mesher a lot)
- Whenever we pass things to other modules, like `shapely` especially, we need to be careful that they are untraced.
- I just made structures static before any meshing, as a cutoff point. So if we add a new `make_grid()` call somewhere, e.g. in a validator, just need to be aware.
