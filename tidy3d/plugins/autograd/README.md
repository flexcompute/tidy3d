# Automatic Differentiation in Tidy3D

### Context

As of version 2.7.0, `tidy3d` supports the ability to differentiate functions involving a `web.run` of a `tidy3d` simulation. This allows users to optimize objective functions involving `tidy3d` simulations using gradient-descent. This gradient calculation is done under the hood using the adjoint method, which requires just 1 additional simulation, no matter how many design parameters are involved.

This functionality was previously available using the `adjoint` plugin, which used `jax`. There were a few issues with this approach:

1. `jax` was often quite difficult to install on many systems and often conflicted with other packages.
2. Because we wanted `jax` to be an optional dependency, the `adjoint` plugin was separated from the regular `tidy3d` components, requiring a new set of `Jax_` classes.
3. Because we inherited these classes from their `tidy3d` components, for technical reasons, we needed to separate the `jax`-traced fields from the regular fields. For example, `JaxSimulation.input_structures` and `.output_monitors` were needed.

All of these limitations (among others) motivated us to come up with a new approach to automatic differentiation, which will be introduced as an experimental feature in `2.7`. The `adjoint` plugin will still be supported in the indefinite future, but will not be developed with new features. We also believe the new approach offers a far better user experience, so we encourage users to switch whenever is convenient. This guide will give some instructions on how to do so.

## New implementation using `autograd`

Automatic differentiation in 2.7 is built directly into `tidy3d`. One can perform objective function differentiation similarly to what was possible in the `adjoint` plugin. However, this can be done using regular `td.` components, such as `td.Simulation`, `td.Structure`, and `td.Medium`. Also, the regular `web.run()` function is now differentiable so there is no need to import a wrapper. In short, users can take existing functional code and differentiate it without changing much:

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

	return np.sum(abs(data['mode'].amps.sel(mode_index=0).values))

# compute derivative of objective(1.0) with respect to input
autograd.grad(objective)(1.0)

```

Instead of using `jax`, we now use the [autograd](https://github.com/HIPS/autograd) package for our "core" automatic differentiation. Many `tidy3d` components now accept and are now compatible with `autograd` arrays. Because `autograd` is far lighter and has very few requirements, it was made a core dependency of `tidy3d`. 

While we use `autograd` internally, in the future, we will include wrappers so you can use automatic differentiation frameworks of your choice (e.g. `jax`, `pytorch`) without much of a change to the syntax.

The usability of `autograd` is extremely similar to `jax` but with a couple modifications, which we'll outline below.

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

`Autograd` supports fewer features than `jax`. So, for example, the `has_aux` option is not supported in the `jax.grad()`, but one can write their own utilities to implement these features, as we show in the notebook examples.

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

These `td.` classes can be used directly in the differentiable objective functions. Like before, only some fields are traceable for differentiation, and we outline the full list of supported fields in the feature roadmap below.

Furthermore, there is no need for separated fields in the `JaxSimulation`, so one can eliminate `output_monitors` and `input_structures` and put everything in `monitors` and `structures`, respectively. `tidy3d` will automatically determine which structure and monitor is traced for differentiation.

Finally, the regular `web.run()` and `web.run_async()` functions have their derivatives registered with `autograd`, so there is no need to use special web API functions. If there are no tracers found in `web.run()` or `web.run_async()` simulations, the original (non-`autograd`) code will be called.

## Feature Roadmap

### Currently Supported

The following components are traceable as inputs to the `td.Simulation`

- `Box.center`
- `Box.size`
- `PolySlab.vertices`

- `Medium.permittivity`
- `Medium.conductivity`

- `CustomMedium.permittivity`
- `CustomMedium.eps_dataset`

- `GeometryGroup.geometries`

- `PoleResidue.eps_inf`
- `PoleResidue.poles`

- `CustomPoleResidue.eps_inf`
- `CustomPoleResidue.poles`

- `Cylinder.radius`
- `Cylinder.center` (along non-axis dimensions)

- `ComplexPolySlab.sub_polyslabs`

The following components are traceable as outputs of the `td.SimulationData`

- `ModeData.amps`

- `DiffractionData.amps`

- `FieldData.field_components`
- `FieldData` operations:
  - `FieldData.flux`
  - `SimulationData.get_intensity`
  - `SimulationData.get_poynting`

Other features
- support for multi-frequency monitors in certain situations (single adjoint source).
- server-side gradient processing by passing `local_gradient=False` to the `web` functions. This can dramatically cut down on data storage time, just be careful about using this with multi-frequency monitors and large design regions as it can lead to large data storage on our servers.

We currently have the following restrictions:

- Only 500 max structures containing tracers can be added to the `Simulation` to cut down on processing time. To bypass this restriction, use `GeometryGroup` to group structures with the same medium.
- `web.run_async` for simulations with tracers does not return a `BatchData` but rather a `dict` mapping task name to `SimulationData`. There may be high memory usage with many simulations or a lot of data for each.

### To be supported soon

Next on our roadmap (targeting 2.8 and 2.9, fall 2024) is to support:

- `TriangleMesh`.
- `GUI` integration of invdes plugin.

### Finally

If you have feature requests or questions, please feel free to file an issue or discussion on the `tidy3d` front end repository.

Happy autogradding!

## Developer Notes

To convert existing tidy3d front end code to be autograd compatible, will need to be aware of
- `numpy` -> `autograd.numpy`
- `x.real` -> `np.real(x)``
- `float()` is not supported as far as I can tell.
- `isclose()` -> `np.isclose()`
- `array[i] = something` needs a different approach (happens in mesher a lot)
- whenever we pass things to other modules, like `shapely` especially, we need to be careful that they are untraced.
- I just made structures static before any meshing, as a cutoff point. So if we add a new `make_grid()` call somewhere, eg in a validator, just need to be aware.
