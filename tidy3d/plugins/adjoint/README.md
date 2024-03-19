# Adjoint Plugin Developer Guide

This guide will explain the internals of the adjoint plugin.

The following flowchart shows the entire process described in this readme for reference while reading.

![FlowAdjoint](../../../docs/source/_static/img/AdjointFlow.svg)


## Context

### The Need for Simulation Differentiation

FDTD simulation is a reliable method for predicting the behavior or performance of a photonic device. Typically, a device is defined as a function of several "design parameters", such as the dimensions of the various components or the material properties of the components. The performance of the device can often be framed mathematically as a function of the result of the simulation or measurement. Thus, `Tidy3D` in its standard form provides a method for (a) defining a device programmatically from the design parameters, (b) simulating the device to give physical quantities of interest, and (c) programmatically post-processing these quantities to determine some figure of merit, which can be used to aid design decisions.

However, in many situations one might like to know more than just the device performance. It can be of great value to know, for example:
- How sensitive is the performance of the device with respect to changes in the various design parameters?
- How would one need to tweak the current design parameters to optimize the device performance?

To answer these questions, it is therefore useful to be able to compute the **derivative** of the device performance with respect to each desgn parameter. For many design parameters, we refer to this as the "gradient".

### Numerical Derivatives

One naive approach to computing the gradient is to use the finite difference method to compute each term of the gradient one by one. More specifically, one may manually perturb each design parameter, simulate the new device, and measure the relative change in the performance to estimate the derivative. The problem with this approach is that
- It is error prone, and requires the selection of a perturbation step size, which requires trial and error.
- As at least one simulation is required to compute each term in the gradient, the number of additional simulations scales linearly with the number of design parameters, making this extremely slow and expensive for many design parameters.

### Automatic Differentiation

Ideally, one would be able to take the gradient of the performance **analytically**, as if it was a simple mathematical function of the inputs. In fact, there exist many tools for taking analytical derivatives of programmatically defined mathematical functions through a technique known as "automatic differentiation" (AD). AD works by tracking which operations were performed to compute the function `f(x)` and then uses the chain rule and knowledge of the derivative of each step to construct `df/dx(x)` automatically. While it would be convenient to apply AD to the calculation of the device performance as a function of the design parameters, we must first define the derivative rules for the actual FDTD simulation step, which is not straightforward and obviously not supported by a generic AD package.

### Adjoint Method

Luckily, there exists a way to differentiate simulations with respect to the parameters that define them, which is commonly referred to as the "adjoint method". The adjoint method exploits knowledge of the mathemtical form of the problem we are trying to solve through FDTD to define a second "adjoint" simulation. Together with our original (often called "forward") simulation, the results of this adjoint simulation can be post processed to give us the gradient we are after. 

The remarkably powerful thing about the adjoint method is that it gives us the gradient with only a single additional simulation required no matter how many design parameters or terms in the gradient. Additionally, the derivatives it returns do not require any custom step size definition, so they are generally more accurate than finite difference approximations.

### Adjoint Plugin

The goal of the `Tidy3D` adjoint plugin is, therefore, to implement the adjoint method in a way that an AD package can make use of. In essence, it defines the "derivative" of the data obtained through a `Tidy3D` simulation with respect to the fields that are contained in the `Tidy3D` `Simulation` definition. This derivative is then fed to an AD package, which can then "close the loop" and let users define and differentiate functions that compute the device performance as a function of some abstract design parameters **through** a `Tidy3D` `Simulation`.

### AD using `Jax`

Rather than write our own AD package to handle everything outside of the FDTD calculation, we make use of the `jax` AD package. The adjoint plugin itself provides `jax`-compatible subclasses of many regular `Tidy3D` components, such as `Simulation`, `Box`, and `ModeData` while also providing a custom `web.run` function that performs the adjoint method under the hood using these custom components, telling `jax` how to perform the steps needed for AD. [Here](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) is the documentation for jax, which provides far more information than I could ever cover in this readme, for more background on how it works, although we will discuss in sufficient detail later on.

### Outline of Guide

In this guide, we will first discuss how `jax` works and what is needed to get `Tidy3D` to integrate with its AD features. Then, we will discuss how the adjoint method is handled within the context of this AD framework. Finally,, we will give a practical guide to using the plugin successfully.

## Automatic Differentiation w/ `Jax`

First, we discuss a bit how `jax`'s AD framework functions and how we have written custom AD rules for FDTD simulation to integrate with it.

### Vector Jacobian Products (VJPs)

In jax, the derivative operations are defined through "vector-Jacobian-product" (VJP) rules. There are several extensive tutorials on this subject on the jax documentation, which are worth a read through for more detail. [This one in particular](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) is a good introduction, but a very basic summary will be provided here.

Say I have defined a function `y = f(x)` and want to define the derivative rule `dy/dx = df/dx(x)`. One might naively think that the approach would be to define a new function `df_dx(x)` that returns the  value of the derivative, associate it with `f`, and be done with it. However, things are a bit more complicated because:
- Most of the time `f(x)` is composed or called along with many other functions and therefore the chain rule necessitates a bit more generality.
- We may not be interested in computing or storing the actual output of `df/dx(x)` as it may be unnecessarily large or complex.

Therefore, in jax, one instead defines a "vector-Jacobian product" function that does not return `df/dx(x)` directly, but rather defines a function of `x` and some value `v` that returns the result of `v^T * df/dx(x)`. We see from this that `v` is the "vector", `df/dx` is the "Jacobian", and the VJP function just returns the product of these quantities. 

Intuitively, we may think of "v" as some data that has a similar structure as our output "y", except contains derivative information from whatever operations are downstream from the `f(x)` call in our application. So for example, if one wants to determine the derivative of a composite function involving `f`, eg. `h(x) = g(f(x)) = g(y)`, jax would first compute `v = dg/dy^T` and then feed this `v` to the VJP of `f` to compute `v^T df/dx = dg/dy df/dx` which is simply equal to `dh/dx` via the regular chain rule. Thus `v` generally stores some derivative of the downstream operations with respect to the output of `f` and using this VJP, we can programmatically stich together the derivative of an arbitrary computation using the chain rule.

### Custom VJPs

`jax` "knows" the VJPs for most mathematical operations found natively in python, such as `+`, `*`, `abs()`. Additionally, it provides a wrapper "`jax.numpy`" for most of the `numpy` operations eg. `np.sum()`, with VJPs defined. This means in most cases, once can write a function using python builtins + `jax.numpy` and have all the VJPs for each step defined without any issue.

However, for some functions, `jax` simply doesn't know the VJP, so we must define it ourselves and register it with jax. There is a thorough tutorial [here](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) explaining this process and the many ways to perform it, but below is a snippet paraphrased from this tutorial showing the basic idea for defining the vjp for a custom function `f`:

```py
from jax import custom_vjp

@custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
# Returns primal output and residuals to be used in backward pass by f_bwd.
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, v):
  cos_x, sin_x, y = res # Gets residuals computed in f_fwd
  return (cos_x * v * y, sin_x * v)

f.defvjp(f_fwd, f_bwd)
```

To highlight, we first register `f` as requiring a `custom_vjp`, then we define a "forward" rule for `f`, which just returns the output we expect + some additional information we might want to cache for later. Then, we define a "backward" rule for `f`, which defines the VJP as a function of our tuple of cached values from the forward pass (`res`) and the "vector" `v`, which looks somewhat like the return of `f` but with derivative information stored in it. The return value of this backwards function is a tuple where each element encodes `v` times the derivative of `f` with respect to the corresponding element of the input arguments of `f`. Finally, we register the two `vjp` functions with `f`.

While this is a bit difficult to digest, it is the core functionality provided by the adjoint plugin and is therefore worth understanding for a simple case. 

### The VJP of `web.run()`

The adjoint plugin actually only provides a single `VJP`, which is defined for the `sim_data = web.run(sim)` function. The plugin therefore implements two functions `sim_data, res = run_fwd(sim)` and `sim_vjp = run_bwd(res, sim_data_vjp)`, which respectively perform the:
- forward simulation: original simulation + some extra information needed for computing derivatives later.
- adjoint simulation: simulation encoding derivative information passed from downstream in the AD process.
and postprocesses the results into derivatve information for `jax`.

The backward vjp function defined for `web.run()` accepts and returns quite abstract objects. First, the "v" provided is a `SimulationData` object storing the derivatives of the downstream operations w.r.t. the data stored in the various `MonitorData` objects. So for example, if the eventual output of the function being differentiated depends on a single amplitude in the `ModeData` of the forward pass, the adjoint `SimulationData` will contain a `ModeData` object with all 0 elements except for the coordinates associated with that amplitude. The value of that amplitude data will store directly the derivative of the function output with respect to that amplitude. 

As the return value of the `web.run()` must look like the derivative of its input arguments, this function returns a tuple of length 1 storing a `Simulation` where the contents of each field are simply the derivatives that field w.r.t. the outputs of the function being differentiated. 

This concept is hard to understand but is the crucial idea behind the workings of the plugin so it's again worth spending some time to digest. To reiterate, in its essence, the goal of the adjoint `web.run()` is to propagate the derivative of the function w.r.t. the output data (stored as a `SimulationData`) into the derivative of the function w.r.t. the `Simulation` fields (stored as a `Simulation`).

### Where do derivatives come from?

As discussed, the goal of our VJP for `web.run()` is to propagate the derivative information from within the `jax`-provided `SimulationData` to the derivative w.r.t. the `Simulation` fields. This is where the adjoint method comes in. We know that the adjoint method can compute this derivative information through another `web.run()` call. So our job, given some `SimulationData` from `jax` is to:
- Construct the proper adjoint `Simulation`.
- Run it to get an adjoint `SimulationData`.
- Process both the cached `SimulationData` from the forward run with the adjoint run to compute the derivative information w.r.t. the `Simulation` fields.
- Construct a `Simulation` storing these derivatives and return.

In summary, the adjoint method tells us that for each output that our eventual function depends on, we must construct a corresponding current source to put into our `Simulation`. Practically speaking, when given a derivative `SimulationData` object, we loop through each non-zero data element, construct a corresponding source, and add it to our adjoint `Simulation`. The mapping between data elements and adjoint sources requires some definition, which we will discuss in the next major section, but for now it's just worth noting that we have these mappings defined as methods of our data classes.

Finally, once we have run our adjoint `Simulation`, we must process the data from each simulation into derivative rules. To perform this, we have also defined some mappings for the `Simulation` components (`Structure`, `Box`, `Medium`) that accept forward and adjoint simulation data, and return a copy that processes this data into derivative information. For example, to compute the derivative of a `Structure` with respect to the `.permittivity` of its `.medium`, the adjoint method says that one must take the dot product of the adjoint and forward fields and integrate this quantity over the volume of the structure `.geometry`. Therefore, the `Medium` has a method that computes this integral and stores it in its `.permittivity` field for further processing.

Finally, it is worth noting that the custom derivative for `web.run()` is not defined in the standard `tidy3d.web` location, but rather a wrapper is provided in the plugin. This decision was made to adhere to the design principle of the main components being agnostic to the plugin contents. This wrapper simply calls `web.run()` under the hood after converting the jax types to regular tidy3d types and then converts the results back into jax types. We will discuss this in more detail in the following section.

## `Jax` Nuts and Bolts

With that introduction to the basic flow for defining the VJP of `web.run()`, now we will discuss a bit about some basic things that are needed to get this working practically using our existing `Tidy3D` objects.

### Registering `Tidy3D` components with `jax`
First, `jax` AD works only with only very basic datastructures, called "pytrees", and therefore does not know how to handle functions accepting and returning `pydantic.BaseModel` subclasses like our regular `Tidy3D` components. Pytrees are essentially nested datastructures in the form of regular python objects, such as tuples, dictionaries, and raw numeric data. For a much more detailed discussion on pytrees, see the [jax documentation page on them](https://jax.readthedocs.io/en/latest/pytrees.html).

To get `Tidy3D` components to be recognized by jax, we must define rules for how each `Tidy3D `component is converted into a pytree (plus some arbitrary auxiliary data) and how a pytree (plus some auxiliary data) is converted back into a `Tidy3D` component. As discussed in the link on pytrees just above, we can perform this registration by:
- decorating our `Tidy3D` component with `@jax.tree_util.register_pytree_node`.
- defining a `.tree_flatten(self)` method to return the pytree plus any auxiliary information we might need to unflatten.
- defining a `.tree_unflatten(cls, aux_data, children` `@classmethod` to return an instance of self given that auxiliary information and the pytree (`children`).

### `Jax` Subclasses of `Tidy3D` objects

To do this registration in a systematic way, the adjoint plugin provides several subclasses of `Tidy3D` components that also inherit from a `JaxObject` base class, providing these methods. If a `Tidy3D` component is named `X`, the `jax`-compatible subclass inherits from `X` and `JaxObject` and is named `JaxX` by convention. Note also that `JaxX` classes must also be decorated with `@jax.tree_util.register_pytree_node` to work properly.

The derivative information passed by `jax` is often in the form of very obscure datastructures. Therefore, if one were to try to directly put this information in a regular `Tidy3D` field, such as `Medium.permittivity`, `pydantic` would complain.  Therefore, for any `JaxX` objects that may contain `jax` tracers, one needs to overwrite the corresponding `pydantic.Field` and make it allow type of `Any` to avoid validation errors. Finally, we must mark this field as potentially containing a `jax` tracer by setting the `jax_field=True` in `pd.Field()`.

The `JaxObject` provides a few convenience methods make use of the presence of this `jax_field` to determine how to correctly `tree_flatten` and `tree_unflatten` any subclasses. Luckily, we do not need to define these operations on subclasses except for `JaxDataArray`, which will be discussed later.

### Conversion Between `JaxTidy3D` and `Tidy3D` Components

Finally, there are some methods in the `JaxX` subclasses for converting `JaxX` to `X` and vice versa. This is required, for example, to run an actual simulation using the standard `web.run()` function, requiring a `JaxSimulation` and its contents to be converted to a regular `Simulation`. It follows also that the return `SimulationData` must be converted back to a `JaxSimulationData` for handling by `jax`. These methods are a bit hard to understand as they operate mainly on the `dict` level data and are therefore one area of possible simplification down the line if we can figure out how to do conversion more naturally.

### The One Exception: `DataArrays`

Unfortunately, all `Tidy3D` components are not handled so simply. It turns out that `jax` datastructures do not work when added to `xarray.DataArray` containers. Therefore, we needed to find a replacement to the `DataArray` class in regular Tidy3D to use in our `JaxMonitorData` fields. To solve this problem, the adjoint plugin introduces a `JaxDataArray`, which stores `.values` as a `jax` `Array` (basically a `jax` version of `numpy` arrays) and `.coords` as a dictionary of coordinates with a dimensions string mapping to a list of values. 

The `JaxDataArray` may freely store `jax` values and tracers and implements a few basic emulations of `xarray.DataArray` objects, such as `.sel` and `.isel`. Note that at this time `.interp` is not supported, but I think we could consider implementing it later. A user could, in principle, wrap `.sel` to do the interpolation themselves, but it was not considered at this stage because some trials to implement it myself ended badly.

The `JaxDataArray` inherits directly from `Tidy3dBaseModel` alone and therefore implements its own pytree flattening and unflattening operations. Luckily, these operations are quite trivial. There are also a set of custom validations in the `JaxDataArray` to ensure it is set up and used properly as it does not natively provide nearly as strict argument checking as its `xarray.DataArray` counterparts.

## Adjoint Method

With an understanding on how `jax` is integrated into the plugin through subclasses of `Tidy3D` components, now we can finish the technical discussion with an overview of how the adjoint components are implemented.

### Gradient Monitors

As discussed, computing the derivative information through the adjoint method requres performing integrals over the forward and adjoint field distributions, as well as the permittivty distributions in the simulations. As such, when running the forward and backward passes of the `web.run()` function, we must add monitors to capture this information and pass it to the VJP makers for each of the jax-compatible fields in the `JaxSimulation`.

For this purpose, the `JaxSimulation` class provides an `add_gradient_monitors()` method, that loops throughout any jax-compatible fields and calls their respective methods for generating whatever `Monitor`s are needed for the adjoint method to work. For example, the `JaxStructure` has a `.make_grad_monitors()` method, that returns a `FieldMonitor` and `PermittivityMonitor` spanning its volume (+ some dilation).

These gradient monitors are generated and added to both forward and adjoint simulations before they are run and the data contained within are processed at the final stage when computing VJP rules for the `pydantic.Fields()` of the `JaxSimulations`.

### Adjoint Source Creation

As mentioned, the derivative `SimulationData` received by the backwards pass of `web.run()` contains `JaxMonitorData` with data values corresponding to downstream derivative information. Following the math of the adjoint method, this derivative data must be converted to current sources to use in the adjoint `Simulation`. As such, each `JaxMonitorData` subclass has a `.to_adjoint_sources()` method, which returns a list of adjoint `Source` objects given whatever data is stored within. 

This method must be implemented separately for each subclass, eg. `JaxModeData` -> `[ModeSource]` and `JaxDiffractionData` -> `[PlaneWave]`. The basic idea for all of the methods, however, is to loop through each non-zero data contained within and use the coordinate and value to determine the proper adjoint source. Note: there is a factor of `1j` built into the amplitude to account for the phase shift between a specified `GaussianPulse` and the corresponding  `J(r,t)` that is injected into the simulation.

### Adjoint Simulation Creation

In the previous section, we discussed how adjoint sources are created from a single `JaxMonitorData`. Now we discuss how to construct an entire adjoint `Simulation` from a derivative `JaxSimulationData`. For this, `JaxSimulationData` contains a single convenience method `JaxSimulationData.make_adjoint_simulation()` that performs the following steps:
- Copy the original simulation.
- Flip all `bloch_vec`s for any bloch boundaries (to model the adjoint physics properly).
- Generate and concatenate all the adjoint sources for each `JaxMonitorData` as described in the previous section.
- Replace any sources from the original simulation with these adjoint sources.
- Replace any monitors from the original simulation with whatever field and permittivity monitors are required for computing the adjoint integral over each `Structure`.

### PostProcessing Forward and Adjoint Fields for VJP Creation

With the forward and adjoint simulations run and all of the field and permittivity data stored for each `JaxStructure` in the simulation, the final stage of the pipeline involves converting these gradient monitor data into VJP values to be stored in the `JaxSimulation` returned by the backwards pass of `web.run()`. 

These VJPs are determined by performing integrals over the monitor data using the `Structure` geometry. For example, the derivative rule for the `Structure.medium.permittivity` involves a volume integral of the dot product of the E fields over the `Structure.geometry` and the derivatve rule for the `Structure.geometry` involves surface integrals over the `Geometry` surfaces. These integrals are defined respectively in the `JaxMedium.store_vjp()` and `JaxBox.store_vjp()` objects and are combined together in the `JaxStructure.store_vjp()`. The return value of these methods is a copy of the original instance where the values stored in the jax-compatible fields are the derivatives of the downstream function outputs w.r.t. the field. 

At the final stage of the backwards pass of `web.run()`, the `JaxSimulation.store_vjp()` loops through each of its `JaxStructure`s and replaces that structure with the return value of its respective  `.store_vjp()` call.

## Using the Plugin

Now we dive into the more surface-level use of the plugin and the design decisions. To define our notation, we consider the user wants to define a function `f(p)` that

- Defines a `Simulation` "`sim`" as a function of a vector of design parameters `p`.
- Runs that simulation to get some `Simulation` "`sim_data`".
- Post processes `sim_data` into a `float` value.

in pseudo-code, this might look something like:

```py

def f(p):
    sim = make_sim(p)
    sim_data = run(sim)
    return postproces(sim_data)
```

Once `f(p)` is defined, the user then wants to compute `df/dp`.
Here we show how to set up `f` and compute and use `df/dp`  using the adjoint plugin.

### Defining Jax Simulation

To use the adjoint plugin, the user must first define their simulation using `Jax` subclasses for any Tidy3D components that may depend on `p`. Right now, we only support differentiating with respect to `Structure` objects with `.medium` of `Medium` and `.geometry` of `Box`. Therefore, any structures of this form depending on `p` must be constructed using `JaxSimulation`, `JaxMedium`, and `JaxBox` objects, respectively. Importantly, the simulation itself should also be a `JaxSimulation`.

#### Input Structures

While the `JaxSimulation` can contain regular `Structure` instances in its `.structures` tuple, for technical reasons, it was much more convenient to separate the `JaxStructures` into their own new field, which is called `JaxSimulation.input_structures`. Therefore, any `JaxStructures` depending on `p` must be added to this field in the `JaxSimulation` and not to `.structures`. 

It is worth noting that:
- `JaxStructures` in `.input_structures` can not overlap as the gradients will be inaccurate. This is validated in `JaxSimulation`.
- `.input_structures` are added **after** regular `.structures` and may overlap with regular `.structures`. One may think of the process of building the material grid as each `Structure` in `.structures` being added to the `Simulation`, followed by each `JaxStructure` in `.input_structures`.

#### Output Monitors

For technical reasons it was also more convenient to separate out the monitors that the function `f(p)` may depend on from the regular monitors. Therefore, a `JaxSimulation` may have regular `.monitors` for eg plotting or diagnostics, but they also may have `.output_monitors`, which must be `Monitor` types with corresponding `JaxMonitorData`, such as `ModeMonitor` and `DiffractionMonitor`.

#### Gradient Monitors

While they are hidden from normal use, the `JaxSimulation` contains fields storing `.grad_monitors` and `.grad_eps_monitors`. These store the `FieldMonitor` and `PermittivityMonitor` for each `JaxStructure` in the `.input_structures` tuple, with a 1 to 1 correspondence, and are only used internally.

#### Other Differences

Finally, the `JaxSimulation` has a few other differences with the regular `Simulation` class.
- It validates that `subpixel=True`, because otherwise adjoint gives inaccurate and ill-defined results.
- It has an `fwidth_adjoint` field, which, if specified, overwrites the `fwidth` of the adjoint source with a custom value. If not specified, the adjoint sources' `fwidth` is determined from the average of the `fwidths` contained in `.sources`.
- To convert a `JaxSimulation` to a regular `Simulation`, one may call  `.to_simulation()` method, which returns a tuple of the `Simulation` and a datastructure containing extra information needed to reconstruct the `JaxSimulation` with `JaxSimulation.from_simulation()`. So a conversion would look like `sim=jax_sim.to_simulation()[0]`. If desired, we could easily add a `.simulation` `@property` to just return the `Simulation` part for convenience.
- The `JaxSimulation` objects have their `.plot` and `.plot_eps` methods overwritten for convenience, which calls `.to_simulation()` and plots the result.

### Adjoint `web.run` function

The adjoint plugin provides a wrapper for the regular `tidy3d.web.api.webapi.run()` function that has the VJPs defined. This function can be imported as `from tidy3d.plugins.adjoint.web import run` and functions exactly the same as the original `web.run()` in all respects. Eg. it takes the same `**kwargs` and passes them to `web.run()`. 

If a user forgets to use this custom `run()` function, the simulation will error as it will attempt to upload a `JaxSimulation` directly to the server. For this reason, I think we should consider adding a check to `web.upload()` that the `type` field is `'Simulation'`. To get around this, the adjoint wrapped `web.run()` first converts the `JaxSimulation` to a regular `Simulation`, uses regular `web.run()`, and converts the result back to a `JaxSimulationData` as there is no webapi to run a `JaxSimulation` object currently.

### Working with JaxSimulationData

The `jax`-compatible `JaxSimulationData` subclasses is more similar to the regular `SimulationData` with just one key difference:
`JaxSimulationData` has `.data` and `.output_data` tuples for storing the `MonitorData` and `JaxMonitorData` corresponding to the respective `.monitors` and `.output_monitors` in the original `JaxSimulation`. Similarly, it has `@properties` of `.monitor_data` and `output_monitor_data` for generating dictionaries mapping the monitor names to these quantities. 

It is worth noting that the `__getitem__` method is overwritten to select by monitor name from both of these tuples, so the user doesn't need to know the previous information to work with it using eg. `sim_data['mode']` if desired.

### Putting it all together

So given all of this and referencing the original `def f(p)` function, the user would need to make sure to write their funtion as

```py
def f(p):

    # make_sim returns JaxSimulation with appropriate `JaxStructures`
    jax_sim = make_sim(p)

    # run is the run imported from td.plugins.adjoint.web
    jax_sim_data = run(jax_sim)

    # postprocess initially grabs the data using
    #      sim_data.output_data[mnt_index]
    #   or sim_data.output_montior_data[mnt_name]
    #   or sim_data[mnt_name]
    return postproces(sim_data)
```


## Differentiating using `jax`

With `f(p)` defined properly using the guidelines in the previous section, now we'll talk about how to use `jax` to properly differentiate our function.

### Derivative Functions

`jax` provides several functions for differentiation, here i'll talk about a few of the most useful to know and their pitfalls.

#### `grad`

`g = jax.grad(f)` [(documentation)](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html)
returns a function `g` that when evaluated `g(p)` returns the gradient of `f` with respect to `p`. As referenced in the documentation, it is important to note a few things:
- If `p` contains `complex`-valued types, one would need `holomorphic=True`
- If `p` contains `int` values, it will error, so convert values in `p` to `float` or `complex` before passing to `g`.
- If `f` has more than one argument, eg `f(p1, p2, p3)`, `jax.grad` will just take the gradient with respect to the first argument by default. In general one needs to pass an `arg_nums` `tuple` to `jax.grad` to tell it which arguments to take the gradient with respect to. Otherwise, the error can be obscure if, eg you try to unpack `g1, g2, g3 = g(p1, p2, p3)`.

There are a few other arguments in `jax.grad` that should all be compatible with the adjoint plugin and are worth exploring.

#### `value_and_grad`

Another useful function provided by jax is `jax.value_and_grad`. This lets you compute the value `f(p)` and the gradient `g(p)` at the same time without needing to recompute `f(p)`. To illustrate this, consider

```py
import jax

# option 1: grad
g = jax.grad(f)
fp = f(p) # evaluates f(p)
gp = g(p) # re-evaluates f(p) and its backwards pass in the VJP

# option 2: value_and_grad
v_and_g = jax.value_and_grad(f)
fp, gp = v_and_g(p) # evaluates f(p) and backwards pass only once

```
for situations where you might want `f(p)`, `value_and_grad` is a better choice as you'd need to run 2 simulations (forward + adjoint) instead of three (2x forward + adjoint).

#### `jax.numpy`

Finally, it's worth repeating again that if `numpy` operations are involved in the calculation of `f(p)`, one must use `jax.numpy` instead of `numpy`. Otherwise, `jax` will error in a very obscure way and it could take a user a while to understand what went wrong. As an example.

```py
import jax.numpy as jnp

def f(p):
    permittivity = jnp.square(p)
    sim = make_sim(permittivity)
    sim_data = run(sim)
    amp = sim_data['mode'].amps.sel(direction="+", f=f0, mode_index=0)
    return jnp.square(abs(amp))

```

## Conclusion

That summarizes just about all of the adjoint plugin. For more details, it is highly recommend to check out the `jax` documentation, especially regarding automatic differentiation, as well as the tutorial notebooks in `tidy3d-docs`. Happy autogradding!
