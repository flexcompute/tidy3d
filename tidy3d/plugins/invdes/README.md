# Inverse Design Plugin


## Introduction

The inverse design plugin (`invdes`) is a wrapper that makes it simpler to set up and run inverse design in `tidy3d`. This plugin allows for high level specification of an inverse design. This specification then gets compiled into code written using the `adjoint` plugin. The `adjoint` plugin can still be used, but `invdes` is sufficient for most practical problems where that level of granularity is not desired and can sometimes even slow down development.

In the coming months, we will develop a graphical user interface (GUI) version of the `invdes` plugin, which will make it simple to set up and run inverse design plugins from our web-based GUI using a similar framework as that outlined here.

## Workflow

First, let's discuss the general workflow of `InverseDesign` plugin. For reference, the following image describes setting up an running an inverse design from left to right.

![FlowInvdes](../../../docs/source/_static/img/InvdesFlow.png)

### Importing the Plugin

The plugin is imported from `tidy3d.plugins.invdes` and we conventionally import this `as tdi`. So the following will assume a `tdi.` namespace for `invdes` components, a `td.` namespace for regular `tidy3d` components, and `tda.` for `adjoint` components.

### Components

An inverse design project is defined within the `tdi.InverseDesign` object. This object contains the following fields

#### simulation

A `td.Simulation` describing the simulation before any of the optimization objects are added. This is sometimes referred to as the "base" simulation in our existing tutorial notebooks. You can think of if as the "static" component of our simulation.

#### design region

A `tdi.DesignRegion` component is introduced in the `invdes` package to describe the region that is being updated over the course of the optimization. For now, only topology optimization (pixel-based permittivity grid) is supported, but we plan to add shape and level-set design regions soon.

This `tdi.DesignRegion` can be thought of as a convenient way to generate a `td.Structure` when passed some optimization parameters. More specifically, in the background, it gets converted to a `tda.JaxStructureStaticGeometry`, which contains a `tda.JaxCustomMedium`. 

The `tdi.DesignRegion` contains fields that capture the geometric information (`center`, `size`), as well as some of the higher level description of the permittivity grid, such as the number of pixels in each dimension and the permittivity min and max bounds.

As in many of our demos, there are often several steps involved in converting an array of optimization parameters into an array of relative permittivity values for the custom medium. In the `invdes` package, we provide a set of "transformations" that wrap those in the `tidy3d.plugins.adjoint.utils`. For topology optimization, we support `tdi.BinaryProjector`, `tdi.ConcicFilter`, and `tdi.CircularFilter`. These are similar to their equivalent components in `adjoint`, except some of the lower level fields, such as the `design_region_dl` are automatically set by the `invdes` plugin. When these components are added to the `DesignRegion.tranformations` `tuple`, they are implicitly `.evaluate`d on the parameters from beginning to end, specifying the full transformation from optimization parameters to relative permittivity.

Similarly, we support the addition of `.penalties` to the `td.DesignRegion`. For topology optimization, there is a `tdi.ErosionDilationPenalty` that works like the `adjoint` equivalent but sets the `pixel_size` automatically. If several penalties are added, their contributions to the objective function will be summed, with an optional weight as set by the corresponding `tdi.DesignRegion.penalty_weights` list.

> Internal Note: i worked a bit on getting user-defined transformations and penalties by exposing a base class. But it's a little strange with pydantic. but this might be a feature.

#### output monitors

Next, like in the `adjoint` plugin, we need to specify a set of output monitors which our objective function will depend on. We pass these monitors directly to the `tdi.DesignRegion.output_monitors` field.

> Internal Note: maybe it makes sense to put the monitors in `DesignRegion.simulation` and just make `tdi.output_monitors` a `tuple` of `str` corresponding to the monitor names?


#### optimizer

Finally, we need to specify an `tdi.Optimizer`, which stores information about how we run the optimization procedure. For now, there is just a single optimizer which implements the "adam" method, similar to how it's done in our tutorial notebooks. The `Optimizer` is passed information about the step size, number of steps, as well as the optimizer parameters.

> Internal Note: The `tdi.Optimizer` is not really necessary as it stands with just a single option, but it is nice for it to be separated for later if we do introduce different optimizers.

#### inverse design

All of these components get put into a single `tdi.InverseDesign` object, which acts as a container with some methods that let you run this inverse design problem. Note that there are some parallels to the `tidy3d.plugins.design` plugin which has a `tdd.Design` object that serves a similar purpose.

### running the inverse design

Now that we've set up our `tdi.InverseDesign`, we want to run it.

#### post-processing function

To fully define our objective function, we need to construct a "post-processing function". This regular python function accepts a `tda.JaxSimulationData` as an argument (corresponding to the result of running the `tda.JaxSimulation` generated by the `tdi.InverseDesign`) and should return a `float` indicating the contribution to the objective function from analyzing the `tidy3d` simulation results.

> Note: the convention we use is for the optimizer to **maximize** this value.

The full objective function we feed to the optimizer is the result of this post-processing function minus the values computed from the `tdi.DesignRegion.penalties`.

> Internal Note: I was considering writing a new set of components that try to capture the possible post-processing of monitors. For example, grab monitor by name, grab it's dataset (eg. `.amps`), `sel` various elements, and then apply a series of operations (`Abs`, `Square`). Then we could specify the postprocessing function as a list of these. I wonder if this is general enough though. For example, what if I want to grab the power of 2 amplitudes and weigh them by some custom amount depending on the step number? 

#### run API

Finally, we can call `

```py
res = InverseDesign.run(f_pp, task_name="inverse_design", ...)`
```

passing our postprocessing function `f_pp` and any keyword arguments to `tda.web.run()`. This will construct an objective function that generates the `JaxSimulation` given some design parameters, runs the simulation through `tda.web.run()` call, and then performs the post-processing of the result using the post-processing function and any penalties. After this function is generated, we use it in a regular gradient-descent optimization loop using `optax`, similar to how it's done in our `adjoint` plugin tutorial notebooks. Information about the optimization is optionally printed to STDOUT and a user can pass a callback function to customize the output.

The returned `res` object is a `tdi.OptimizerResult`, which is a container storing the history of the optimization, including the `optax` optimization states. This object provides several convenience methods for plotting and grabbing data.

If the user runs an optimization and then wants to continue the process, they can call

```py
InverseDesign.continue_run(res, f_pp)
```

, passing the `InverseDesignResult`. It will then pick up where it was before using the history stored in `res`.

## API example

For an overview of the entire API, below is an example, copied and modified from the `test_invdes.py` test.

```py
import numpy as np
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import tidy3d.plugins.invdes as tdi
import matplotlib.pyplot as plt

FREQ0 = 1e14
L_SIM = 2.0
MNT_NAME = "mnt_name"
PARAMS_SHAPE = (18, 19, 20)
PARAMS_0 = np.random.random(PARAMS_SHAPE)

simulation = td.Simulation(
    size=(L_SIM, L_SIM, L_SIM),
    grid_spec=td.GridSpec.auto(wavelength=td.C_0 / FREQ0),
    sources=[
        td.PointDipole(
            center=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FREQ0 / 10),
            polarization="Ez",
        )
    ],
    run_time=1e-12,
)

mnt = td.FieldMonitor(
    center=(L_SIM / 3.0, 0, 0), size=(0, td.inf, td.inf), freqs=[FREQ0], name=MNT_NAME
)

"""make a design region and call some of its methods."""
design_region = tdi.TopologyDesignRegion(
    size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
    center=(0, 0, 0),
    eps_bounds=(1.0, 4.0),
    symmetry=(0, 1, -1),
    params_shape=PARAMS_SHAPE,
    transformations=[
        tdi.CircularFilter(radius=0.2, design_region_dl=0.1),
        tdi.BinaryProjector(beta=2.0, vmin=0.0, vmax=1.0),
        tdi.ConicFilter(radius=0.2, design_region_dl=0.1),
    ],
    penalties=[
        tdi.ErosionDilationPenalty(length_scale=0.2, pixel_size=0.1),
    ],
    penalty_weights=[0.2],
)

design_region.material_density(PARAMS_0)
design_region.penalty_value(PARAMS_0)

"""make an inverse design"""
design = tdi.InverseDesign(
    simulation=simulation,
    design_region=test_design_region(),
    output_monitors=[mnt],
    params0=np.random.random(PARAMS_SHAPE).tolist(),
    history_save_fname="tests/data/invdes_history.pkl",
)

"""Define the post processing function."""
def post_process_fn(sim_data: tda.JaxSimulationData, scale: float = 2.0) -> float:
    """Define a postprocessing function"""
    intensity = sim_data.get_intensity(MNT_NAME)
    return scale * jnp.sum(intensity.values)

"""Make an optimizer"""
optimizer = tdi.Optimizer(
    params0=PARAMS_0,
    learning_rate=0.2,
    num_steps=3,
)

"""running the inverse design"""
result = optimizer.run(design=design, post_process_fn=post_process_fn, task_name="blah")

"""continuing an already run inverse design."""
result_continued = optimizer.continue_run(result=result, post_process_fn=post_process_fn, task_name="blah")

"""Grabbing information from a result and exporting."""
final_params = result.params
final_simulation = result.get_final("simulation")

result.plot_optimization()

gds_layer_dtype_map = {td.Medium(permittivity=4.0): (2, 1), td.Medium(): (1, 0)}
result.to_gds_file("sim_final.gds", z=0, gds_layer_dtype_map=gds_layer_dtype_map)

sim_data_final = result.sim_data_final(task_name="final")
```

let me know if there's any feedback!


## Notes / Testing ground


### Packaging results

#### Option 1: like `SimulationData`

```py
class InverseDesign:
    pass

class Result:
    data : ...
    design : InverseDesign

def run(design: InverseDesign) -> Result:
    pass

```

Pros:
* `Result` has `InverseDesign` information

Cons:
* Need an external `run()` function.
* Unnecessary coupling between design and result (what if not 1->1?)
* What about a changing design? eg step size? which one goes in the result?

#### Option 2: like `Design`

```py
class InverseDesign:
    pass
    
    def run() -> Result:
        pass

class Result:
    data : ...

```

#### Option 3: optimizer that captures running stuff.

```py
class InverseDesign:
    pass

class Result:
    data: ...
    design: InverseDesign

class Optimizer:
    num_steps : int

    def run(design: InverseDesign) -> Result:
        pass

    def continue_run(result: Result) -> Result:
        pass

Pro:
* Cleanest for continuing run
* Reduces coupling

Cons:
* Still doesnt 100 handle changing design
```
Some use cases:

### single run

```py
results = run(design)
results = design.run()
results = optimizer.run(design)
```

### continue running

```py
results = run_continue(design, results) <- X do we put both results in?
results = design.continue_run(results)
results = optimizer.continue_run(results) <- cleanest, re-use the design
```

### change the design and continue running

```py
ff
```

### plot the design with the results?

```py
ff
```

## Design QUestions
* separate `optimizer` with a `.run(design) -> result`? 
    seems like a good way to go
* make `history` explicit in the result?
    would be better
* put penalties in optimizer?
    probably not, penalties are more specific to the type of DesignRegion.

