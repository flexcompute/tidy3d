# Parameter Sweep Plugin

## Basics

The parameter sweep plugin `tidy3d.plugins.design` is a wrapper designed to make it simple and convenient for `tidy3d` users to define their parameter scans.

In short, users define the dimensions of their design design, as well as the method they wish to explore the design space. These specifications are combined in a `DesignSpace` object.

Then, the user passes a function that defines the input / output relationship they wish to explore. The function arguments correspond to the dimensions defined in the `DesignSpace` and the function outputs can be anything.

The result is stored as a `Result` object, which can be easily converted to a `pandas.DataFrame` for analysis, post processing, or visualization. The columns in this `DataFrame` correspond to the function inputs and outputs and each datapoint corresponds to a row in this `DataFrame`.

The plugin is imported from `tidy3d.plugins.design` so it's convenient to import this namespace first

```py
import tidy3d.plugins.design as tdd
```

##  Function

The first step in using the parameter sweep is to write the design design as a function that we would like to explore. This function could ideally involve a `tidy3d` simulation, but actually does not have to.

For a concrete example, say we are analyzing a system comprised of a set of `n` spheres, each with radius `r`. We could write a function to compute the transmission through this system as follows (with some pseudo-code used for brevity).

```py
def transmission(n: int, r: float) -> float:
	"""Transmission as a function of number of spheres and radius."""
	spheres = make_spheres(num_spheres=n, radius=r)
	sim = td.Simulation(structures=spheres, ...)
	data = web.run(sim, ...)
	return np.sum(data['flux'])
```

This function captures the relationship between `n`, `r`, and transmission `T`. We notice right away that it can be easily split into pre and post processing functions.

```py
def pre(n: int, r:float) -> td.Simulation:
	spheres = make_spheres(num_spheres=n, radius=r)
	sim = td.Simulation(structures=spheres, ...)

def post(data: td.SimulationData) -> float:
	return np.sum(data['flux'])

def transmission_split(n: int, r: float) -> float:
	"""Transmission as a function of number of spheres and radius."""
	sim = pre(n=n, r=r)
	data = web.run(sim, ...)
	return post(data=data)
```

As we'll see, putting it in this form will be useful although it is not necessary.

##  Parameters

Now, we could query our transmission function directly to construct a parameter scan, but it would be more convenient to simply **define** our parameter scan as a specification and have the `tidy3d` wrapper do the accounting for us.

The first step is to define the design "parameters" (or dimensions), which also serve as inputs to our function defined earlier.

In this case, we have a parameter `n`, which is a non-negative integer and a parameter `r`, which is a positive float.

We can construct a named `tdd.Parameter` for each of these and define some spans as well.

```py
param_n = tdd.ParameterInt(name='n', span=(0, 5))
param_r = tdd.ParameterFloat(name='r', span=(0.1, 0.5))

```

Note that the `name` should correspond to the argument name defined in the function.

Also, it is possible to define parameters that are simply sets of quantities that we might want to select, such as:

```py
param_str = tdd.ParameterAny(name="n", values=("these", "are", "values"))
```
but we will ignore this case as it's similar in logic to integer parameters.

By defining our design parameters like this, we are mainly specifying what type, and allowed values can be passed to each argument in our function.

##  Method

Now that we've defined our parameters, we also need to define the procedure that we should use to query the design space we've defined. One approach is to randomly sample points within the design bounds, another is to perform a grid search to uniformly scan the bounds. There are also more complex methods, such as Bayesian Optimization, which are not implemented yet.

For this example, let's define a random sampling of the design parameters with 20 points.

```py
method = tdd.MethodMonteCarlo(num_points=20)
```

## Design Space

With the design parameters and our method defined, we can combine everything into a `DesignSpace`, which is mainly a container that provides some higher level methods for interacting with these objects.

```py
design_space = tdd.DesignSpace(parameters=[param_n, param_r], method=method)
```

Note, we haven't told the `DesignSpace` anything about our function. For that, it has two methods `.run()` and `.run_batch()`, which accept our function.

`DesignSpace.run(f)` samples the design parameters according to the `method` and then evaluates each point one by one. Note that it has to be sequential because of the generality of this approach, in short `f` could be anything so we don't make any assumptions about whether it can be parallelized.

On the other hand, `DesignSpace.run_batch(f_pre, f_post)` accepts our pre and post processing functions and assumes that a simulation is run in between. Because of this, this method can safely construct a `web.Batch` under the hood and run the tasks in parallel, stitching together the final results.

## Results

The `DesignSpace.run()` and `DesignSpace.run_batch()` functions described before both return a `Result` object, which is basically a dataset containing the function inputs, outputs, source code, and any task ID information corresponding to each data point. Note that task ID can only be gathered if using `run_batch` because in the more general `run()` function, `tidy3d` can't know which tasks were involved in each data point.

The `Results` can be converted to a `pandas.DataFrame` where each row is a separate data point and each column is either an input or output for a function. It also contains various methods for plotting and managing the data.


```py
results = design_space.run(transmission)

# print the first 5 data points
print(results.to_dataframe().head())
    r               n    output
0   0.42            5    0.7803
1   0.50            1    0.5513
2   0.22            2    0.6342
3   0.17            3    0.7534
4   0.37            1    0.9162

# plot a hexagonal bin of the results
im = results.hexbin(x="r", y="n", C="output")
plt.show()

```

See our tutorial notebook `Design.ipynb` for more details.
