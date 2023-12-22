# Tidy3d Components

This file explains the various `tidy3d` components that make up the core of the simulation file definition.

## Background

### Base

All  `tidy3d` components are subclasses of the `Tidy3dBaseModel` defined in `base.py`. 
This `Tidy3dBaseModel` itself is a subclass of [`pydantic.BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/) which makes them `pydantic` models.
In `Tidy3dBaseModel`, we also specify some configuration options, such as whether we validate fields that change after initialization, among others.

### Types

Input argument types are largely defined in `types.py`
The `typing` module to define basic types, such as `Dict`, `List`, `Tuple`.
More complex type definitions can be constructed from these primitive types, python builtin types, `pydantic` types, for example

```python
Coordinate = Tuple[float, float, float]
```
defines a coordinate in 3D and
```python
Indices = Tuple[pydantic.nonNegativeInt, ...]
```
defines a tuple of non-negative integers, for example, a tuple of indices.

This file provides a way for one to import the same type into multiple components.
Often times, though, if a type is being used once, it is just declared in the component definition itself.

### Validators

`validators.py` defines `pydantic` validators that are used in several of the components.
These validators let one define functions that check whether input arguments or combinations of input arguments meet certain acceptance criteria.
For example, there is a validator that checks whether a given `Box` class or subclass is planar (ie whether it's `size` attribute has exactly one `0.0`).

### Constants

Several physical constants are defined in `constants.py` as well as the default value for various parameters, like `td.inf`.

### Abstract Base Classes

You may see `ABC` being used in a subclass definition.
This signifies that the class is an "abstract base class".
For all intents and purposes, this means that the class exists to give some organizational structure, but is not intended to be used directly.
For example, `AbstractMedium(Tidy3dBaseModel, ABC)` is used to define common characteristics of both dispersive and non-dispersive media, but isn't intemded to be used in a simulation.

The `@abstractmethod` decorator is similarly used to indicate that the method is intended to be implemented in the subclasses of the `ABC`, but not used or defined directly in the base class.

## Component Structure

Here is a map of the inheritance of various tidy3d components.

```
pydantic.BaseModel
	Tidy3dBaseModel
		Geometry
			Box
				Simulation
				Source
					CurrentSource
					DirectionalSource
						PlaneWave
						GaussianBeam
				Monitor
					FieldMonitor
					FluxMonitor
					ModeMonitor
			Sphere
			Cylinder
			PolySlab
		PMLLayer
		AbstractMedium
			Medium
			DispersiveMedium
				PoleResidue
				Sellmeier
				Lorentz
				Debye
		Structure
		SourceTime
			GaussianPulse
			CW
		Sampler
			TimeSampler
			FreqSampler
		Mode
```

### Geometry

The `Geometry` component is used to define the layout of objects with a spatial component.

Each `Geometry` subclass implements a `._bounds(self)` method, which returns the min and max coordinates of a bounding box around the structure.

The base class also implements a `._instersects(self, other)` method, which returns True if the bounding boxes of `self` and `other` intersect.
This is useful for error checking of the simulation.

The following subclasses of `Geometry` are importable and often subclassed in the rest of the code.
- `Box(center, size)`
- `Sphere(center, radius)`
- `Cylinder(center, radius, length, axis)`
- `PolySlab(vertices, slab_bounds, axis)`

### Simulation

The `Simulation` is the core datastructure in `tidy3d` and contains all of the parameters exported into the .json file.

`Simulation` inherits from `Box` and therefore accepts `center` and `size` arguments.

It also accepts many arguments related to the global configuration of the simulation, including:
- `grid_spec` (defines the discretization).
- `medium` (the background medium).
- `run_time`
- `pml_layers` (a list of three `PML(profile, num_layers)` objects specifying the PML, defined in `pml.py`).
- `symmetry`
- `courant`
- `shutoff`
- `subpixel`

Crucially, the `Simulation` also stores three dictionaries containing important `tidy3d` components.
The keys of these dictionaries are the names of the components and the values are instances of the components.

- `structures`, a list of `Structure()` objects, defining the various objects in the simulation domain.
- `sources`, a dictionary of `Source()` objects, defining the current sources in the simulation domain.
- `monitors`, a dictionary of `Monitor()` objects, defining what data is being measured and where.

![Call Structure](../../docs/_static/img/diagram_Simulation.png)

#### Validations

Upon initialization, the simulation checks whether any of the objects are completely outside of the simulation bounding box, at which point it will error.
Other checks may be added in future development.

#### JSON Operations

The `Simulation` can be exported as .json-like dictionary with `Simulation.json()`
The schema corresponding to `Simulation` can be generated with `Simulation.schema()`

## Medium

The `AbstractMedium()` base class define the properties of the medium of which the simulation and it's structures are made of.

`AbstractMedium()` also contains a `frequency_range` tuple, which specifies the frequency range of validity of the mode, default is -infinity to infinity.

`AbstractMedium()` subclasses must implement a `eps_model(self, freq)` method,  which returns the complex permittivity at a given frequency.

### Dispersionless Media

A Dispersionless medium is created with `Medium(permittivity, conductivity)`.

The following functions are useful for defining a dispersionless medium using other possible inputs:
- `AbstractMedium.nk_to_eps_sigma` (convert refractive index parameters (n, k) to permittivity and conductivity).
- `Medium.from_nk` (convert refractive index parameters (n, k) to a `Medium()` directly).
- `AbstractMedium.nk_to_eps_complex` (convert refractive index parameters (n, k) to a complex-valued permittivity).
- `AbstractMedium.eps_sigma_to_eps_complex` (convert permittivity and conductivity to complex-valued permittiviy)

### Dispersive Media

Several Dispersive Models can be defined through their various model coefficients.

- `PoleResidue()` model
- `Sellmeier()` model
- `Lorentz()` model
- `Debye()` model

### Material Library

Note that there is an extensive library of pre-defined dispersive materials, all implemented using the `PoleResidue()` model using published data.

## Structures

`Structure()` objects simply combine a shape definition through `Geometry()` with a medium definition through `Medium()`.

![Call Structure](../../docs/_static/img/diagram_Structure.png)

## Modes

`Mode()` objects store the parameters that tell the mode solver how to set up the mode profile for the source.
In the current version, they simply store the `mode_index` telling the mode solver to select the mode with the `mode_index`-th smallest effective index.
More development work will be needed here to make the `Mode()` definition more robust.

## Sources

`Sources()` define the electromagnetic current sources in the simulation, which give rise to the field patterns we are interested in.

Sources are geometric objects (have spatial definitions) so they inherit from the `Geometry` class.
However, since we only consider rectangular sources in this version of the code, sources are `Box` subclasses and therefore take `center` and `size` among other arguments.

The `Source()` is the base class for all sources, which contains the `SourceTime()`

From `Source()`, we get either:
- `CurrentSource(polarization)` (defines a `Box` with current oscillating in uniform the direction and component specified in `polarization`).
- `DirectionalSource(direction)` (which defines a plane that emits waves in one direction (+ or -), normal to the plane).

The three types of `DirectionalSources` are:
- `PlaneWave()` (plane wave in homogeneous medium).
- `GaussianBeam(waist_size)` (Gaussian Beam (tbd))
- `ModeSource(mode)` (which defines modal injection with mode solver parameters specified in `mode`.

![Call Structure](../../docs/_static/img/diagram_Source.png)

### Source Time-Dependence

All sources require a `SourceTime()` definition, which specifies the time-dependence.
`SourceTime()` objects implement a `amp_time(self, time)` method, which returns the complex amplitude of the source at a given `time`.

A `Pulse()` is the only subclass of `SourceTime()` currently implemented, but is used to define an object with an envelope "ramping up" in time and containing a central frequency.
In the future, we may want to allow arbitrary `SourceTime()` definitions through either a function or data array.

The only `Pulse` time-dependences (and `SourceTime()` objects more generally) currently defined are `GaussianPulse()` and `CW()`.
However, `CW()` is not currently supported on the backend.

## Monitors

`Monitors` define how the simulation data is stored.
As monitors also have spatial extend, which is restricted to rectangular objects, they are subclasses of `Box` and have a `center` and `size`.

There are three types of usable monitors, each stores different types of data:
- `FieldMonitors()` store the E and H fields within their domain.
- `FluxMonitors()` store the flux through their (planar only) geometry
- `ModeMonitors(mode)` store the mode amplitudes found by decomposition on their (planar only) geometry.

![Call Structure](../../docs/_static/img/diagram_Monitor.png)

### Samplers

All monitors must be told how to sample the data in either time or frequency domain.
For this, the `Sampler()` object is supplied to each `Monitor()`, which can either be:
- `TimeSampler(times)` (defines the time steps to sample the data at)
- `FreqSampler(freqs)` (defines the frequencies to sample the data at using running time discrete Fourier transform).
Note that `ModeMonitors()` can only accept `FreqSamplers()` as the modes are only defined at a specific frequency.
The functions `uniform_times()` and `uniform_freqs()` allow one to easily make evenly spaced samplers.
