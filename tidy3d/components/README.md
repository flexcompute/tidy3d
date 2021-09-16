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
Indices = List[pydantic.nonNegativeInt]
```
defines a list of non-negative integers, for example, a list of indices.

This file provides a way for one to import the same type into multiple components.
Often times, though, if a type is being used once, it is just declared in the component definition itself.

### Validators

`validators.py` defines `pydantic` validators that are used in several of the components.
These validators let one define functions that check whether input arguments or combinations of input arguments meet certain acceptance criteria.
For example, there is a validator that checks whether a given `Box` class or subclass is planar (ie whether it's `size` attribute has exactly one `0.0`).

### Constants

Several physical constants are defined in `constants.py` as well as the default value for various parameters, like `td.inf`.

## Component Structure

### Geometry

The `Geometry` component is used to define the layout of objects with a spatial component.

Each `Geometry` subclass implements a `._get_bounds(self)` method, which returns the min and max coordinates of a bounding box around the structure.

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
- `grid_size` (defines the discretization).
- `medium` (the background medium).
- `run_time`
- `pml_layers` (a list of three `PMLLayer(profile, num_layers)` objects specifying the PML, defined in `pml.py`).
- `symmetry`
- `courant`
- `shutoff`
- `subpixel`

Crucially, the `Simulation` also stores three dictionaries containing important `tidy3d` components.
The keys of these dictionaries are the names of the components and the values are instances of the components.

- `structures`, a list of `Structure()` objects, defining the various objects in the simulation domain.
- `sources`, a dictionary of `Source()` objects, defining the current sources in the simulation domain.
- `monitors`, a dictionary of `Monitor()` objects, defining what data is being measured and where.

#### Validations

Upon intialization, the simulation checks whether any of the objects are completely outside of the simulation bounding box, at which point it will error.
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
- `nk_to_eps_sigma` (convert refractive index parameters (n, k) to permittivity and conductivity).
- `nk_to_medium` (convert refractive index parameters (n, k) to a `Medium()` directly).
- `nk_to_eps_complex` (convert refractive index parameters (n, k) to a complex-valued permittivity).
- `eps_sigma_to_eps_complex` (convert permittivity and conductivity to complex-valued permittiviy)

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

## Modes

`Mode()` objects store the parameters that tell the mode solver how to set up the mode profile for the source.
In the current version, they simply store the `mode_index` telling the mode solver to select the mode with the `mode_index`-th smallest effective index.
More development work will be needed here to make the `Mode()` definition more robust.

## Sources

## Monitors

