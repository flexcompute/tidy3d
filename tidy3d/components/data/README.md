# Tidy3d Data

This document will overview how datastructures are organized in the data refactor set to launch in 2.0.

## Files

The data organization lives in `tidy3d/components/data` directory.

The directory contains three files:

- `data_array.py` defines the most atomic datastructures, which are subclasses of `xarray.DataArray`.
- `monitor_data.py` defines the datastructures associated with each monitor type. They are regular tidy3d components that contain `DataArray` fields, among others.
- `sim_data.py` holds the `SimulationData`, which basically holds a dictionary of `MonitorData` objects for the monitors in the simulation.

## Structure

### ``DataArray`` Objects

The most atomic datastructure is the `DataArray`.

A `DataArray` represents a dataset with a multidimensional array and labelled coordinates. For example, a scalar field, flux over time, etc.

For simplicity, `DataArray` inherits from `xarray.DataArray` and therefore supports all of the selection, indexing, and other features of a `xarray.DataArray`.

The varios `DataArray` subclasses are templates, which define the dimensions and attributes of the data expected.  For example:

```python
class ScalarFieldTimeDataArray(DataArray):
    """Spatial distribution in the time-domain."""
    __slots__ = ("x", "y", "z", "t")
    _data_attrs = {"long_name": "field value"}
```

Defines a scalar field in the time-domain, which must have coordinates with keys `"x"`, `"y"`, `"z"`, and `"t"` and we've given the values a long name for plotting purposes.  `__slots__` is just a way to hardcode the dims in `xarray`, so we use it instead of `dims`.

We construct a `DataArray` by supplying the raw values (multi-dimensional array) as the first `*arg` and then coords as a dict `**kwarg`, ie. `flux = FluxDataArray(values, coords={'f': 4e14})`. This construction something that needs to change often in the backend.

The naming convention for `DataArray` objects is to append `DataArray` to their name, ie. "`FluxDataArray`.

TLDR: We use these `DataArray` instances primarily as a template for how to define the various xarray data that gets put in the monitor data described below.

### ``MonitorData`` objects

The ``MonitorData`` objects store the data for a single type of monitor. There is a one-to-one correspondence with each ``Monitor`` type.  

Every ``MonitorData`` contains a field that holds the ``monitor`` it corresponds to. In some cases, we want to create some data without an actual monitor (for example a ``FieldData`` representing the surface currents in Near2Far). To handle this, we create an equivalent monitor using the relevant context at the time of creation. This makes things easier in many places.

``MonitorData`` objects will contain one or more ``Field``s holding data in the form of ``DataArray`` objects.  For example:

```python
class FieldTimeData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields."""
    monitor: FieldTimeMonitor
    Ex: ScalarFieldTimeDataArray = None
    Ey: ScalarFieldTimeDataArray = None
    Ez: ScalarFieldTimeDataArray = None
    Hx: ScalarFieldTimeDataArray = None
    Hy: ScalarFieldTimeDataArray = None
    Hz: ScalarFieldTimeDataArray = None
```

The naming convention for `MonitorData` objects is to replace `Monitor` with `Data` in the monitor name, eg. "`FluxTimeMonitor`" becomes "`FluxTimeData`".

Final note: data for a `FluxMonitor` and `FluxTimeMonitor` are loaded in `FluxData` and `FluxTimeData` instances, respectively. However, these classes contain a single `.flux` field that is a `FluxDataArray` and `FluxTimeDataArray`, respectively.  So to access the raw flux data for a monitor, one must do `flux_data.flux` instead of the data being stored directly.  This is a change that comes up a lot in the backend.

#### Normalization

All `MonitorData` subclasses have a `.normalize()` method, which returns a copy of the instance normalized by a given source spectrum.

```python
def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> MonitorData:
```

Rather than raw data being passed to this, `source_spectrum_fn` is a function of frequency that returns the complex-valued source spectrum.  This was done to simplify things at the `SimulationData` level and provide more customizability.


#### Symmetry

All `MonitorData` subclasses also have an `.apply_symmetry()` method, which returns a copy of the instance with symmetry applied. There is therefore no notion of "state" with regard to the symmetry of a monitor data.

```python
def apply_symmetry(
    self,
    symmetry: Tuple[Symmetry, Symmetry, Symmetry],
    symmetry_center: Coordinate,
    grid_expanded: Grid,
) -> MonitorData:
```

#### Field-Like Data

The `MonitorData` subclass `AbstractFieldData` defines a few methods and properties needed to properly handle field-like data, such as `FieldData`, `FieldTimeData`, `PermittivityData`, and `ModeSolverData`.

There are a few convenient properties defined for each `AbstractFieldData`:

- `.field_components` is a dict mapping of the field name (str, eg. 'Ey') to the scalar field `DataArray`. It is very useful for iterating through the components and selecting by string.
- `.grid_locations` is a dict mapping of the field name to the "grid_key" used to select the position in the yee lattice. For example, for a `PermittivityMonitor` called `p`, we would have `p.grid_locations['eps_yy'] == 'Ey'`.
- `.symmetry_eigenvalues` returns a dict mapping of the field name to a function of axis (0,1,2) that returns the eignenvalue of that field component under symmetry transformation along this axis.

Field-like data also support `def colocate(x=None, y=None, z=None) -> xr.Dataset`, which returns an `xarray.Dataset` of all the field components colocated at the supplied x,y,z coordinates. If any of the coordinates are `None`, nothing is done to colocate along that coordinate.

#### Data Type Map

The ``DATA_TYPE_MAP`` is defined in `monitor_data.py` as a dictionary mapping the various `Monitor` types to the corresponding `MonitorData` types. This is used in the backend to select the right data type to load for a given monitor.

### `SimulationData` objects

Like before, the `SimulationData` object contains all of the data for a given `Simulation`. The `Simulation` is still stored directly as a `pd.Field` and the `monitor_data` is still a dictionary mapping the names of the `Monitor` objects to the corresponding `MonitorData` objects.

#### Normalizing

The frequency-dependent data in `SimulationData` objects is normalized to the source given by `SimulationData.Simulation.normalize_index`, such that it matches exactly what is written to file (as opposed to the old workflow of the file containing un-normalized data). A copy of the `SimulationData` with a different normalization can be obtained using the

Normalization is achieved through the 
```python
def renormalize(self, normalize_index: int) -> SimulationData:`
```
method.

#### Applying Symmetry

Again, `SimulationData` objects have no notion of state regarding the symmetry. But they do contain a method
```python
def apply_symmetry(self, monitor_data: MonitorDataType) -> MonitorDataType:
```
which computes the expanded grid for a monitor and then returns a symmetry-applied version of the monitor data.

#### Selecting Monitor Data

Selection with square brackets (`sim_data[monitor_name]`) returns a copy of that monitor data with symmetry and normalization applied using the functions described above. Accessing the monitor_data through `SimulationData.monitor_data` dictionary gives direct access if desired (use with caution).

#### Getting Fields
There are a few other convenience methods for dealing with ``AbstractFieldData`` objects stored in the `SimulationData.monitor_data` dict.
`sim_data.at_centers(monitor_name)` gets the field-like data at the yee cell centers.
`sim_data.get_intensity(monitor_name)` gets the intensity data for a field-like data evaluated at the yee cell centers.
`sim_data.get_poynting_vector(monitor_name)` gets the Poynting vector data at the yee cell centers.

#### Plotting

Plotting is very similar to before, except now instead of `freq`, `time`, `mode_index` and the spatial coordinates being explicitly defined as their own kwargs, the `plot_field` method accepts `**sel_kwargs`.

`**sel_kwargs` are any extra kwargs that can be applied through `.interp()` to the field-like data to get it into the proper form for plotting, namely a state where there are only two spatial coordinates with more than one value.

For example, if my data contains `x,y,z,f,mode_index` data, I might need to supply `z=0, f=4e14, mode_index=2` to the `plot_field()` as `**sel_kwargs` to get it into the form needed to plot field(x,y) on the z=0 plane. If there was only one frequency in the data, i could safely leave it out of the `**sel_kwargs`. The code will automatically detect the proper axis position for plotting.

## File IO

### JSON

`SimulationData` and `MontorData` objects inherit directly from `Tidy3dBaseModel` and can be written directly to json. `DataArray` objects can also be written to json as we define a json encoder that uses [`xarray.DataArray.to_dict()`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.to_dict.html`). Therefore, an entire `SimulationData`, `MonitorData`, or `DataArray` object can be written to json without issue.

### HDF5

We may also write any tidy3d component to hdf5. The logic for this is defined in `Tidy3dBaseModel` and involves a recursive writing and reading of files to hdf5 groups.

To write object `obj` to hdf5 format, simply use `obj.to_file(path)` where the path includes `.hdf5` as the extension.


