# KLayout DRC Integration for Tidy3D

This module provides integration between [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) and [KLayout](https://www.klayout.de)'s [Design Rule Check (DRC) engine](https://www.klayout.de/doc/manual/drc.html), allowing you to perform design rule checks on GDS files and Tidy3D objects.

## Quickstart

For a full quickstart example, please see [this quickstart notebook](https://github.com/flexcompute/tidy3d-notebooks/blob/develop/KLayoutPlugin_DRCQuickstart.ipynb).

## Features

- Run DRC on GDS files or Tidy3D objects ([Geometry](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Geometry.html), [Structure](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Structure.html#tidy3d.Structure), or [Simulation](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Simulation.html#tidy3d.Simulation)) with `DRCRunner.run()`.
- Load DRC results into a `DRCResults` data structure with `DRCResults.load()`.
- Limit how many violation markers are loaded by passing `max_results` to `DRCRunner.run()`,
  `run_drc_on_gds()`, or `DRCResults.load()`.

## Prerequisites

1. Have the full KLayout application installed and added to your system PATH.

To install KLayout, please refer to https://www.klayout.de/build.html. 

This module will attempt to locate the klayout executable in the typical installation locations after KLayout has been installed on your system.

To check if KLayout is found by this module, you can use the provided `check_installation()` utility:
```python
from tidy3d.plugins.klayout import check_installation
# Prints the full path to the executable if found, otherwise returns None
print(check_installation())
```
The full path to the application should be displayed if KLayout has been added to the system PATH.

If the installation could not be found, the application may need to be manually added to the system PATH. The method to add KLayout to your system PATH depends on your operating system. For example, on MacOS, you can add the following line to your `~/.zshrc` file. This will permanently add KLayout to your PATH:
```zsh
export PATH="$PATH:/Applications/klayout.app/Contents/MacOS"
```
Run `check_installation()` again to check if the application is found.

2. Provide a KLayout DRC runset script that defines the source (input gds) using `source($gdsfile)` and the report (output result file) using `report("DRC results", $resultsfile)`. Please refer to the [DRC Runset Formatting section below](#drc-runset-file-formatting).

## DRC Runset File Formatting

DRC runsets are defined in KLayout DRC's domain-specific-language. Please refer to the [KLayout User Manual](https://www.klayout.de/doc/manual/drc.html) for details regarding syntax and functionality.

**Importantly**, for compatibility with this plugin, the runset must include the following:

1. The source must be defined as: `source($gdsfile)`
2. The report (output result file) must be defined as: `report("DRC results", $resultsfile)`

This is to ensure that the GDS file created from a Tidy3D object is properly loaded into KLayout, and the results are saved to a file of the user's choosing.

Here is an example DRC runset script that checks for minimum width, space, area, and hole on layer (0,0):

```
# Simple DRC rules for testing

# Define the source and output (report)
source($gdsfile)
report("DRC results", $resultsfile)

# Checks minimum width of 300 nm
input(0, 0).width(300.nm).output("min_width", "minimum width")

# Checks minimum gap of 300 nm
input(0, 0).space(300.nm).output("min_gap", "minimum gap")

# Checks minimum area of 1e5 nm^2
input(0, 0).drc(area < 100000).output("min_area", "minimum area")

# Checks minimum hole of 1e5 nm^2
input(0, 0).holes.drc(area < 100000).output("min_hole", "minimum hole")
```

### Running DRC

To run DRC, create an instance of `DRCRunner` and use `DRCRunner.run()`
You can run DRC on a GDS file as follows:

```python
from tidy3d.plugins.klayout.drc import DRCRunner

# Run DRC on a Tidy3D object
runner = DRCRunner(
    drc_runset="example_runset.drc",
    verbose=True,
)
results = runner.run("geom.gds")
```

Or you can run DRC on a Tidy3D [Geometry](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Geometry.html), [Structure](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Structure.html#tidy3d.Structure), or [Simulation](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Simulation.html#tidy3d.Simulation) object:

```python
# Create a simple polygon geometry
vertices = [(-2, 0), (-1, 1), (0, 0.5), (1, 1), (2, 0), (0, -1)]
geom = td.PolySlab(vertices=vertices, slab_bounds=(0, 0.22), axis=2)

# Run DRC and get results
runner = DRCRunner(
    drc_runset="example_runset.drc",
    verbose=True,
)
results = runner.run(geom, z=0.1, gds_layer=0, gds_dtype=0)
```

In the case of running DRC on a Tidy3D object, the object will first be saved to a GDS file, with additional keyword args passed into the object's `to_gds_file()` method (eg. [Simulation.to_gds_file()](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Simulation.html#tidy3d.Simulation.to_gds_file)).

### Analyzing DRC Results

The output of `DRCRunner.run_drc()` is a `DRCResults` object that stores the results of the DRC run.

The user can check whether DRC passed with `DRCResults.is_clean`:

```python
print(results.is_clean)
```

A summary of DRC violations can be displayed by printing `DRCResults`:

```python
print(results)
```

Individual violation categories can be indexed by key:

```python
# This will show how many violation shapes were found for the 'min_width' rule.
print(results['min_width'].count)

# This will show all of the violation marker shapes for the 'min_width' rule
print(results['min_width'].markers)
```

Results can also be loaded from a KLayout DRC database file with `DRCResults.load(resultsfile)`:

```python
from tidy3d.plugins.klayout.drc import DRCResults

print(DRCResults.load("drc_results.lyrdb"))
```

### Limiting Loaded Results

Large designs can generate an enormous number of violations. Pass the optional `max_results`
argument to `DRCRunner.run()`, `run_drc_on_gds()`, or `DRCResults.load()` to retain only the first
`N` markers across all categories. When the option is not set, a warning is emitted if more than
100,000 markers are present so you can set an appropriate limit. When the option is set and more
total violations are present than the limit allows, a warning indicates that the results were
truncated before parsing individual markers.
