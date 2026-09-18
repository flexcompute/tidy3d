# KLayout DRC Integration for Tidy3D

This module provides integration between [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) and [KLayout](https://www.klayout.de)'s [Design Rule Check (DRC) engine](https://www.klayout.de/doc/manual/drc.html), allowing you to perform design rule checks on GDS files and Tidy3D objects.

## Quickstart

For a full quickstart example, please see [this quickstart notebook](https://github.com/flexcompute/tidy3d-notebooks/blob/develop/KLayoutPlugin_DRCQuickstart.ipynb).

## Features

- Run DRC on GDS files or Tidy3D objects ([Geometry](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Geometry.html), [Structure](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Structure.html#tidy3d.Structure), or [Simulation](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.Simulation.html#tidy3d.Simulation)) with `DRCRunner.run()`.
- Load DRC results into a `DRCResults` data structure with `DRCResults.load()`.
- Limit how many violation markers are loaded by passing `max_results` to `DRCRunner.run()`,
  `run_drc_on_gds()`, or `DRCResults.load()`.
- Evaluate one global or per-parameter optimizer candidate batch with one KLayout process using
  `BatchedDRCChecker`.

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

On macOS, Homebrew cask app-suite installs such as
`/Applications/KLayout/klayout.app/Contents/MacOS/klayout` are also discovered automatically.

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

Use `max_results_per_cell` instead when every violating cell must remain represented. For example,
`max_results_per_cell=1` loads one marker from each violating cell even when early cells contain
many violations. `max_results` and `max_results_per_cell` are mutually exclusive.

### Faster DRC Checks for Multiple Designs

Starting a separate KLayout process for every design can be slow. `BatchedDRCChecker` checks
multiple designs in one KLayout invocation and returns one validity boolean for each design. The
designs may be controlled by different parameter values, as in an optimization line search, but
the checker can also be used by any workflow that can export each design to GDS.

Internally, the checker places each flattened layout in a uniquely named GDS subcell. It creates a
temporary copy of the `.drc` or `.lydrc` runset beside the original containing the `deep` directive,
leaving the user's runset unchanged while preserving relative rule-file references. A runset that
already enables `deep` is used directly. Otherwise, its containing directory must be writable for
the duration of the check. Hierarchical result cell names are then mapped back to the corresponding
designs:

```python
from tidy3d.plugins.klayout import BatchedDRCChecker

def export_design(design_parameters, gds_path):
    make_geometry(design_parameters).to_gds_file(
        fname=gds_path,
        z=0,
        gds_layer=1,
        gds_dtype=0,
    )

drc_checker = BatchedDRCChecker(
    export_design,
    "foundry_rules.drc",
    max_results_per_cell=1,
)

valid = drc_checker([design_a_parameters, design_b_parameters])
# valid contains one boolean for each input design.
```

The same checker can be passed to `BacktrackingSafeUpdate` when the designs are candidate updates
from an optimization. It implements the autograd plugin's `ConstraintChecker` interface, so each
global or recovery candidate batch is delegated to it directly.

The checker arranges candidates in a near-square grid to limit the stitched layout's coordinate
range. It automatically separates rows and columns by
`max(min(candidate_width, candidate_height))` across the batch. For standard distance-based checks,
the checker detects an edge-pair whose edges belong to different candidate cells and raises instead
of reporting either candidate as invalid. It also raises if KLayout reports a marker in `TOP` or
another unknown cell. If the automatic separation is insufficient for a runset, use scalar checks
instead. Because arbitrary runsets can emit custom marker geometry, this cannot diagnose every
possible cross-candidate interaction. Per-cell result truncation can also omit a later
cross-candidate marker after an earlier marker has been retained for that cell.

Set `debug_dir` to audit failed checks. Whenever at least one candidate is invalid or checking
raises, the checker copies the individual candidate GDS files, stitched layout, raw KLayout
`.lyrdb`, and effective runset into a unique child of that directory. The raw report is not altered
by `max_results_per_cell`, so it can be inspected for a suspected cross-candidate marker. The path
is logged and is also available as `drc_checker.last_debug_dir`. Clean batches do not leave debug
artifacts. Failed batches can produce large files, so enable this option only while diagnosing a
problem and remove artifacts that are no longer needed.

This implementation uses GDS files and the KLayout command line interface; it does not require or
enable the `klayout.db` Python module.

Spatial batching is intended for local, translation-invariant rules with a finite interaction
distance. Whole-layout rules such as global density, connectivity, or aggregate geometry checks can
couple otherwise separated candidates and should not use `BatchedDRCChecker`. Such rules can still
be used with `BacktrackingSafeUpdate`: provide a scalar checker that runs KLayout separately for one
parameterization and returns `results.is_clean`. Scalar checking evaluates candidates lazily and
one at a time.
