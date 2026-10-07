# KLayout LVS Integration for Tidy3D

This module runs [KLayout](https://www.klayout.de)'s [layout versus schematic (LVS)](https://www.klayout.de/doc/manual/lvs.html) checks from Python and loads their results. KLayout extracts a netlist from the layout: the cells placed, their ports, and the connections between them. It compares that netlist with a schematic netlist, circuit by circuit, and writes the comparison to a results file (`.lvsdb`).

- `LVSRunner.run()` runs a KLayout LVS runset on a GDS or OASIS layout and a schematic netlist.
- `LVSResults.load()` loads a results file without KLayout.
- The results hold a verdict, the pairs of circuits, nets, pins, devices and subcircuits that KLayout compared, both netlists, and KLayout's messages.

## Requirements

1. Install the full KLayout application from https://www.klayout.de/build.html. The plugin looks for the `klayout` executable on the system PATH and in common installation locations. `check_installation()` returns the path it finds, or `None`:

```python
from tidy3d.plugins.klayout import check_installation

print(check_installation())
```

The plugin runs KLayout as a separate process and reads the results file itself, so it doesn't need the `klayout` Python package.

2. Provide an LVS runset, written in [KLayout's LVS language](https://www.klayout.de/doc/about/lvs_ref.html), and a schematic netlist for the runset to read. A foundry's process design kit can include the runset. To write your own, see [Writing an LVS Runset](#writing-an-lvs-runset).

## Running LVS

Create an `LVSRunner` with the runset, and run it on a layout and a schematic. Here, `photonic.lvs` and `monitor.cir` are the runset and schematic of [A Photonic Example](#a-photonic-example):

```python
from tidy3d.plugins.klayout import LVSRunner

runner = LVSRunner(lvs_runset="photonic.lvs")
results = runner.run(
    gdsfile="monitor.gds",
    schematicfile="monitor.cir",
    resultsfile="monitor.lvsdb",
)
print(results.verdict)
```

- **Layout:** a `.gds` or `.oas` file.
- **Runset:** a `.lvs` or `.lylvs` file.
- **Schematic:** any file the runset can read. The runner only checks that it exists.
- **Results file:** `lvs_results.lvsdb` in the working folder by default. A name ending in `.gz` gets a gzip-compressed file. KLayout writes to a temporary file beside the results file, which then replaces it, so a run in which KLayout writes nothing leaves an existing results file untouched.

The runner makes every path absolute before starting KLayout, which would otherwise resolve relative paths against the layout's folder.

`run()` raises an error when:

- KLayout isn't found;
- an input file is missing or has the wrong suffix, the results file's folder doesn't exist, or the results file is one of the input files;
- `runset_variables` or `lvs_args` is invalid, for example when `lvs_args` sets a file variable;
- KLayout writes no results file, in which case the error shows the end of KLayout's output;
- the results file can't be read.

It logs a warning when the runset never mentions one of the file variables, and when KLayout exits with a non-zero code but writes the results file, as some runsets do when the netlists don't match.

`LVSConfig` holds the same settings in one object, and `run_lvs_on_gds()` runs it.

### Runset Variables

The runner passes the file paths to the runset in the variables `$gdsfile`, `$schematicfile` and `$resultsfile`. Some runsets read them from other variables, such as `$input`, `$schematic` and `$report`. Give the runset's names with `runset_variables`:

```python
runner = LVSRunner(
    lvs_runset="pdk.lvs",
    runset_variables={"gdsfile": "input", "schematicfile": "schematic", "resultsfile": "report"},
)
```

The keys are `gdsfile`, `schematicfile` and `resultsfile`. Each value is the name of a runset variable; a leading `$` is ignored.

### Other Runset Arguments

`lvs_args` sets other runset variables, as KLayout's `-rd name=value` options do. Keys and values are converted to strings, and a key can't be one of the file variables:

```python
results = runner.run(
    gdsfile="chip.gds",
    schematicfile="chip.cdl",
    lvs_args={"run_mode": "flat", "topcell": "CHIP"},
)
```

## Analyzing LVS Results

### Verdict

`results.verdict` summarizes the comparison:

| Verdict | When |
|---|---|
| `"mismatch"` | A circuit pair's status isn't `match`, or KLayout recorded an error message. |
| `"match_with_warnings"` | Otherwise, a net or device pair's status is `warning`, or KLayout recorded a warning message. |
| `"match"` | Otherwise. |

KLayout's `compare` succeeds exactly when every circuit pair's status is `match`, so a failed `compare` always gives `"mismatch"`. So does a failed `flag_missing_ports`, a check that runsets call after `compare`: it records an error message for each top-level net whose layout and schematic names differ, and leaves every status unchanged. Other checks that a runset calls after `compare` affect the verdict only if they record messages in the results file. A result that a runset only prints, or reports with its exit code, is in `log` and `exit_code`. Messages from netlist extraction don't affect the verdict.

`"match_with_warnings"` is worth inspecting, although `compare` succeeds. For example, when a component's ports have no names, KLayout can't tell interchangeable ports apart. It pairs them anyway and records warnings, so a connection to the wrong port can pass.

### Summary

`print(results)` shows the verdict, the circuit counts, and each section of mismatches, warnings and messages, with at most 20 entries per section. Here, the photonic example's schematic puts the input on the MMI's port `o2` instead of `o1`:

```text
LVS verdict: mismatch
Circuits: 5 compared: 4 match, 1 nomatch (MONITOR)

Mismatches in MONITOR:
  net $1 <-> IN: mismatch
    layout:    MMI1X2 $1 o2, GC $2 o1
    schematic: EC EC_1 O1, MMI1X2 MMI_1 O2
  net $3 <-> OUT: mismatch
    layout:    MMI1X2 $1 o1, EC $4 o1
    schematic: MMI1X2 MMI_1 O1, GC GC_1 O1
  subcircuit $4 <-> GC_1: mismatch (layout: EC at (0, 0))
  subcircuit $2 <-> EC_1: mismatch (layout: GC at (80, 20), rotated 180°)

KLayout exit code: 0
```

Each net pair lists what the net joins on each side, as the placed circuit, the subcircuit and the pin. Layout subcircuits have no names, and layout nets have names only where labels give them, so the summary shows `$` and an ID for unnamed objects. In the schematic, KLayout writes names in upper case and drops the `X` that starts a subcircuit instance's name: `Xmmi_1` becomes `MMI_1`, and the net `in` becomes `IN`.

### Pairs and Netlists

`results.comparisons` holds one `LVSComparison` per circuit pair. `results.comparison(name)` finds one by either circuit's name, and `results.top` is the comparison of the layout's top cell. A comparison holds net, pin, device and subcircuit pairs. Each pair holds the two objects KLayout paired, `layout` and `schematic`, and either is `None` when KLayout found no partner. Statuses use KLayout's words: `match`, `warning`, `mismatch`, `nomatch` and `skipped`.

`results.mismatches` lists the pairs of every kind whose status is `mismatch`, `nomatch` or `skipped`, and `results.warnings` those whose status is `warning`:

```python
for pair in results.mismatches:
    print(pair.type, pair.circuit, pair.status)
```

`results.layout` and `results.schematic` hold the two netlists, with circuits by name. A circuit holds its nets, pins, devices and placed subcircuits. Each net lists the pins and device terminals it connects to, and each layout subcircuit has a placement:

```python
circuit = results.layout.circuits["MONITOR"]
for subcircuit in circuit.subcircuits:
    print(subcircuit.id, subcircuit.circuit, subcircuit.placement)
```

`results.messages` holds KLayout's comparison messages, and `results.extraction_messages` those from netlist extraction.

### Units and Shapes

Coordinates, shapes and placements are in µm, converted with the layout's database unit, `results.unit`. Device parameters keep the values the runset wrote.

A layout net's `shapes` hold its polygons by runset layer name, in the coordinates of the cell that holds the net. Shapes can make up most of a results file. With `load_shapes=False`, in `run()` or `LVSResults.load()`, the results leave out net shapes, net labels and device terminal shapes.

### KLayout's Output

`results.log` holds everything KLayout printed during the run, standard error included, and `results.exit_code` its exit code. Some runsets report their own verdict there, in the log or with the exit code.

### Loading Results

`LVSResults.load()` reads a results file without running KLayout, and accepts gzip-compressed files. Loaded results have no `log` or `exit_code`.

```python
from tidy3d.plugins.klayout import LVSResults

results = LVSResults.load("monitor.lvsdb")
print(results)
```

### What the Results File Doesn't Record

- **An overall verdict:** the file stores neither the result of `compare` nor a runset's own verdict. `verdict` is computed from the pairs and messages in the file.
- **Earlier comparisons:** for a runset that runs `compare` more than once, the file holds only the last comparison.
- **Circuits that `align` removed:** before comparing, `align` flattens each circuit that has no counterpart on the other side, except top circuits, so these circuits are in neither netlist. An empty component cell whose name matches no schematic subcircuit disappears this way, and the run can pass without comparing it.

## Writing an LVS Runset

An LVS runset is a KLayout script that extracts a netlist from the layout and compares it with the schematic. For this plugin, the runset must:

1. Read the layout from `$gdsfile`, with `source($gdsfile)`. The variable holds an OASIS layout's path too.
2. Read the schematic netlist from `$schematicfile`, with `schematic($schematicfile)`.
3. Request the results file with `report_lvs($resultsfile)`.
4. Run `compare`.

The runner sets the three variables to the file paths. KLayout writes the results file when the runset finishes, and only if the runset called `report_lvs` and ran `compare`. For a runset that uses other variable names, see [Runset Variables](#runset-variables).

### A Photonic Example

In this example, each component is a cell of the layout and a subcircuit of the schematic. A component cell has an outline on layer 100/0 and, at each port, a box on layer 101/0 named by a text label on layer 101/1. Waveguides on layer 1/0 connect the components by touching their port boxes.

```ruby
deep
source($gdsfile)
schematic($schematicfile)
report_lvs($resultsfile)

wg        = input(1, 0)     # waveguides
outline   = input(100, 0)   # component outlines
port      = input(101, 0)   # port boxes
port_name = labels(101, 1)  # port names

routing = wg - outline
connect(routing, port)
connect(port, port_name)

compare
```

`deep` keeps the layout's hierarchy, so each component cell becomes a circuit and each placed component a subcircuit. Subtracting the outlines leaves only the waveguides between components, so a component's own waveguide doesn't join all of its ports into one net. The port names become the component's pin names, which lets KLayout tell the ports apart.

The schematic is a SPICE netlist. Each component is a subcircuit whose pins are its port names, and the top circuit places the components and names their connections:

```text
.SUBCKT EC o1
.ENDS EC

.SUBCKT MMI1X2 o1 o2 o3
.ENDS MMI1X2

.SUBCKT GC o1
.ENDS GC

.SUBCKT PD o1
.ENDS PD

* An edge coupler feeds a 1x2 MMI, whose outputs go to a grating coupler and a photodetector.
.SUBCKT MONITOR
Xec_1  in          EC
Xmmi_1 in out tap  MMI1X2
Xgc_1  out         GC
Xpd_1  tap         PD
.ENDS MONITOR
```

KLayout pairs circuits by name, so each component's cell must have its subcircuit's name. A runset can pair circuits with different names with `same_circuits`.

A runset can also be a `.lylvs` file, KLayout's XML macro format. It must mark its text as an LVS script with the tags `<interpreter>dsl</interpreter>` and `<dsl-interpreter-name>lvs-dsl-xml</dsl-interpreter-name>`. Without them, KLayout runs the text as plain Ruby, which fails.
