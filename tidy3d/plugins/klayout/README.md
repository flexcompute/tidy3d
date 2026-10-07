# KLayout Integration for Tidy3D

This module provides integration between [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) and [KLayout](https://www.klayout.de).

## Design Rule Checking (DRC)

The DRC submodule provides integration between [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/) and [KLayout](https://www.klayout.de)'s [Design Rule Check (DRC) engine](https://www.klayout.de/doc/manual/drc.html), allowing you to perform design rule checks on GDS files and Tidy3D objects.

For more details, please see the [DRC README](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/plugins/klayout/drc/README.md).

## Layout Versus Schematic (LVS)

The LVS submodule runs [KLayout](https://www.klayout.de)'s [layout versus schematic (LVS) checks](https://www.klayout.de/doc/manual/lvs.html), which extract a netlist from a layout and compare it with a schematic netlist, and loads their results.

For more details, please see the [LVS README](https://github.com/flexcompute/tidy3d/blob/develop/tidy3d/plugins/klayout/lvs/README.md).
