# Monitor data

Monitor data models are grouped by responsibility:

- `base.py`: shared monitor and electromagnetic-field behavior.
- `field/`: field, surface, point-cloud, time-domain, and material data.
  `ElectromagneticFieldData` binds its grid, metrics, algebra, and I/O
  methods from focused implementation modules.
- `mode/`: overlap, mode, and mode-solver data.
- `flux.py`: flux and time-domain modal data.
- `projection/`: field-projection, diffraction, and directivity data.

The package initializer is the compatibility facade for the former monolithic
`monitor_data` module. It preserves its package-level model, alias, and
data-array imports. The focused subpackages expose their canonical model
modules; use those modules for new imports.
