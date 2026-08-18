# Dipole Emission Plugin

The dipole emission plugin evaluates angular radiation intensity from classical
electric dipoles embedded in a passive optical structure.

Use `DipoleEmissionStudy` for the public workflow. A study takes a source-free
base simulation, an emission analysis region, sampled dipole positions,
far-field angles, and optional position weights. It returns compact data summed
over all sampled positions by default, with optional position-resolved radiation
intensity at selected indexes.

The returned radiation intensity is normalized per squared electric dipole
moment. If a physical dipole moment `d` is expressed in `C*um`, multiplying by
`|d|^2` gives angular power density in `W/sr`.

Use `angular_radiation_transfer(bulk_refractive_index)` to divide the angular
radiation intensity by the total power of the same dipole in a homogeneous
reference medium. The result has units `1/sr`. The supplied index belongs to the
emitter reference medium; the collection-medium index is detected automatically
from the TFSF injection face. The legacy `radiation_intensity_transfer(...)`
method is a deprecated alias.

See `docs/api/plugins/dipole_emission.rst` for the public API reference.
