# Electromagnetic media

This package contains the electromagnetic material models exposed through `tidy3d` and the eager `tidy3d.components.medium` compatibility facade. Thermal, charge, and combined multiphysics material helpers live in the sibling [`material`](../material/README.md) package.

The modules are organized by model family:

- `base.py` and `abstract_custom.py` define the shared uniform and spatially varying abstractions.
- `isotropic.py`, `anisotropic.py`, and `two_d.py` contain nondispersive and tensor media.
- `pole_residue.py`, `sellmeier.py`, `lorentz.py`, `drude.py`, and `debye.py` contain dispersive models.
- `custom/` mirrors those families for spatially varying isotropic, dispersive, and anisotropic media.
- `lossy_metal.py` and `roughness.py` contain surface-impedance conductor models.
- `perturbation.py` contains temperature and carrier-density perturbation models.
- `medium_types.py` assembles public type aliases after all model families are defined.
- `_rebuild.py` resolves Pydantic forward references during package initialization.

Keep implementation imports acyclic. Public and historical component-level names should be re-exported eagerly from `__init__.py`; top-level user imports such as `tidy3d.CustomMedium` remain the canonical API.
