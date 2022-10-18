#!/bin/bash
set -e

black .
python lint.py

pytest -ra tests/test_components/test_apodization.py
pytest -ra tests/test_components/test_base.py
pytest -ra tests/test_components/test_boundaries.py
pytest -ra tests/test_components/test_custom.py
pytest -ra tests/test_components/test_geometry.py
pytest -ra tests/test_components/test_grid.py
pytest -ra tests/test_components/test_grid_spec.py
pytest -ra tests/test_components/test_IO.py
pytest -ra tests/test_components/test_medium.py
pytest -ra tests/test_components/test_meshgenerate.py
pytest -ra tests/test_components/test_mode.py
pytest -ra tests/test_components/test_monitor.py
pytest -ra tests/test_components/test_field_projection.py
pytest -ra tests/test_components/test_sidewall.py
pytest -ra tests/test_components/test_simulation.py
pytest -ra tests/test_components/test_source.py
pytest -ra tests/test_components/test_types.py
pytest -ra tests/test_components/test_viz.py

pytest -ra tests/test_data/test_data_arrays.py
pytest -ra tests/test_data/test_monitor_data.py
pytest -ra tests/test_data/test_sim_data.py

pytest -ra tests/test_package/test_config.py
pytest -ra tests/test_package/test_log.py
pytest -ra tests/test_package/test_main.py
pytest -ra tests/test_package/test_make_script.py
pytest -ra tests/test_package/test_material_library.py

pytest -ra tests/test_plugins/test_adjoint.py
pytest -ra tests/test_plugins/test_component_modeler.py
pytest -ra tests/test_plugins/test_mode_solver.py
pytest -ra tests/test_plugins/test_dispersion_fitter.py
pytest -ra tests/test_plugins/test_resonance_finder.py

pytest -ra tests/test_web/test_auth.py
pytest -ra tests/test_web/test_task.py
pytest -ra tests/test_web/test_webapi.py

pytest --doctest-modules tidy3d/components
