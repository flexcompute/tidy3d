#!/bin/bash
set -e

black .
python lint.py

pytest -ra tests/test_data/test_data_arrays.py
pytest -ra tests/test_data/test_dataset.py
pytest -ra tests/test_data/test_monitor_data.py
pytest -ra tests/test_data/test_sim_data.py

pytest -ra tests/test_components/test_boundaries.py
pytest -ra tests/test_components/test_components.py
pytest -ra tests/test_components/test_construct.py
pytest -ra tests/test_components/test_geo_group.py
pytest -ra tests/test_components/test_grid.py
pytest -ra tests/test_components/test_IO.py
pytest -ra tests/test_components/test_meshgenerate.py
pytest -ra tests/test_components/test_near2far.py
pytest -ra tests/test_components/test_sidewall.py

pytest -ra tests/test_package/test_config.py
pytest -ra tests/test_package/test_log.py
pytest -ra tests/test_package/test_main.py
pytest -ra tests/test_package/test_make_script.py
pytest -ra tests/test_package/test_material_library.py

pytest -ra tests/test_plugins/test_component_modeler.py
pytest -ra tests/test_plugins/test_plugins.py

pytest --doctest-modules tidy3d/components
