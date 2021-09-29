# Tidy3d Notebooks

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-client-revamp/HEAD?filepath=notebooks)

This repository provides a collection of examples and tutorials for the 
Tidy3D solver by Flexcompute Inc. 

Complete documentation can be found [here](http://simulation.cloud/docs/html/index.html).

## Note on imports

When running the notebook locally, it assumes your working directory is `notebooks/`.

Therefore, to import tidy3d (if not pip installed), you must append `..` to path.

````python
import sys
sys.path.append('..')

import tidy3d as td
```


When running `tests/test_notebooks.py`, path='notebooks/' is specified, therefore, the tests also run as if they were from `notebooks/` directory.


