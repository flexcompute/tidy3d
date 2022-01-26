# Tidy3D (Beta release)

![tests](https://github.com/flexcompute/Tidy3D-client-revamp/actions/workflows//run_tests.yml/badge.svg)

<img src="img/Tidy3D-logo.svg">

<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flexcompute/Tidy3D-client-revamp/HEAD?filepath=notebooks) -->

Beta release of the FDTD solver by Flexcompute.

<img src="img/snippet.png">

## Installation

For now:

```
git clone https://github.com/flexcompute/tidy3d.git
cd tidy3d
pip install -e .
```

Can verify it worked by running

```
python -c "import tidy3d as td; print(td.__version__)"
```

and it should print out 

```
0.2.0
```

After we put on pyPI, it will be installable via

```
pip install tidy3d-beta
```

but the package will still be imported in python as 

```python
import tidy3d as td
```
