## API key

### Generating an API key

You can find your API key in the web http://tidy3d.simulation.cloud


### Setup API key

#### Environment Variable
``export SIMCLOUD_API_KEY="your_api_key"``

#### Command line
``tidy3d configure``, then enter your API key when prompted

#### Manually
``echo 'apikey = "your_api_key"' > ~/.tidy3d/config``

## Publishing Package

First, configure poetry to work with test.PyPI. Give it a name of `test-pypi`.

``poetry config repositories.test-pypi https://test.pypi.org/legacy/``

``poetry config pypi-token.test-pypi <<test.pypi TOKEN>>``

Then, build and upload, make sure to specify repository `-r` of `test-pypi`.

``poetry publish --build -r test-pypi``

The changes should be reflected on test PyPI https://test.pypi.org/project/tidy3d-beta/1.8.0/

### Fixing pyproject.toml

Note, this did not work originally because while the package directory name is `tidy3d/`, the repository name on PyPI is `tidy3d-beta`.

So a couple changes were required in `pyproject.toml`.

I needed to change the `name=tidy3d` to `name=tidy3d-beta`
I needed to add what packages to include, namely
```
packages = [
    { include = "tidy3d" },
    { include = "tidy3d/web" },
    { include = "tidy3d/plugins" },
]
```
(note, not 100% sure I needed to include `web` and `plugins`, but I think they are needed because they aren't imported in the top level `tidy3d/__init__.py` file.)

Once I did this, the steps from the previous section worked properly.

### Testing installation from test.PyPI

To test, in a clean environment

``python3.9 -m pip install --index-url https://test.pypi.org/simple/ tidy3d-beta``

note: I was getting errors doing this, because it was trying to install all previously uploaded versions of `tidy3d-beta`. So when I did

``python3.9 -m pip install --index-url https://test.pypi.org/simple/ tidy3d-beta==1.8.0``

It started working, however I got another error

```
Collecting tidy3d-beta==1.8.0
  Using cached https://test-files.pythonhosted.org/packages/5d/67/0cd75f00bb851289c79b584600b17daa7e5d077d2afa7ab8bfccc0331b3b/tidy3d_beta-1.8.0-py3-none-any.whl (257 kB)
ERROR: Could not find a version that satisfies the requirement pyroots<0.6.0,>=0.5.0 (from tidy3d-beta) (from versions: none)
ERROR: No matching distribution found for pyroots<0.6.0,>=0.5.0
```

It turns out this is expected using test pyPI as explained [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/#installing-your-newly-uploaded-package).

When I try to install `tidy3d-beta` from test.pyPI in an environment with the dependencies already installed from 
``python3.9 -m pip install -e '.[dev]'``
``python3.9 -m pip uninstall tidy3d-beta``

It works as expected, besides needing `click`

``python3.9 -c "import tidy3d as td; import tidy3d.web as web; from tidy3d.plugins.mode import ModeSolver"``

```
[11:04:02] INFO     Using client version: 1.8.0                                                                                                                                              __init__.py:112
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.9/site-packages/tidy3d/web/__init__.py", line 8, in <module>
    from .cli import tidy3d_cli
  File "/usr/local/lib/python3.9/site-packages/tidy3d/web/cli/__init__.py", line 4, in <module>
    from .app import tidy3d_cli
  File "/usr/local/lib/python3.9/site-packages/tidy3d/web/cli/app.py", line 7, in <module>
    import click
ModuleNotFoundError: No module named 'click'
```

### Fixing the error

So I added `click` with 

``poetry add click``

This changed the `poetry.lock` and `pyproject.toml`.

I then bumped the version in `version.py` and `pyproject.toml` to `1.8.1` (otherwise, could not upload again to PyPI for same version) and repeated the publishing steps from above again.

Testing the newly pip-installed version I was able to successfully import tidy3d and run a simulation!



