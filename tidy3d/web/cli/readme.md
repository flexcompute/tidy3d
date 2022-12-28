## API key

### Generating an API key

You can find your API key in the web http://tidy3d.simulation.cloud


### Setup API key

#### Environment Variable
``export SIMCLOUD_API_KEY="your_api_key"``

#### Command line
``tidy3d configure``, then enter your API key

#### Manually
``echo 'apikey = "your_api_key"' > ~/.tidy3d/config``

## Publishing Package

First, configure poetry to work with test.PyPI. Give it a name of `test-pypi`.

``poetry config repositories.test-pypi https://test.pypi.org/legacy/``

``poetry config pypi-token.test-pypi <<test.pypi TOKEN>>``

Then, build and upload, make sure to specify repository `-r` of `test-pypi`.

``poetry publish --build -r test-pypi``

The changes should be reflected on test PyPI https://test.pypi.org/project/tidy3d-beta/1.8.0/

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

Work in progress.

