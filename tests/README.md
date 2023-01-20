# Tests

## Brief pytest explanation

Disclaimer: this is as far as I understand, see [documentation](https://docs.pytest.org/en/6.2.x/) for more details and complex cases.

Running

```pytest dir```

Will run tests contained in any files called `dir/test*.py`.
For example, if `dir` contains `test_code.py` and `utils.py` it will only run tests in `test_code.py`.

Inside `dir/test_code.py` pytest will run any code in the main file and then call any function beggining with `test`. For example, if the contents of `test_code.py` are as follows:

```python

# gets evaluated first
my_two = 2.0
my_three = add_one(my_two)

def add_one(x):
    # calls this function
    return x+1

def test_one_plus_three():
    # pytests runs this "test", passes test if no Exceptions
    assert 1.0 + my_three == 4.0, "addition failed" 

def something_else():
    # doesnt get called ever
    pass

```

See `tests/` directory for some examples.

## Writing Tests

Ideally, the tests should be as granular as possible and test specific operations or functions.  General tests are useful to include as well, but granular / unit tests can isolate the specific bugs more quickly.

When you are developing, please err on the side of adding lots of tests, even if they seem trivial, they can help catch subtle bugs.

If you think your tests fall into a new category, just add a new file in `tests/`

To reduce redundancy, functions and objects used a lot can be defined in `tests/utils.py` and used in all tests.  There are some directory clearing functions and initialized simulation objects in there, which are useful.

If you can, err on the side of tests that run quickly.  For example, if testing some loading or exporting feature, rather than specifying 10001 frequencies in the `Simulation` monitors, only provide 2 or 3 just to speed things up.  Use good judgement.

## How to:

### Run all tests

All tests must be run from the root directory.
Ie: `tidy3d/`

From there, to run all tests:

```pytest -rA tests```

You should probably do this before committing changes as this is what the automated tester will likely use.

`-rA` are just options that nicely format the test output, see more by running `pytest -h`.

### Test specific things

To run tests in a specific test file:

```pytest -rA tests/test_specific.py```

To run a specific test in a specific test file:

```pytest -rA tests/test_specific.py -k my_test```

Pytest does partial matching of argument of -k against the test function name.  For example,

```python
def test_my_test():
    # tested
    pass

def test_xmy_testx():
    # tested
    pass

def test_test_other():
    # not tested
    pass
```

## Test coverage

The test coverage tells you what percentage of lines in the source code were traversed over the course of the tests. The test coverage is at 92% as of Jan 20, 2023. We want to make sure any new code is very well tested, ideally as close to 100% as is reasonable. 

To measure the test coverage:

```
pip install pytest-cov
```
if not already installed

To run coverage tests with results printed to STDOUT:
```
pytest tests --cov-report term-missing --cov=tidy3d
```
To run coverage tests and get output as .html (more intuitive):
```
pytest tests --cov-report=html --cov=tidy3d
open htmlcov/index.html
```