# Tests

All tests must be run from the root directory.
Ie: `tidy3d-client-revamp/`

From there, to run all tests:

```pytest -rA tests```

To run tests in a specific test file:

```pytest -rA tests/test_specific.py```

To run a specific test in a specific test file:

```pytest -rA tests/test_specific.py -k specific_test```
