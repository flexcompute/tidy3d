"""
Thank you, ChatGPT, for the core of the following test code.
"""

import pytest
from tidy3d.packaging import check_import, verify_packages_import, Tidy3dImportError

assert check_import("tidy3d") == True


# Mock module import function to simulate availability
def mock_check_import(module_name):
    if module_name == "tidy3d":
        return True
    return False


def test_verify_packages_import_all_required():
    @verify_packages_import(["tidy3d", "module2"], required="all")
    def my_function():
        pass

    with pytest.raises(Tidy3dImportError) as exc_info:
        my_function()

    # TODO assert here


def test_verify_packages_import_either_required():
    @verify_packages_import(["tidy3d", "module2"], required="either")
    def my_function():
        pass

    # When at least one module is imported, it should not raise an error
    my_function()

    @verify_packages_import(["module2", "module3"], required="either")
    def my_function2():
        pass

    with pytest.raises(Tidy3dImportError) as exc_info:
        my_function2()

    # TODO assert here


def test_check_import():
    # Mock the check_import function to use the mock_check_import function
    import sys

    sys.modules["tidy3d"].check_import = mock_check_import

    assert mock_check_import("tidy3d") == True
    assert mock_check_import("module2") == False


if __name__ == "__main__":
    pytest.main()
