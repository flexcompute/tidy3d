import pytest
from tidy3d.packaging import Tidy3dImportError, check_import, verify_packages_import

assert check_import("tidy3d") is True


# Mock module import function to simulate availability
def mock_check_import(module_name):
    """
    Mock the check_import function to simulate availability of tidy3d module.
    """
    if module_name == "tidy3d":
        return True
    return False


def test_verify_packages_import_all_required():
    """
    Test the verify_packages_import function with all required. Verifies that the decorator works to trigger a
    Tidy3dImportError when module2 is unavailable.
    """

    @verify_packages_import(["tidy3d", "module2"], required="all")
    def my_function():
        pass

    with pytest.raises(Tidy3dImportError):
        my_function()


def test_verify_packages_import_either_required():
    """
    Test the verify_packages_import function with either required. Verifies the decorator works by not triggering an
    error when the module2 is not found. However it should throw an error when either module2 or module3 are found.
    """

    @verify_packages_import(["tidy3d", "module2"], required="any")
    def my_function():
        pass

    # When at least one module is imported, it should not raise an error
    my_function()

    @verify_packages_import(["module2", "module3"], required="any")
    def my_function2():
        pass

    with pytest.raises(Tidy3dImportError):
        my_function2()


def test_check_import():
    """
    Test the check_import function with mock_check_import. Just standard test to verify the mock function works
    compared to check_import.
    """
    import sys

    sys.modules["tidy3d"].check_import = mock_check_import

    assert mock_check_import("tidy3d") is True
    assert mock_check_import("module2") is False


if __name__ == "__main__":
    pytest.main()
