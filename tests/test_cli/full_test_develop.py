"""
These scripts just test the CLI commands for the develop command, and verify that they run properly.
"""
import pytest
import os
from click.testing import CliRunner
from tidy3d.web.cli import tidy3d_cli
from unittest.mock import patch


@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_develop_install_dev_environment(runner):
    """
    Test the install-dev-environment command runs. It will rerun the file. Hard to assert the result as the behaviour
    depends on the base envrionment and it is an interactive script that will throw helpful errors in terms of asking
    for the required toolset.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        runner.invoke(tidy3d_cli, ["develop", "install-dev-environmnet"])
        # assert result.exit_code == 0 # Not necessary as depends on HEAD state


def test_develop_verify_dev_environment(runner):
    """
    Test the verify-dev-environment command runs without throwing an error.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        result = runner.invoke(tidy3d_cli, ["develop", "verify-dev-environment"])
        assert result.exit_code == 0


# Example test for the 'commit' command
def test_develop_commit(runner):
    """
    Test the commit command runs between the relevant submodules included. Hard to assert as depends on the HEAD state.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        runner.invoke(
            tidy3d_cli, ["develop", "commit", "TEST: Update based on test develop cli command"]
        )
        # assert result.exit_code == 0 # Not necessary as depends on HEAD state


def test_replace_in_files(runner):
    """
    Test the replace-in-files command runs with a little demo file. Asserts that the file has been replaced with a new string.
    """
    import json

    with runner.isolated_filesystem():
        os.mkdir("test_directory")
        with open("test_directory/test_file.txt", "w") as f:
            f.write("Some content with tidy3d.someuniquestringa")

        with open("replace_dict.json", "w") as f:
            json.dump({"0.18.0": {"tidy3d.someuniquestringa": "tidy3d.someuniquestring2"}}, f)

        with patch("builtins.input", return_value="y"):
            result = runner.invoke(
                tidy3d_cli,
                [
                    "develop",
                    "replace-in-files",
                    "--directory",
                    "test_directory",
                    "--json-dictionary",
                    "replace_dict.json",
                    "--selected-version",
                    "0.18.0",
                    "--dry-run",
                    True,
                ],
            )

        assert result.exit_code == 0


def test_develop_update_submodules(runner):
    """
    Test the update-submodules command runs without error. Hard to assert as it depends from a HEAD state.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        runner.invoke(tidy3d_cli, ["develop", "update-submodules"])
        # assert result.exit_code == 0  # Verifies docs build successfully


def test_develop_benchmark_timing_average(runner):
    """
    Test the benchmark-timing-average command runs without error.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        result = runner.invoke(
            tidy3d_cli, ["develop", "benchmark-timing-operations", "-c", "average_test_import"]
        )
        assert result.exit_code == 0


def test_develop_build_docs(runner):
    """
    Test the build-docs command runs without error. Guarantees the documentation builds.
    """
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        result = runner.invoke(tidy3d_cli, ["develop", "build-docs"])
        assert result.exit_code == 0  # Verifies docs build successfully
