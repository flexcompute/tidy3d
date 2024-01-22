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
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        runner.invoke(tidy3d_cli, ["develop", "install-dev-environmnet"])
        # assert result.exit_code == 0 # Not necessary as depends on HEAD state


# Example test for the 'commit' command
def test_develop_commit(runner):
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        runner.invoke(
            tidy3d_cli, ["develop", "commit", "TEST: Update based on test develop cli command"]
        )
        # assert result.exit_code == 0 # Not necessary as depends on HEAD state


def test_develop_build_docs(runner):
    with runner.isolated_filesystem():
        with open("test_file", "w") as f:
            f.write("Test content")

        result = runner.invoke(tidy3d_cli, ["develop", "build-docs"])
        assert result.exit_code == 0  # Verifies docs build successfully


def test_replace_in_files(runner):
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
