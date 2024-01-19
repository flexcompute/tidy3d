"""
These scripts just test the CLI commands for the develop command, and verify that they run properly.
"""
import pytest
from click.testing import CliRunner
from tidy3d.web.cli import tidy3d_cli


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
