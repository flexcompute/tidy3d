from __future__ import annotations

from click.testing import CliRunner

from tidy3d.web.cli.app import tidy3d_cli


def test_tidy3d_root_command_names_are_unique():
    runner = CliRunner()
    result = runner.invoke(tidy3d_cli, ["--help"])
    assert result.exit_code == 0, result.output

    command_names = list(tidy3d_cli.commands.keys())
    assert len(command_names) == len(set(command_names))
    assert "config" in tidy3d_cli.commands
    assert {"configure", "convert", "develop"}.issubset(set(command_names))
    assert "migrate" not in tidy3d_cli.commands


def test_config_group_commands_are_namespaced():
    config_group = tidy3d_cli.commands["config"]

    migrate_cmd = config_group.commands["migrate"]
    reset_cmd = config_group.commands["reset"]

    assert migrate_cmd.name == "config-migrate"
    assert reset_cmd.name == "config-reset"
    assert "config-migrate" not in tidy3d_cli.commands
    assert "config-reset" not in tidy3d_cli.commands
