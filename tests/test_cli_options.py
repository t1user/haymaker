"""Tests for profile-specific command parsing."""

from pathlib import Path

import pytest

from haymaker.config.cli_options import (
    dataloader_parser,
    parse_dataloader_args,
    parse_live_args,
)


def test_live_command_keeps_module_out_of_config_overrides() -> None:
    command = parse_live_args(["strategy.py"])

    assert command.module_path == Path("strategy.py")
    assert command.overrides == ()


def test_set_option_is_repeatable_and_typed() -> None:
    command = parse_live_args(
        [
            "strategy.py",
            "--set-option",
            "controller.sync_frequency",
            "60",
            "--set-option",
            "logging.log_broker",
            "true",
        ]
    )

    assert command.overrides == (
        ("controller.sync_frequency", 60),
        ("logging.log_broker", True),
    )


def test_live_dedicated_options_are_appended_after_generic_options() -> None:
    command = parse_live_args(
        ["strategy.py", "--set-option", "startup.reset", "false", "--reset"]
    )

    assert command.overrides[-1] == ("startup.reset", True)


def test_dataloader_positional_and_gap_options_are_explicit_overrides() -> None:
    command = parse_dataloader_args(["contracts.csv", "--gap-fill-mode", "schedule"])

    assert command.overrides == (
        ("download.source", "contracts.csv"),
        ("download.gap_fill_mode", "schedule"),
    )


def test_config_file_path_expands_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/example-home")

    command = parse_live_args(["strategy.py", "--file", "~/live.yaml"])

    assert command.config_file == Path("/tmp/example-home/live.yaml")


def test_old_cli_alias_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_live_args(["strategy.py", "--coldstart"])


def test_dataloader_help_uses_normalized_option_name() -> None:
    help_text = dataloader_parser().format_help()

    assert "--gap-fill-mode" in help_text
    assert "--gap_fill_mode" not in help_text
