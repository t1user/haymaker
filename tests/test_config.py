"""Tests for strict typed framework configuration loading."""

from __future__ import annotations

from pathlib import Path

import ib_insync as ibi
import pytest

import haymaker.config as config_package
from haymaker.config import (
    ConfigError,
    DataloaderCommand,
    LiveCommand,
    load_dataloader_settings,
    load_live_settings,
    parse_live_args,
)
from haymaker.config.loader import deep_merge, load_yaml


def live_command(
    config_file: Path | None = None,
    *overrides: tuple[str, object],
) -> LiveCommand:
    """Return a minimal live loader command."""

    return LiveCommand(Path("strategy.py"), config_file, overrides)


def test_process_global_config_singleton_is_not_exported() -> None:
    """Importing configuration must not create process-owned settings."""

    assert not hasattr(config_package, "CONFIG")


def test_live_defaults_are_typed() -> None:
    settings = load_live_settings(live_command(), environ={})

    assert settings.connection.client_id == 0
    assert settings.connection.probe_contract == ibi.Forex("EURUSD")
    assert settings.timeout.seconds == 300
    assert settings.orders.open["algoParams"] == [
        ibi.TagValue("adaptivePriority", "Normal")
    ]


def test_dataloader_defaults_are_profile_specific() -> None:
    settings = load_dataloader_settings(DataloaderCommand(None, ()), environ={})

    assert settings.connection.client_id == 1
    assert settings.download.source == "contracts.csv"
    assert settings.download.gap_fill_mode == "off"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("futures.selector", "unknown"),
        ("futures.full_chain_spec", "unknown"),
    ],
)
def test_dataloader_rejects_unknown_futures_policy(path: str, value: str) -> None:
    command = DataloaderCommand(None, ((path, value),))

    with pytest.raises(ConfigError, match=path):
        load_dataloader_settings(command, environ={})


def test_deep_merge_recurses_and_replaces_lists() -> None:
    merged = deep_merge(
        {"section": {"kept": 1, "values": [1]}, "root": "old"},
        {"section": {"values": [2]}, "root": "new"},
    )

    assert merged == {
        "section": {"kept": 1, "values": [2]},
        "root": "new",
    }


def test_deep_merge_replaces_discriminated_mapping_when_type_changes() -> None:
    merged = deep_merge(
        {"saver": {"type": "csv", "options": {"name": "blotter"}}},
        {"saver": {"type": "mongo", "options": {"collection": "blotter"}}},
    )

    assert merged == {"saver": {"type": "mongo", "options": {"collection": "blotter"}}}


def test_dotted_overrides_replace_discriminated_mapping_as_one_layer() -> None:
    settings = load_live_settings(
        live_command(
            None,
            ("blotter.saver.type", "mongo"),
            ("blotter.saver.options.collection", "audit"),
        ),
        environ={},
    )

    assert settings.blotter.saver is not None
    assert settings.blotter.saver.type == "mongo"
    assert settings.blotter.saver.options == {"collection": "audit"}


def test_source_precedence_environment_yaml_then_cli_yaml_then_cli_values(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / "environment.yaml"
    environment_file.write_text(
        "controller:\n  sync_frequency: 10\n  broker_request_timeout: 20\n"
    )
    cli_file = tmp_path / "cli.yaml"
    cli_file.write_text("controller:\n  sync_frequency: 30\n")
    command = live_command(
        cli_file,
        ("controller.sync_frequency", 40),
    )

    settings = load_live_settings(
        command,
        environ={"HAYMAKER_HAYMAKER_CONFIG_OVERRIDES": str(environment_file)},
    )

    assert settings.controller.sync_frequency == 40
    assert settings.controller.broker_request_timeout == 20


def test_direct_environment_values_are_ignored() -> None:
    settings = load_live_settings(
        live_command(),
        environ={
            "HAYMAKER_CONNECTION__PORT": "9999",
            "HAYMAKER_LOGGING_CONFIG": "other.yaml",
        },
    )

    assert settings.connection.port == 4002
    assert settings.logging.config_file == Path("logging_config.yaml")


def test_dedicated_cli_option_wins_over_generic_override() -> None:
    command = parse_live_args(
        ["strategy.py", "--set-option", "startup.reset", "false", "--reset"]
    )

    settings = load_live_settings(command, environ={})

    assert settings.startup.reset is True


def test_cli_values_preserve_yaml_scalar_and_collection_types() -> None:
    command = parse_live_args(
        [
            "strategy.py",
            "--set-option",
            "logging.log_broker",
            "true",
            "--set-option",
            "controller.future_roll_time",
            "[15, 30]",
        ]
    )

    settings = load_live_settings(command, environ={})

    assert settings.logging.log_broker is True
    assert settings.controller.future_roll_time == (15, 30)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- one\n- two\n", "root"),
        ("value: 1\nvalue: 2\n", "Duplicate YAML key"),
        (
            "value: !!python/object/apply:ib_insync.Forex [EURUSD]\n",
            "constructor",
        ),
    ],
)
def test_yaml_rejects_invalid_document_shapes(
    tmp_path: Path, text: str, message: str
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(text)

    with pytest.raises(ConfigError, match=message):
        load_yaml(config_file)


def test_safe_yaml_loader_supports_standard_mapping_merges(tmp_path: Path) -> None:
    config_file = tmp_path / "merged.yaml"
    config_file.write_text(
        "defaults: &defaults\n"
        "  sync_frequency: 10\n"
        "  health_check_frequency: 20\n"
        "controller:\n"
        "  <<: *defaults\n"
        "  sync_frequency: 30\n"
    )

    config = load_yaml(config_file)

    assert config["controller"] == {
        "sync_frequency": 30,
        "health_check_frequency": 20,
    }


def test_unknown_old_key_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "old.yaml"
    config_file.write_text("coldstart: true\n")

    with pytest.raises(ConfigError, match="coldstart"):
        load_live_settings(live_command(config_file), environ={})


def test_wrong_type_reports_dotted_path(tmp_path: Path) -> None:
    config_file = tmp_path / "wrong.yaml"
    config_file.write_text("connection:\n  port: '4002'\n")

    with pytest.raises(ConfigError, match="connection.port"):
        load_live_settings(live_command(config_file), environ={})


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_live_settings(live_command(tmp_path / "missing.yaml"), environ={})


def test_plain_probe_contract_mapping_is_converted(tmp_path: Path) -> None:
    config_file = tmp_path / "probe.yaml"
    config_file.write_text(
        "connection:\n"
        "  probe_contract:\n"
        "    secType: STK\n"
        "    symbol: SPY\n"
        "    exchange: SMART\n"
        "    currency: USD\n"
    )

    settings = load_live_settings(live_command(config_file), environ={})

    assert settings.connection.probe_contract == ibi.Stock("SPY", "SMART", "USD")
