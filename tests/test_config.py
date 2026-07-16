"""Tests for live configuration merging and target-owned construction."""

from __future__ import annotations

from pathlib import Path

import ib_insync as ibi
import pytest

import haymaker.config as config_package
from haymaker.blotter import blotter_factory
from haymaker.config import (
    ConfigError,
    DataloaderCommand,
    LiveCommand,
    load_dataloader_config,
    load_live_config,
    parse_live_args,
)
from haymaker.config.loader import deep_merge, load_yaml
from haymaker.config.settings import StorageSettings
from haymaker.databases import StoreFactory
from haymaker.dataloader.contract_selectors import FuturesSelectionPolicy
from haymaker.order_defaults import OrderDefaults
from haymaker.supervisor import ConnectionSettings
from haymaker.timeout import TimeoutPolicy


def live_command(
    config_file: Path | None = None,
    *overrides: tuple[str, object],
) -> LiveCommand:
    """Return a minimal live loader command."""

    return LiveCommand(Path("strategy.py"), config_file, overrides)


def test_process_global_config_singleton_is_not_exported() -> None:
    """Importing configuration must not create process-owned settings."""

    assert not hasattr(config_package, "CONFIG")
    assert not hasattr(config_package, "load_live_settings")
    assert not hasattr(config_package, "load_dataloader_settings")


def test_live_defaults_are_composed_by_target_objects() -> None:
    config = load_live_config(live_command(), environ={})
    connection = ConnectionSettings.from_mapping(config.connection)
    timeout = TimeoutPolicy.from_mapping(config.timeout)
    orders = OrderDefaults.from_mapping(config.orders)

    assert connection.client_id == 0
    assert connection.probe_contract == ibi.Forex("EURUSD")
    assert timeout.seconds == 300
    assert orders.open["algoParams"] == [ibi.TagValue("adaptivePriority", "Normal")]


def test_order_defaults_reject_invalid_order_fields_during_construction() -> None:
    with pytest.raises(TypeError):
        OrderDefaults.from_mapping({"open": {"notAnOrderField": True}})


def test_timeout_policy_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="restart or log"):
        TimeoutPolicy.from_mapping({"action": "ignore"})


def test_blotter_factory_rejects_unknown_saver_type() -> None:
    with pytest.raises(ValueError, match="csv or mongo"):
        blotter_factory(
            {"enabled": True, "saver": {"type": "unknown", "options": {}}},
            StoreFactory(StorageSettings()),
        )


def test_dataloader_defaults_are_profile_specific() -> None:
    config = load_dataloader_config(DataloaderCommand(None, ()), environ={})
    connection = ConnectionSettings.from_mapping(config.connection)
    futures = FuturesSelectionPolicy.from_mapping(config.futures)

    assert connection.client_id == 1
    assert connection.app_timeout == 600
    assert config.download == {}
    assert config.pacing == {}
    assert futures.selector == "current_and_expired"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("futures.selector", "unknown"),
        ("futures.full_chain_spec", "unknown"),
    ],
)
def test_futures_target_rejects_unknown_policy(path: str, value: str) -> None:
    command = DataloaderCommand(None, ((path, value),))
    config = load_dataloader_config(command, environ={})

    with pytest.raises(ValueError, match="futures"):
        FuturesSelectionPolicy.from_mapping(config.futures)


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
    config = load_live_config(
        live_command(
            None,
            ("blotter.saver.type", "mongo"),
            ("blotter.saver.options.collection", "audit"),
        ),
        environ={},
    )

    assert config.blotter["saver"] == {
        "type": "mongo",
        "options": {"collection": "audit"},
    }


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

    config = load_live_config(
        command,
        environ={"HAYMAKER_HAYMAKER_CONFIG_OVERRIDES": str(environment_file)},
    )

    assert config.controller["sync_frequency"] == 40
    assert config.controller["broker_request_timeout"] == 20


def test_direct_environment_values_are_ignored() -> None:
    config = load_live_config(
        live_command(),
        environ={
            "HAYMAKER_CONNECTION__PORT": "9999",
            "HAYMAKER_LOGGING_CONFIG": "other.yaml",
        },
    )

    assert ConnectionSettings.from_mapping(config.connection).port == 4002
    assert config.logging == {}


def test_dedicated_cli_option_wins_over_generic_override() -> None:
    command = parse_live_args(
        ["strategy.py", "--set-option", "startup.reset", "false", "--reset"]
    )

    config = load_live_config(command, environ={})

    assert config.startup["reset"] is True


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

    config = load_live_config(command, environ={})

    assert config.logging["log_broker"] is True
    assert config.controller["future_roll_time"] == [15, 30]


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
        load_live_config(live_command(config_file), environ={})


def test_target_rejects_unknown_section_key(tmp_path: Path) -> None:
    config_file = tmp_path / "unknown.yaml"
    config_file.write_text("connection:\n  unknown: true\n")
    config = load_live_config(live_command(config_file), environ={})

    with pytest.raises(TypeError, match="unknown"):
        ConnectionSettings.from_mapping(config.connection)


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_live_config(live_command(tmp_path / "missing.yaml"), environ={})


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

    config = load_live_config(live_command(config_file), environ={})
    settings = ConnectionSettings.from_mapping(config.connection)

    assert settings.probe_contract == ibi.Stock("SPY", "SMART", "USD")
