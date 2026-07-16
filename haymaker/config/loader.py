"""Strict YAML loading and conversion to typed framework settings."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import ib_insync as ibi
import yaml

from haymaker.supervisor.settings import ConnectionSettings

from .settings import (
    BlotterSettings,
    ControllerSettings,
    DataloaderFuturesSettings,
    DataloaderSettings,
    DownloadSettings,
    FuturesSettings,
    LiveSettings,
    LoggingSettings,
    MongoSettings,
    OrderDefaults,
    PacingSettings,
    SaverSettings,
    StartupSettings,
    StateMachineSettings,
    StorageSettings,
    TimeoutPolicy,
)

if TYPE_CHECKING:
    from .cli_options import DataloaderCommand, LiveCommand

Profile = Literal["live", "dataloader"]
module_directory = Path(__file__).parent


class ConfigError(ValueError):
    """Raised when framework configuration is missing or invalid."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    """Construct one YAML mapping and reject duplicate keys."""

    explicit_keys: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        if key in explicit_keys:
            raise ConfigError(f"Duplicate YAML key: {key!r}")
        explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one safe YAML mapping.

    Args:
        path: YAML file to load.

    Returns:
        Parsed mapping, or an empty mapping for an empty document.

    Raises:
        ConfigError: If YAML is invalid or its root is not a mapping.
    """

    filename = Path(path).expanduser()
    try:
        with filename.open() as stream:
            data = yaml.load(stream, Loader=UniqueKeySafeLoader)
    except ConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load configuration {filename}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root in {filename} must be a mapping")
    if not all(isinstance(key, str) for key in data):
        raise ConfigError(f"Configuration keys in {filename} must be strings")
    return cast(dict[str, Any], data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return recursively merged mappings without mutating either input."""

    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            if "type" in value and value.get("type") != current.get("type"):
                merged[key] = dict(value)
            else:
                merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _set_path(config: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set a dotted override path in a nested mapping."""

    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ConfigError(f"Invalid configuration path: {path!r}")
    target = config
    for part in parts[:-1]:
        existing = target.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            target[part] = nested
            target = nested
        elif isinstance(existing, MutableMapping):
            target = cast(MutableMapping[str, Any], existing)
        else:
            raise ConfigError(f"Cannot set nested value below {part!r} in {path!r}")
    target[parts[-1]] = value


def _merged_config(
    profile: Profile,
    config_file: Path | None,
    overrides: Sequence[tuple[str, Any]],
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Load and merge all configuration sources for a command profile."""

    config = load_yaml(module_directory / f"{profile}_base_config.yaml")
    environment = os.environ if environ is None else environ
    selector = {
        "live": "HAYMAKER_HAYMAKER_CONFIG_OVERRIDES",
        "dataloader": "HAYMAKER_DATALOADER_CONFIG_OVERRIDES",
    }[profile]
    if environment_file := environment.get(selector):
        config = deep_merge(config, load_yaml(environment_file))
    if config_file is not None:
        config = deep_merge(config, load_yaml(config_file))
    command_overrides: dict[str, Any] = {}
    for path, value in overrides:
        _set_path(command_overrides, path, value)
    return deep_merge(config, command_overrides)


T = TypeVar("T")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    """Return a string-keyed mapping or raise a path-aware error."""

    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return dict(value)


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return one required top-level configuration section."""

    if key not in config:
        raise ConfigError(f"Missing configuration section: {key}")
    return _mapping(config[key], key)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject keys outside the explicit schema at one path."""

    if unknown := set(data) - allowed:
        names = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown configuration key(s) at {path}: {names}")


def _typed(value: Any, expected: type[T], path: str) -> T:
    """Validate and return one scalar with a path-aware type error."""

    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{path} must be an int")
    if (
        expected is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return cast(T, float(value))
    if not isinstance(value, expected):
        raise ConfigError(
            f"{path} must be {expected.__name__}, not {type(value).__name__}"
        )
    return value


def _optional_str(value: Any, path: str) -> str | None:
    """Validate an optional string value."""

    if value is None:
        return None
    return _typed(value, str, path)


def _positive_int(value: Any, path: str) -> int:
    """Validate a strictly positive integer value."""

    result = _typed(value, int, path)
    if result <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return result


def _non_negative_number(value: Any, path: str) -> float:
    """Validate a non-negative numeric value and normalize it to float."""

    result = _typed(value, float, path)
    if result < 0:
        raise ConfigError(f"{path} cannot be negative")
    return result


def _parse_logging(data: Mapping[str, Any], *, default_file: str) -> LoggingSettings:
    """Build validated logging settings from plain configuration data."""

    _reject_unknown(data, {"config_file", "directory", "log_broker"}, "logging")
    config_file = data.get("config_file", default_file)
    return LoggingSettings(
        config_file=(
            Path(_typed(config_file, str, "logging.config_file")).expanduser()
            if config_file is not None
            else None
        ),
        directory=_typed(data.get("directory", "logs"), str, "logging.directory"),
        log_broker=_typed(data.get("log_broker", False), bool, "logging.log_broker"),
    )


def _parse_connection(data: Mapping[str, Any]) -> ConnectionSettings:
    """Build supervisor settings and explicitly construct its probe contract."""

    allowed = {field.name for field in fields(ConnectionSettings)}
    _reject_unknown(data, allowed, "connection")
    probe_data = _mapping(data.get("probe_contract", {}), "connection.probe_contract")
    contract_fields = {field.name for field in fields(ibi.Contract)}
    _reject_unknown(probe_data, contract_fields, "connection.probe_contract")
    try:
        probe_contract = (
            ibi.Contract.create(**probe_data) if probe_data else ibi.Forex("EURUSD")
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid connection.probe_contract: {exc}") from exc
    return ConnectionSettings(
        host=_typed(data.get("host", "127.0.0.1"), str, "connection.host"),
        port=_positive_int(data.get("port", 4002), "connection.port"),
        client_id=_typed(data.get("client_id", 0), int, "connection.client_id"),
        connect_timeout=_non_negative_number(
            data.get("connect_timeout", 15), "connection.connect_timeout"
        ),
        retry_delay=_non_negative_number(
            data.get("retry_delay", 30), "connection.retry_delay"
        ),
        app_timeout=_non_negative_number(
            data.get("app_timeout", 90), "connection.app_timeout"
        ),
        probe_contract=probe_contract,
        probe_timeout=_non_negative_number(
            data.get("probe_timeout", 15), "connection.probe_timeout"
        ),
        connection_lost_retry_delay=_non_negative_number(
            data.get("connection_lost_retry_delay", 90),
            "connection.connection_lost_retry_delay",
        ),
        auto_recovery_grace_period=_non_negative_number(
            data.get("auto_recovery_grace_period", 120),
            "connection.auto_recovery_grace_period",
        ),
        restart_on_recovered_connection=_typed(
            data.get("restart_on_recovered_connection", False),
            bool,
            "connection.restart_on_recovered_connection",
        ),
        log_datafarm_status=_typed(
            data.get("log_datafarm_status", True),
            bool,
            "connection.log_datafarm_status",
        ),
    )


def _parse_storage(data: Mapping[str, Any]) -> StorageSettings:
    """Build validated persistence and filesystem settings."""

    _reject_unknown(
        data,
        {
            "base_directory",
            "mongodb",
            "block_library",
            "market_data_library",
            "dataframe_save_frequency",
        },
        "storage",
    )
    mongo = _mapping(data.get("mongodb", {}), "storage.mongodb")
    _reject_unknown(mongo, {"client", "database"}, "storage.mongodb")
    client = _mapping(mongo.get("client", {}), "storage.mongodb.client")
    return StorageSettings(
        base_directory=_typed(
            data.get("base_directory", "ib_data"), str, "storage.base_directory"
        ),
        mongodb=MongoSettings(
            client=client,
            database=_optional_str(mongo.get("database"), "storage.mongodb.database"),
        ),
        block_library=_optional_str(data.get("block_library"), "storage.block_library"),
        market_data_library=_typed(
            data.get("market_data_library", "market_data"),
            str,
            "storage.market_data_library",
        ),
        dataframe_save_frequency=_typed(
            data.get("dataframe_save_frequency", 900),
            int,
            "storage.dataframe_save_frequency",
        ),
    )


def _parse_startup(data: Mapping[str, Any]) -> StartupSettings:
    """Build validated one-run live startup settings."""

    allowed = {field.name for field in fields(StartupSettings)}
    _reject_unknown(data, allowed, "startup")
    return StartupSettings(
        **{key: _typed(data.get(key, False), bool, f"startup.{key}") for key in allowed}
    )


def _parse_controller(data: Mapping[str, Any]) -> ControllerSettings:
    """Build validated live controller policy settings."""

    allowed = {field.name for field in fields(ControllerSettings)}
    _reject_unknown(data, allowed, "controller")
    ignore_errors = data.get("ignore_errors", [])
    if not isinstance(ignore_errors, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in ignore_errors
    ):
        raise ConfigError("controller.ignore_errors must be a list of integers")
    roll_time = data.get("future_roll_time")
    if roll_time is not None:
        if (
            not isinstance(roll_time, list)
            or len(roll_time) != 2
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in roll_time
            )
        ):
            raise ConfigError("controller.future_roll_time must be [hour, minute]")
        if not 0 <= roll_time[0] <= 23 or not 0 <= roll_time[1] <= 59:
            raise ConfigError("controller.future_roll_time must be a valid UTC time")
        future_roll_time = (roll_time[0], roll_time[1])
    else:
        future_roll_time = None
    missing_brackets = data.get("missing_brackets", "ignore")
    if missing_brackets not in {"ignore", "warn", "remove"}:
        raise ConfigError("controller.missing_brackets must be ignore, warn, or remove")
    return ControllerSettings(
        log_order_events=_typed(
            data.get("log_order_events", False), bool, "controller.log_order_events"
        ),
        sync_frequency=_typed(
            data.get("sync_frequency", 0), int, "controller.sync_frequency"
        ),
        health_check_frequency=_typed(
            data.get("health_check_frequency", 0),
            int,
            "controller.health_check_frequency",
        ),
        execution_verification_delay=_typed(
            data.get("execution_verification_delay", 0),
            int,
            "controller.execution_verification_delay",
        ),
        execution_verification_max_retries=_typed(
            data.get("execution_verification_max_retries", 5),
            int,
            "controller.execution_verification_max_retries",
        ),
        broker_request_timeout=_typed(
            data.get("broker_request_timeout", 10),
            int,
            "controller.broker_request_timeout",
        ),
        sync_max_attempts=_typed(
            data.get("sync_max_attempts", 3), int, "controller.sync_max_attempts"
        ),
        sync_resync_delay=_typed(
            data.get("sync_resync_delay", 1), float, "controller.sync_resync_delay"
        ),
        cancel_unknown_trades=_typed(
            data.get("cancel_unknown_trades", False),
            bool,
            "controller.cancel_unknown_trades",
        ),
        missing_brackets=cast(Literal["ignore", "warn", "remove"], missing_brackets),
        ignore_errors=tuple(ignore_errors),
        future_roll_time=future_roll_time,
    )


def _parse_state_machine(data: Mapping[str, Any]) -> StateMachineSettings:
    """Build validated state persistence settings."""

    allowed = {field.name for field in fields(StateMachineSettings)}
    _reject_unknown(data, allowed, "state_machine")
    return StateMachineSettings(
        save_delay=_non_negative_number(
            data.get("save_delay", 1), "state_machine.save_delay"
        ),
        strategy_collection_name=_typed(
            data.get("strategy_collection_name", "strategies"),
            str,
            "state_machine.strategy_collection_name",
        ),
        order_collection_name=_typed(
            data.get("order_collection_name", "orders"),
            str,
            "state_machine.order_collection_name",
        ),
        max_rejected_orders=_positive_int(
            data.get("max_rejected_orders", 3),
            "state_machine.max_rejected_orders",
        ),
    )


def _parse_saver(data: Mapping[str, Any]) -> SaverSettings:
    """Build one safe built-in blotter saver specification."""

    _reject_unknown(data, {"type", "options"}, "blotter.saver")
    saver_type = _typed(data.get("type"), str, "blotter.saver.type")
    options = _mapping(data.get("options", {}), "blotter.saver.options")
    if saver_type == "csv":
        _reject_unknown(
            options, {"name", "folder", "use_timestamp"}, "blotter.saver.options"
        )
        if "name" not in options:
            raise ConfigError("blotter.saver.options.name is required for csv")
    elif saver_type == "mongo":
        _reject_unknown(options, {"collection", "query_key"}, "blotter.saver.options")
        if "collection" not in options:
            raise ConfigError("blotter.saver.options.collection is required for mongo")
    else:
        raise ConfigError("blotter.saver.type must be csv or mongo")
    return SaverSettings(
        type=cast(Literal["csv", "mongo"], saver_type), options=options
    )


def _parse_blotter(data: Mapping[str, Any]) -> BlotterSettings:
    """Build validated blotter enablement and saver settings."""

    _reject_unknown(data, {"enabled", "saver"}, "blotter")
    enabled = _typed(data.get("enabled", True), bool, "blotter.enabled")
    saver_data = data.get("saver")
    saver = _parse_saver(_mapping(saver_data, "blotter.saver")) if saver_data else None
    if enabled and saver is None:
        raise ConfigError("blotter.saver is required when blotter.enabled is true")
    return BlotterSettings(enabled=enabled, saver=saver)


def _parse_order_mapping(value: Any, path: str) -> dict[str, Any]:
    """Validate IB order defaults and explicitly convert algorithm tags."""

    order = _mapping(value, path)
    order_fields = {field.name for field in fields(ibi.Order)}
    _reject_unknown(order, order_fields, path)
    if "algoParams" in order:
        params = order["algoParams"]
        if not isinstance(params, list):
            raise ConfigError(f"{path}.algoParams must be a list")
        converted: list[ibi.TagValue] = []
        for index, item in enumerate(params):
            param_path = f"{path}.algoParams[{index}]"
            mapping = _mapping(item, param_path)
            _reject_unknown(mapping, {"tag", "value"}, param_path)
            converted.append(
                ibi.TagValue(
                    _typed(mapping.get("tag"), str, f"{param_path}.tag"),
                    _typed(mapping.get("value"), str, f"{param_path}.value"),
                )
            )
        order["algoParams"] = converted
    try:
        ibi.Order(**order)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {path}: {exc}") from exc
    return order


def _parse_orders(data: Mapping[str, Any]) -> OrderDefaults:
    """Build validated execution-model order defaults."""

    _reject_unknown(
        data, {"open", "close", "stop", "take_profit", "oca_type"}, "orders"
    )
    return OrderDefaults(
        open=_parse_order_mapping(data.get("open", {}), "orders.open"),
        close=_parse_order_mapping(data.get("close", {}), "orders.close"),
        stop=_parse_order_mapping(data.get("stop", {}), "orders.stop"),
        take_profit=_parse_order_mapping(
            data.get("take_profit", {}), "orders.take_profit"
        ),
        oca_type=_positive_int(data.get("oca_type", 1), "orders.oca_type"),
    )


def _parse_timeout(data: Mapping[str, Any]) -> TimeoutPolicy:
    """Build the default streamer timeout policy."""

    _reject_unknown(data, {"seconds", "action"}, "timeout")
    action = data.get("action", "restart")
    if action not in {"restart", "log"}:
        raise ConfigError("timeout.action must be restart or log")
    return TimeoutPolicy(
        seconds=_non_negative_number(data.get("seconds", 0), "timeout.seconds"),
        action=cast(Literal["restart", "log"], action),
    )


def _parse_futures(data: Mapping[str, Any]) -> FuturesSettings:
    """Build validated live futures-roll offsets."""

    _reject_unknown(data, {"roll_bdays", "roll_margin_bdays"}, "futures")
    return FuturesSettings(
        roll_bdays=_positive_int(data.get("roll_bdays", 3), "futures.roll_bdays"),
        roll_margin_bdays=_positive_int(
            data.get("roll_margin_bdays", 3), "futures.roll_margin_bdays"
        ),
    )


def load_live_settings(
    command: "LiveCommand", environ: Mapping[str, str] | None = None
) -> LiveSettings:
    """Load and validate settings for one live command."""

    config = _merged_config("live", command.config_file, command.overrides, environ)
    allowed = {
        "startup",
        "connection",
        "logging",
        "controller",
        "state_machine",
        "storage",
        "blotter",
        "orders",
        "timeout",
        "futures",
    }
    _reject_unknown(config, allowed, "root")
    return LiveSettings(
        startup=_parse_startup(_section(config, "startup")),
        connection=_parse_connection(_section(config, "connection")),
        logging=_parse_logging(
            _section(config, "logging"), default_file="logging_config.yaml"
        ),
        controller=_parse_controller(_section(config, "controller")),
        state_machine=_parse_state_machine(_section(config, "state_machine")),
        storage=_parse_storage(_section(config, "storage")),
        blotter=_parse_blotter(_section(config, "blotter")),
        orders=_parse_orders(_section(config, "orders")),
        timeout=_parse_timeout(_section(config, "timeout")),
        futures=_parse_futures(_section(config, "futures")),
    )


def _parse_download(data: Mapping[str, Any]) -> DownloadSettings:
    """Build validated historical-download policy settings."""

    allowed = {field.name for field in fields(DownloadSettings)}
    _reject_unknown(data, allowed, "download")
    gap_fill_mode = data.get("gap_fill_mode", "off")
    if gap_fill_mode not in {"off", "heuristic", "schedule", "auto"}:
        raise ConfigError("download.gap_fill_mode has an invalid value")
    max_lookback = data.get("max_lookback_days")
    if max_lookback is not None:
        max_lookback = _positive_int(max_lookback, "download.max_lookback_days")
    return DownloadSettings(
        source=_typed(data.get("source", "contracts.csv"), str, "download.source"),
        bar_size=_typed(data.get("bar_size", "30 secs"), str, "download.bar_size"),
        what_to_show=_typed(
            data.get("what_to_show", "TRADES"), str, "download.what_to_show"
        ),
        max_lookback_days=max_lookback,
        gap_fill_mode=cast(
            Literal["off", "heuristic", "schedule", "auto"], gap_fill_mode
        ),
        use_rth=_typed(data.get("use_rth", False), bool, "download.use_rth"),
        save_every_chunks=_positive_int(
            data.get("save_every_chunks", 10), "download.save_every_chunks"
        ),
        number_of_workers=_positive_int(
            data.get("number_of_workers", 10), "download.number_of_workers"
        ),
    )


def _parse_pacing(data: Mapping[str, Any]) -> PacingSettings:
    """Build validated dataloader request-pacing settings."""

    _reject_unknown(data, {"no_restriction", "allowance_fraction"}, "pacing")
    allowance = _typed(
        data.get("allowance_fraction", 1.0), float, "pacing.allowance_fraction"
    )
    if allowance <= 0:
        raise ConfigError("pacing.allowance_fraction must be positive")
    return PacingSettings(
        no_restriction=_typed(
            data.get("no_restriction", False), bool, "pacing.no_restriction"
        ),
        allowance_fraction=allowance,
    )


def _parse_dataloader_futures(
    data: Mapping[str, Any],
) -> DataloaderFuturesSettings:
    """Build validated dataloader futures-selection settings."""

    allowed = {field.name for field in fields(DataloaderFuturesSettings)}
    _reject_unknown(data, allowed, "futures")
    selector = _typed(
        data.get("selector", "current_and_expired"), str, "futures.selector"
    )
    if selector not in {
        "contfuture",
        "fullchain",
        "current",
        "exact",
        "current_and_contfuture",
        "current_and_expired",
    }:
        raise ConfigError("futures.selector has an invalid value")
    full_chain_spec = _typed(
        data.get("full_chain_spec", "full"), str, "futures.full_chain_spec"
    )
    if full_chain_spec not in {"full", "active", "expired"}:
        raise ConfigError("futures.full_chain_spec has an invalid value")
    return DataloaderFuturesSettings(
        selector=selector,
        full_chain_spec=full_chain_spec,
        current_index=_typed(
            data.get("current_index", 0), int, "futures.current_index"
        ),
    )


def load_dataloader_settings(
    command: "DataloaderCommand", environ: Mapping[str, str] | None = None
) -> DataloaderSettings:
    """Load and validate settings for one dataloader command."""

    config = _merged_config(
        "dataloader", command.config_file, command.overrides, environ
    )
    allowed = {"connection", "logging", "storage", "download", "pacing", "futures"}
    _reject_unknown(config, allowed, "root")
    return DataloaderSettings(
        connection=_parse_connection(_section(config, "connection")),
        logging=_parse_logging(
            _section(config, "logging"),
            default_file="dataloader_logging_config.yaml",
        ),
        storage=_parse_storage(_section(config, "storage")),
        download=_parse_download(_section(config, "download")),
        pacing=_parse_pacing(_section(config, "pacing")),
        futures=_parse_dataloader_futures(_section(config, "futures")),
    )
