"""Safe profile loading, merging, and staged configuration construction."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import yaml

from .settings import (
    DataloaderConfig,
    LiveConfig,
    MongoSettings,
    StorageSettings,
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


def load_live_config(
    command: "LiveCommand", environ: Mapping[str, str] | None = None
) -> LiveConfig:
    """Load and merge configuration for one live command."""

    config = _merged_config("live", command.config_file, command.overrides, environ)
    allowed = {
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
    return LiveConfig(
        connection=_section(config, "connection"),
        logging=_section(config, "logging"),
        controller=_section(config, "controller"),
        state_machine=_section(config, "state_machine"),
        storage=_parse_storage(_section(config, "storage")),
        blotter=_section(config, "blotter"),
        orders=_section(config, "orders"),
        timeout=_section(config, "timeout"),
        futures=_section(config, "futures"),
    )


def load_dataloader_config(
    command: "DataloaderCommand", environ: Mapping[str, str] | None = None
) -> DataloaderConfig:
    """Load and merge configuration for one dataloader command."""

    config = _merged_config(
        "dataloader", command.config_file, command.overrides, environ
    )
    allowed = {"connection", "logging", "storage", "download", "pacing", "futures"}
    _reject_unknown(config, allowed, "root")
    return DataloaderConfig(
        connection=_section(config, "connection"),
        logging=_section(config, "logging"),
        storage=_parse_storage(_section(config, "storage")),
        download=_section(config, "download"),
        pacing=_section(config, "pacing"),
        futures=_section(config, "futures"),
    )
