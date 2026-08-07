"""Configuration aggregates and typed settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self


@dataclass(frozen=True)
class MongoClientSettings:
    """Keyword arguments used to construct a Mongo client."""

    client: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MongoSettings(MongoClientSettings):
    """Mongo client arguments and framework database name."""

    database: str | None = None


@dataclass(frozen=True)
class DataloaderStorageSettings:
    """Filesystem and Mongo client settings used by the dataloader."""

    base_directory: str = "ib_data"
    mongodb: MongoClientSettings = field(default_factory=MongoClientSettings)


@dataclass(frozen=True)
class StorageSettings:
    """Filesystem and framework Mongo settings used by live execution."""

    base_directory: str = "ib_data"
    mongodb: MongoSettings = field(default_factory=MongoSettings)


@dataclass(frozen=True)
class MarketDataStoreSettings:
    """Runtime defaults used by market-data components requesting a store.

    Attributes:
        library: Arctic dataframe library for persisted broker-bar history.
    """

    library: str = "market_data"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct and validate runtime-default market-data storage.

        Args:
            values: Merged ``market_data_store`` configuration section.

        Returns:
            Validated market-data store settings.
        """

        settings = cls(**dict(values))
        if not isinstance(settings.library, str):
            raise TypeError("market_data_store.library must be a string")
        if not settings.library:
            raise ValueError("market_data_store.library must not be empty")
        return settings


@dataclass(frozen=True)
class SignalFramePersistenceSettings:
    """Runtime defaults used by ``PandasSignalModel(persistence=True)``.

    Attributes:
        library: Arctic dataframe library for calculated Signal history.
    """

    library: str = "signal_data"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct and validate Signal dataframe persistence defaults.

        Args:
            values: Merged ``signal_persistence`` configuration section.

        Returns:
            Validated runtime persistence settings.
        """

        settings = cls(**dict(values))
        if not isinstance(settings.library, str):
            raise TypeError("signal_persistence.library must be a string")
        if not settings.library:
            raise ValueError("signal_persistence.library must not be empty")
        return settings


@dataclass(frozen=True)
class TimeoutPolicy:
    """Runtime defaults for market-data inactivity monitoring.

    Live users normally configure these values through the YAML ``timeout``
    section rather than constructing this class. ``LiveRuntime`` installs the
    resulting policy in ``RuntimeContext`` for
    :meth:`haymaker.components.MarketDataTimeout.from_atom`. General
    :class:`haymaker.components.EventTimeout` instances are user-configured and
    do not consult this policy.
    """

    seconds: float = 0
    action: Literal["restart", "log"] = "restart"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct and validate a timeout policy from plain configuration.

        Args:
            values: Merged ``timeout`` configuration section.

        Returns:
            Timeout policy ready to install in a runtime context.
        """

        policy = cls(**dict(values))
        if policy.seconds < 0:
            raise ValueError("timeout.seconds cannot be negative")
        if policy.action not in ("restart", "log"):
            raise ValueError("timeout.action must be restart or log")
        return policy

    @property
    def log_only(self) -> bool:
        """Return whether a timeout should only be logged."""

        return self.action == "log"


@dataclass(frozen=True)
class LiveConfig:
    """Merged live configuration grouped by target or subsystem boundary.

    Attributes:
        connection: Broker connection and recovery options.
        logging: Logging setup and broker-log options.
        controller: Controller startup, reconciliation, and scheduling options.
        book: Typed state, order persistence, and rejection options.
        storage: Filesystem and framework Mongo infrastructure settings.
        market_data_store: Defaults for runtime-created broker-bar datastores.
        signal_persistence: Defaults for optional Signal dataframe persistence.
        blotter: Blotter enablement and saver specification.
        orders: Default IB order fields.
        timeout: Default streamer timeout policy.
        futures: Live futures selection offsets.
    """

    connection: Mapping[str, Any]
    logging: Mapping[str, Any]
    controller: Mapping[str, Any]
    book: Mapping[str, Any]
    storage: StorageSettings
    market_data_store: Mapping[str, Any]
    signal_persistence: Mapping[str, Any]
    blotter: Mapping[str, Any]
    orders: Mapping[str, Any]
    timeout: Mapping[str, Any]
    futures: Mapping[str, Any]


@dataclass(frozen=True)
class DataloaderConfig:
    """Merged dataloader configuration grouped by target or subsystem boundary.

    Attributes:
        connection: Broker connection and recovery options.
        logging: Logging setup options.
        storage: Filesystem and Mongo client settings used by the dataloader.
        download: Historical request and worker options.
        pacing: Client-side request pacing options.
        futures: Futures contract-selection policy.
    """

    connection: Mapping[str, Any]
    logging: Mapping[str, Any]
    storage: DataloaderStorageSettings
    download: Mapping[str, Any]
    pacing: Mapping[str, Any]
    futures: Mapping[str, Any]
