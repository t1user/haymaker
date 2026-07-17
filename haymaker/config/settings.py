"""Configuration aggregates and retained storage settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MongoSettings:
    """Mongo client arguments and framework database name."""

    client: Mapping[str, Any] = field(default_factory=dict)
    database: str | None = None


@dataclass(frozen=True)
class StorageSettings:
    """Filesystem, Mongo, and default market-data storage settings."""

    base_directory: str = "ib_data"
    mongodb: MongoSettings = field(default_factory=MongoSettings)
    block_library: str | None = None
    market_data_library: str = "market_data"
    dataframe_save_frequency: int = 900


@dataclass(frozen=True)
class LiveConfig:
    """Merged live configuration grouped by target or subsystem boundary.

    Attributes:
        connection: Broker connection and recovery options.
        logging: Logging setup and broker-log options.
        controller: Controller startup, reconciliation, and scheduling options.
        state_machine: State persistence and rejection options.
        storage: Retained typed storage settings pending datastore refactoring.
        blotter: Blotter enablement and saver specification.
        orders: Default IB order fields.
        timeout: Default streamer timeout policy.
        futures: Live futures selection offsets.
    """

    connection: Mapping[str, Any]
    logging: Mapping[str, Any]
    controller: Mapping[str, Any]
    state_machine: Mapping[str, Any]
    storage: StorageSettings
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
        storage: Retained typed storage settings pending datastore refactoring.
        download: Historical request and worker options.
        pacing: Client-side request pacing options.
        futures: Futures contract-selection policy.
    """

    connection: Mapping[str, Any]
    logging: Mapping[str, Any]
    storage: StorageSettings
    download: Mapping[str, Any]
    pacing: Mapping[str, Any]
    futures: Mapping[str, Any]
