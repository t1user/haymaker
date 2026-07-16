"""Configuration aggregates retained by the live and dataloader loaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from haymaker.supervisor.settings import ConnectionSettings


@dataclass(frozen=True)
class LoggingSettings:
    """Dataloader logging configuration retained during staged migration."""

    config_file: Path | None = None
    directory: str = "logs"
    log_broker: bool = False


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
    """Merged live framework configuration grouped by original section.

    Attributes:
        startup: One-run controller startup requests.
        connection: Broker connection and recovery options.
        logging: Logging setup and broker-log options.
        controller: Controller reconciliation and scheduling options.
        state_machine: State persistence and rejection options.
        storage: Retained typed storage settings pending datastore refactoring.
        blotter: Blotter enablement and saver specification.
        orders: Default IB order fields.
        timeout: Default streamer timeout policy.
        futures: Live futures selection offsets.
    """

    startup: Mapping[str, Any]
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
class DownloadSettings:
    """Historical download planning and worker settings."""

    source: str = "contracts.csv"
    bar_size: str = "30 secs"
    what_to_show: str = "TRADES"
    max_lookback_days: int | None = None
    gap_fill_mode: Literal["off", "heuristic", "schedule", "auto"] = "off"
    use_rth: bool = False
    save_every_chunks: int = 10
    number_of_workers: int = 10


@dataclass(frozen=True)
class PacingSettings:
    """Client-side historical-request pacing settings."""

    no_restriction: bool = False
    allowance_fraction: float = 1.0


@dataclass(frozen=True)
class DataloaderFuturesSettings:
    """Futures expansion policy for dataloader contract rows."""

    selector: str = "current_and_expired"
    full_chain_spec: str = "full"
    current_index: int = 0


@dataclass(frozen=True)
class DataloaderSettings:
    """Complete validated settings for one dataloader process."""

    connection: ConnectionSettings
    logging: LoggingSettings
    storage: StorageSettings
    download: DownloadSettings
    pacing: PacingSettings
    futures: DataloaderFuturesSettings
