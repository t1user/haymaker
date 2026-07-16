"""Typed framework settings produced by the configuration loader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from haymaker.supervisor.settings import ConnectionSettings


@dataclass(frozen=True)
class StartupSettings:
    """Operational behavior requested for one live startup."""

    cold_start: bool = False
    reset: bool = False
    zero: bool = False
    nuke: bool = False


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration file and output policy."""

    config_file: Path | None = None
    directory: str = "logs"
    log_broker: bool = False


@dataclass(frozen=True)
class ControllerSettings:
    """Validated settings used to construct the live controller."""

    log_order_events: bool = False
    sync_frequency: int = 0
    health_check_frequency: int = 0
    execution_verification_delay: int = 0
    execution_verification_max_retries: int = 5
    broker_request_timeout: int = 10
    sync_max_attempts: int = 3
    sync_resync_delay: float = 1
    cancel_unknown_trades: bool = False
    missing_brackets: Literal["ignore", "warn", "remove"] = "ignore"
    ignore_errors: tuple[int, ...] = ()
    future_roll_time: tuple[int, int] | None = None


@dataclass(frozen=True)
class StateMachineSettings:
    """Persistence and rejection settings for the live state machine."""

    save_delay: float = 1
    strategy_collection_name: str = "strategies"
    order_collection_name: str = "orders"
    max_rejected_orders: int = 3


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
class SaverSettings:
    """Safe built-in saver specification."""

    type: Literal["csv", "mongo"]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class BlotterSettings:
    """Blotter enablement and persistence settings."""

    enabled: bool = True
    saver: SaverSettings | None = None


@dataclass(frozen=True)
class OrderDefaults:
    """IB order defaults shared by execution models."""

    open: Mapping[str, Any] = field(default_factory=dict)
    close: Mapping[str, Any] = field(default_factory=dict)
    stop: Mapping[str, Any] = field(default_factory=dict)
    take_profit: Mapping[str, Any] = field(default_factory=dict)
    oca_type: int = 1


@dataclass(frozen=True)
class TimeoutPolicy:
    """Default stale-data timeout interval and action."""

    seconds: float = 0
    action: Literal["restart", "log"] = "restart"

    @property
    def log_only(self) -> bool:
        """Return whether a timeout should only be logged."""

        return self.action == "log"


@dataclass(frozen=True)
class FuturesSettings:
    """Live futures roll-selection offsets."""

    roll_bdays: int = 3
    roll_margin_bdays: int = 3


@dataclass(frozen=True)
class LiveSettings:
    """Complete validated settings for one live process."""

    startup: StartupSettings
    connection: ConnectionSettings
    logging: LoggingSettings
    controller: ControllerSettings
    state_machine: StateMachineSettings
    storage: StorageSettings
    blotter: BlotterSettings
    orders: OrderDefaults
    timeout: TimeoutPolicy
    futures: FuturesSettings


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
