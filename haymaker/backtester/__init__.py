"""Experimental event-driven simulation adapters for Haymaker strategies.

This package is intentionally separate from :mod:`haymaker.research.backtester`.
It replays saved broker bars through a live-style component graph and remains
an incomplete experimental extra.
"""

from .broker import (
    BacktestBrokerError,
    SimulatedIB,
    UnsupportedBacktestOrderError,
)
from .controller import SimulationController
from .data import (
    AmbiguousBacktestDataError,
    BacktestDataError,
    BacktestDataNotLoadedError,
    BacktestDataRepository,
    BacktestDataStore,
    InvalidBacktestDataError,
    MissingBacktestDataError,
    MissingContractMetadataError,
    ReplayTimestamp,
    StoredSeries,
    load_backtest_data,
)
from .engine import Backtester, StrategyFactory
from .exceptions import (
    BacktestConfigurationError,
    BacktestError,
    BacktestStrategyError,
    UnsupportedBacktestFeatureError,
)
from .results import BacktestFill, BacktestOrder, BacktestResult, ContractResult

__all__ = [
    "AmbiguousBacktestDataError",
    "BacktestBrokerError",
    "BacktestConfigurationError",
    "BacktestDataError",
    "BacktestDataNotLoadedError",
    "BacktestDataRepository",
    "BacktestDataStore",
    "BacktestError",
    "BacktestFill",
    "BacktestOrder",
    "BacktestResult",
    "BacktestStrategyError",
    "Backtester",
    "ContractResult",
    "InvalidBacktestDataError",
    "MissingBacktestDataError",
    "MissingContractMetadataError",
    "ReplayTimestamp",
    "SimulatedIB",
    "SimulationController",
    "StoredSeries",
    "StrategyFactory",
    "UnsupportedBacktestFeatureError",
    "UnsupportedBacktestOrderError",
    "load_backtest_data",
]
