"""Public exceptions raised by the experimental event-driven backtester."""


class BacktestError(RuntimeError):
    """Base exception for simulation configuration and execution failures."""


class BacktestConfigurationError(BacktestError, ValueError):
    """Raised when a simulation or strategy graph cannot be configured safely."""


class BacktestStrategyError(BacktestError):
    """Raised when a strategy callback fails during replay."""


class UnsupportedBacktestFeatureError(BacktestError, NotImplementedError):
    """Raised when a live-only behavior is outside the simulated subset."""


__all__ = [
    "BacktestConfigurationError",
    "BacktestError",
    "BacktestStrategyError",
    "UnsupportedBacktestFeatureError",
]
