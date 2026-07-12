"""Evaluate pandas-based strategies and inspect their performance.

Use :func:`no_stop` followed by :func:`perf` for ordinary position or event
strategies. Use :func:`haymaker.research.stop.stop_loss` before :func:`perf`
when exact stop, take-profit, or scheduled-close prices are needed.

``perf()`` uses the first instrument price as initial capital unless the caller
passes ``capital`` explicitly. Standard session returns use beginning account
equity. ``fixed_return`` uses unchanged initial capital and supports a
non-compounding view of the same one-unit PnL stream. Both capital assumptions
must use the same units as PnL.

Sunday observations are combined with the following Monday by default. Pass
``sunday_to_monday=False`` for markets where Sunday is an independent reporting
day. This is a reporting convention rather than an exchange calendar.

Example:
    Convert a position strategy and evaluate it with an explicit capital base::

        from haymaker.research.backtester import no_stop, perf

        transactions = no_stop(strategy, price_column="open")
        result = perf(
            transactions,
            slippage=1.5,
            min_tick=0.25,
            capital=5_000,
        )

See the research backtester user guide for result columns, metric definitions,
warnings, and examples of stock and futures capital assumptions.
"""

from ..result_analysis import (
    excursions,
    factor_extractor,
    winning_trade_adverse_excursions,
)
from .pipeline import (
    Results,
    _TransactionFrame,
    auto_perf,
    get_min_tick,
    no_stop,
    perf,
)

__all__ = [
    "Results",
    "_TransactionFrame",
    "auto_perf",
    "excursions",
    "factor_extractor",
    "get_min_tick",
    "no_stop",
    "perf",
    "winning_trade_adverse_excursions",
]
