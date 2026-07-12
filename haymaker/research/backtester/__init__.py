"""Vector backtesting from prepared pandas strategy data.

The package separates execution accounting from performance reporting:

* :func:`no_stop` prepares ordinary position or event strategies.
* :func:`perf` runs the Numba or Python mark-to-market engine and reports
  account, fixed-capital, PnL, trade, and drawdown statistics.
* :mod:`haymaker.research.backtester.metrics` derives returns only after
  reconciled bar PnL has been grouped into observed reporting dates.

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

The active implementation is in ``pipeline.py``, ``engine.py``, and
``metrics.py``. Historical code under ``retired`` is non-public reference code.
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
