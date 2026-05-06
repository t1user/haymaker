"""
Backtester Package
==================

This package provides a highly optimized, single-pass Numba execution
engine for evaluating algorithmic trading strategy performance over
vectorized price bars.

Rationale
---------

The purpose of this engine is to serve as the research bridge between
signal generation (typically developed in pandas) and rigorous
performance evaluation.

Historically, strategies were evaluated via split logic paths
(position-based vs blip-based) and occasionally suffered from PnL
reconciliation issues (where the sum of daily returns did not
perfectly match the sum of trade-level PnL).

This refactored package solves those issues by:

    1. **Unified Pipeline**: Standardizing on a `TransactionFrame`
       schema to guarantee that price and transaction inputs are
       strictly typed and structurally sound before evaluation.

    2. **Mark-to-Market Accounting**: Using a pure mark-to-market
       engine that calculates PnL on every bar.  This guarantees that
       `sum(bar_net_pnl)` exactly equals `sum(trade_pnl)`.

    3. **High Performance**: Bypassing Pandas row-by-row iteration
       overhead by implementing the core engine in Numba, yielding a
       ~5x speedup for raw calculations and ~20% end-to-end.

Usage
-----

    1. Standard Signals (Positions): If you have a Pandas DataFrame
       with a price column and a position column:

from haymaker.research.backtester import no_stop, perf

# Convert raw dataframe into a validated TransactionFrame tx =
no_stop(df, price_column="open")

# Evaluate the strategy results = perf(tx, slippage=1.5)

    2. Stop-Loss Integration: If you use the stop-loss execution
       engine, it will return a `TransactionFrame` directly:

from haymaker.research.stop import stop_loss from
haymaker.research.backtester import perf

tx = stop_loss(...) results = perf(tx, slippage=1.5)

Assumptions
-----------

    - **Slippage**: Slippage is symmetric and applied per-leg (opening
      costs `cost`, closing costs `cost`).

    - **Return Denominator**: Returns are computed unconditionally
      against the prior bar's price (`price[i-1]`) to properly align
      with standard continuous financial return math.

    - **Minimum Tick**: The minimum tick size is automatically
      inferred from the price array (unless `slippage` is 0, in which
      case the expensive inference is skipped).

    - **Final Positions**: Open positions left at the end of the
      simulation are synthetically "force-closed" at the final bar's
      price to ensure accurate trailing PnL, unless explicitly
      suppressed via the `skip_last_open` flag.

Modules
-------

    - `pipeline`: The user-facing API (`perf`, `no_stop`) and Pyfolio
      stats construction.

    - `engine`: The low-level Numba JIT core and Python fallback
      engines.

    - `utilities`: Legacy research utilities that most likely need to
      be reviewed
"""

from .pipeline import (
    Results,
    _TransactionFrame,
    get_min_tick,
    no_stop,
    perf,
)
from .utilities import (
    adverse_excursions,
    blip_extractor,
    excursions,
    factor_extractor,
    profitable_excursions,
    summary,
    v_backtester,
)

__all__ = [
    "Results",
    "_TransactionFrame",
    "adverse_excursions",
    "blip_extractor",
    "excursions",
    "factor_extractor",
    "get_min_tick",
    "no_stop",
    "perf",
    "profitable_excursions",
    "summary",
    "v_backtester",
]
