"""
Numba-optimised core engine for the vector backtester.

Two implementations are provided:
- ``_perf_engine``        – Numba JIT version (production)
- ``_perf_engine_python`` – Pure-Python reference (testing / debugging)

Both must produce bit-identical results.

Bar PnL is computed using pure mark-to-market:
- Open bar:    current_position * (bar_price - open_price_magnitude)
- Holding bar: current_position * (bar_price - prev_bar_price)
- Close bar:   current_position_before_close * (exit_price_magnitude - prev_bar_price)

This guarantees that sum(bar_net_pnl) == sum(trade.gross_pnl - trade.slippage),
i.e. positions PnL and bar-level PnL always reconcile.
"""

import numpy as np
from numba import jit  # type: ignore


@jit(nopython=True)
def _perf_engine(
    bar_price: np.ndarray,
    open_price: np.ndarray,
    close_price: np.ndarray,
    stop_price: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass PnL engine over price bars.

    Uses pure mark-to-market for per-bar PnL so that bar PnL sums equal
    trade PnL sums exactly.

    Args:
        bar_price:   Reference price for every bar.  Shape ``(n,)``.
        open_price:  Signed entry price; non-zero when a position opens.
        close_price: Signed normal exit price (mutually exclusive with stop_price).
        stop_price:  Signed stop/tp/time-stop exit price.
        cost:        Slippage cost per transaction leg (price units).

    Returns:
        bar_log_returns: shape ``(n,)`` – per-bar log return.
        bar_net_pnl:     shape ``(n,)`` – per-bar net dollar PnL (after slippage).
        trade_records:   shape ``(n_trades, 7)`` – columns:
                         entry_bar, exit_bar, entry_price, exit_price,
                         gross_pnl, slippage, direction.
    """
    n = len(bar_price)
    bar_log_returns = np.zeros(n, dtype=np.float64)
    bar_net_pnl = np.zeros(n, dtype=np.float64)

    max_trades = n * 2
    trade_records = np.zeros((max_trades, 7), dtype=np.float64)
    trade_count = 0

    current_position: int = 0
    entry_price: float = 0.0       # signed (positive = long entry, negative = short)
    entry_price_abs: float = 0.0   # unsigned entry price for MtM
    entry_cost: float = 0.0        # slippage paid at entry bar
    entry_bar: int = 0
    prev_bar_price: float = 0.0

    for i in range(n):
        bp = bar_price[i]
        op = open_price[i]
        cp = close_price[i]
        sp = stop_price[i]

        t_count = int(op != 0.0) + int(cp != 0.0) + int(sp != 0.0)
        bar_slippage = t_count * cost
        bar_pnl: float = -bar_slippage  # slippage always costs

        # Determine exit price (close and stop are mutually exclusive per bar)
        exit_price: float = cp if cp != 0.0 else (sp if sp != 0.0 else 0.0)
        exit_price_abs: float = abs(exit_price)

        # ------------------------------------------------------------------
        # 1. Close existing position: MtM from prev_bar_price to exit price
        # ------------------------------------------------------------------
        if exit_price != 0.0 and current_position != 0:
            # MtM the position up to exit price
            if prev_bar_price != 0.0:
                bar_pnl += current_position * (exit_price_abs - prev_bar_price)
            else:
                # Exit on very first bar (no previous reference): use entry price
                bar_pnl += current_position * (exit_price_abs - entry_price_abs)

            gross_pnl = -(entry_price + exit_price)
            total_slippage = entry_cost + cost  # entry + this exit leg
            trade_records[trade_count, 0] = float(entry_bar)
            trade_records[trade_count, 1] = float(i)
            trade_records[trade_count, 2] = entry_price
            trade_records[trade_count, 3] = exit_price
            trade_records[trade_count, 4] = gross_pnl
            trade_records[trade_count, 5] = total_slippage
            trade_records[trade_count, 6] = float(current_position)
            trade_count += 1

            current_position = 0
            entry_price = 0.0
            entry_price_abs = 0.0
            entry_cost = 0.0
            prev_bar_price = exit_price_abs  # next bar MtMs from exit price

        # ------------------------------------------------------------------
        # 2. Open new position: MtM from open price to bar_price
        # ------------------------------------------------------------------
        if op != 0.0:
            op_abs = abs(op)
            new_position = 1 if op > 0.0 else -1
            # MtM from entry price to bar close price
            bar_pnl += new_position * (bp - op_abs)

            entry_price = op
            entry_price_abs = op_abs
            entry_bar = i
            entry_cost = cost  # one leg paid at open
            current_position = new_position
            prev_bar_price = bp  # next bar MtMs from this bar_price

        # ------------------------------------------------------------------
        # 3. Holding bar (no transaction): MtM from prev_bar_price to bar_price
        # ------------------------------------------------------------------
        elif current_position != 0 and exit_price == 0.0:
            if prev_bar_price != 0.0:
                bar_pnl += current_position * (bp - prev_bar_price)
            prev_bar_price = bp

        # ------------------------------------------------------------------
        # 4. Log return
        # ------------------------------------------------------------------
        denom = bar_price[i - 1] if i > 0 else bar_price[i]
        bar_log_returns[i] = np.log1p(bar_pnl / denom) if denom != 0.0 else 0.0
        bar_net_pnl[i] = bar_pnl

        # Update prev_bar_price for bars with no open (already set above otherwise)
        if op == 0.0 and exit_price == 0.0 and current_position == 0:
            prev_bar_price = bp

    # Force close any remaining position at the final bar price
    if current_position != 0:
        exit_price_synthetic = -current_position * prev_bar_price
        gross_pnl = -(entry_price + exit_price_synthetic)
        total_slippage = entry_cost + cost
        trade_records[trade_count, 0] = float(entry_bar)
        trade_records[trade_count, 1] = float(n - 1)
        trade_records[trade_count, 2] = entry_price
        trade_records[trade_count, 3] = exit_price_synthetic
        trade_records[trade_count, 4] = gross_pnl
        trade_records[trade_count, 5] = total_slippage
        trade_records[trade_count, 6] = float(current_position)
        trade_count += 1
        
        # Adjust last bar PnL to include the closing slippage
        bar_net_pnl[n - 1] -= cost
        # The MtM for the last bar is already computed correctly up to prev_bar_price (which is the last bar_price)
        # We just need to deduct the final slippage from the log return as well
        denom = bar_price[n - 2] if n >= 2 and bar_price[n - 2] != 0.0 else bar_price[n - 1]
        if denom != 0.0:
            # We subtract cost from the raw PnL and recompute
            bar_pnl_last = bar_net_pnl[n - 1]
            bar_log_returns[n - 1] = np.log1p(bar_pnl_last / denom)

    trade_records = trade_records[:trade_count]
    return bar_log_returns, bar_net_pnl, trade_records


def _perf_engine_python(
    bar_price: np.ndarray,
    open_price: np.ndarray,
    close_price: np.ndarray,
    stop_price: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python reference implementation of :func:`_perf_engine`.

    Produces identical results to the Numba version.
    """
    n = len(bar_price)
    bar_log_returns = np.zeros(n, dtype=np.float64)
    bar_net_pnl = np.zeros(n, dtype=np.float64)
    trade_rows: list[list[float]] = []

    current_position: int = 0
    entry_price: float = 0.0
    entry_price_abs: float = 0.0
    entry_cost: float = 0.0
    entry_bar: int = 0
    prev_bar_price: float = 0.0

    for i in range(n):
        bp = float(bar_price[i])
        op = float(open_price[i])
        cp = float(close_price[i])
        sp = float(stop_price[i])

        t_count = int(op != 0.0) + int(cp != 0.0) + int(sp != 0.0)
        bar_slippage = t_count * cost
        bar_pnl: float = -bar_slippage

        exit_price: float = cp if cp != 0.0 else (sp if sp != 0.0 else 0.0)
        exit_price_abs: float = abs(exit_price)

        if exit_price != 0.0 and current_position != 0:
            if prev_bar_price != 0.0:
                bar_pnl += current_position * (exit_price_abs - prev_bar_price)
            else:
                bar_pnl += current_position * (exit_price_abs - entry_price_abs)

            gross_pnl = -(entry_price + exit_price)
            total_slippage = entry_cost + cost
            trade_rows.append([
                float(entry_bar), float(i), entry_price, exit_price,
                gross_pnl, total_slippage, float(current_position),
            ])

            current_position = 0
            entry_price = 0.0
            entry_price_abs = 0.0
            entry_cost = 0.0
            prev_bar_price = exit_price_abs

        if op != 0.0:
            op_abs = abs(op)
            new_position = 1 if op > 0.0 else -1
            bar_pnl += new_position * (bp - op_abs)
            entry_price = op
            entry_price_abs = op_abs
            entry_bar = i
            entry_cost = cost
            current_position = new_position
            prev_bar_price = bp
        elif current_position != 0 and exit_price == 0.0:
            if prev_bar_price != 0.0:
                bar_pnl += current_position * (bp - prev_bar_price)
            prev_bar_price = bp

        denom = bar_price[i - 1] if i > 0 else bar_price[i]
        bar_log_returns[i] = np.log1p(bar_pnl / denom) if denom != 0.0 else 0.0
        bar_net_pnl[i] = bar_pnl

        if op == 0.0 and exit_price == 0.0 and current_position == 0:
            prev_bar_price = bp

    if current_position != 0:
        exit_price_synthetic = -current_position * prev_bar_price
        gross_pnl = -(entry_price + exit_price_synthetic)
        total_slippage = entry_cost + cost
        trade_rows.append([
            float(entry_bar), float(n - 1), entry_price, exit_price_synthetic,
            gross_pnl, total_slippage, float(current_position),
        ])
        bar_net_pnl[n - 1] -= cost
        denom = bar_price[n - 2] if n >= 2 and bar_price[n - 2] != 0.0 else bar_price[n - 1]
        if denom != 0.0:
            bar_pnl_last = bar_net_pnl[n - 1]
            bar_log_returns[n - 1] = np.log1p(bar_pnl_last / denom)

    trade_records = (
        np.array(trade_rows, dtype=np.float64)
        if trade_rows
        else np.zeros((0, 7), dtype=np.float64)
    )
    return bar_log_returns, bar_net_pnl, trade_records
