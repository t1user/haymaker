"""
Refactored vector backtester.

Public interface
----------------
``no_stop(df, price_column)``
    Convert a DataFrame containing ``position`` or ``blip`` into a
    :class:`TransactionFrame`-compatible DataFrame ready for :func:`perf`.
    This is meant as an alternative to
    :func:`~haymaker.research.stop.interface.stop_loss`
    for strategies that don't use stop-loss.

``perf(data, slippage, skip_last_open, raise_exceptions)``
    Main entry point.  Accepts the output of :func:`no_stop` or
    :func:`~haymaker.research.stop.interface.stop_loss`.

``Results``
    NamedTuple returned by :func:`perf`.

Design
------
The old ``backtester/vector_backtester.py`` is untouched.  Once this module is verified
correct, ``__init__.py`` can be updated to expose this ``perf`` as the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd
from pyfolio.timeseries import perf_stats  # type: ignore

from haymaker.research.signal_converters import blip_sig

from .engine import _perf_engine, _perf_engine_python

# ---------------------------------------------------------------------------
# Public NamedTuples
# ---------------------------------------------------------------------------


class Results(NamedTuple):
    """Output of :func:`perf`."""

    stats: pd.Series
    daily: pd.DataFrame
    positions: pd.DataFrame
    df: pd.DataFrame
    warnings: list[str]


# ---------------------------------------------------------------------------
# Internal validator
# ---------------------------------------------------------------------------


@dataclass
class _TransactionFrame:
    """Internal validator for the DataFrame passed to :func:`perf`.

    Not part of the public API.  Researchers should use :func:`no_stop` or
    :func:`~haymaker.research.stop.interface.stop_loss` to produce valid input.
    """

    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"bar_price", "open_price", "close_price", "stop_price", "position"}
    )

    data: pd.DataFrame

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self.data.columns)
        if missing:
            raise ValueError(
                f"perf() received a DataFrame missing required columns: {missing}. "
                "Use no_stop() or stop_loss() to produce the input."
            )


# ---------------------------------------------------------------------------
# Factory 1: simple path (no stop-loss)
# ---------------------------------------------------------------------------


def no_stop(
    df: pd.DataFrame,
    price_column: str = "open",
) -> pd.DataFrame:
    """Convert researcher signals into a transaction DataFrame for :func:`perf`.

    Accepts a DataFrame with either a ``position`` column or a ``blip`` column
    (and optionally a ``close_blip`` column).

    Dispatch precedence:
    1. ``position`` column – used directly.
    2. ``blip`` column    – shifted by 1 bar then converted to a position series.
    3. Neither present    – raises :exc:`ValueError`.

    Args:
        df:           Input DataFrame.  Must contain ``price_column`` and either
                      ``position`` or ``blip``.
        price_column: Column used for execution prices and mark-to-market.

    Returns:
        DataFrame with columns: ``bar_price``, ``position``, ``open_price``,
        ``close_price``, ``stop_price``.
    """
    if price_column not in df.columns:
        raise ValueError(
            f"'{price_column}' indicated as price_column but not found in df"
        )

    price = df[price_column]

    # ------------------------------------------------------------------
    # Derive position series
    # ------------------------------------------------------------------
    if "position" in df.columns:
        position: pd.Series = df["position"].astype(int)
    elif "blip" in df.columns:
        # Blips are generated at bar N; execution is at bar N+1.
        shifted_blip = df["blip"].shift().fillna(0).astype(int)
        if "close_blip" in df.columns:
            shifted_close = df["close_blip"].shift().fillna(0).astype(int)
            blip_df = pd.DataFrame({"blip": shifted_blip, "close_blip": shifted_close})
            position = blip_sig(blip_df, always_on=True)
        else:
            position = blip_sig(shifted_blip, always_on=True)
        position = position.astype(int)
    else:
        raise ValueError("df must have either a 'position' column or a 'blip' column")

    # ------------------------------------------------------------------
    # Derive per-bar transaction prices using fast Numpy operations
    # ------------------------------------------------------------------
    price_arr = price.to_numpy(dtype=np.float64, copy=False)
    pos_arr = position.to_numpy(dtype=np.int8, copy=False)

    # We can shift using numpy
    prev_pos_arr = np.empty_like(pos_arr)
    prev_pos_arr[0] = 0
    prev_pos_arr[1:] = pos_arr[:-1]

    changed = pos_arr != prev_pos_arr
    opening = changed & (pos_arr != 0)
    closing = changed & (pos_arr == 0)
    reversal = changed & (pos_arr != 0) & (prev_pos_arr != 0)

    open_price = np.where(opening, price_arr * pos_arr, 0.0)
    close_price = np.where(closing | reversal, price_arr * -prev_pos_arr, 0.0)
    stop_price = np.zeros(len(df), dtype=np.float64)

    out = pd.DataFrame(
        {
            "bar_price": price_arr,
            "position": pos_arr,
            "open_price": open_price,
            "close_price": close_price,
            "stop_price": stop_price,
        },
        index=df.index,
    )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def get_min_tick(data: pd.Series) -> float:
    """Estimate minimum tick from price data using fast numpy ops."""
    arr = np.sort(data.to_numpy(dtype=np.float64, copy=False))
    diffs = np.abs(np.diff(arr))
    diffs = diffs[diffs > 1e-10]
    if len(diffs) == 0:
        return 0.0
    diffs_rounded = np.round(diffs, 8)
    vals, counts = np.unique(diffs_rounded, return_counts=True)
    return float(vals[np.argmax(counts)])


def _daily_returns(lreturn: pd.Series) -> pd.DataFrame:
    """Resample bar log-returns to business-daily simple returns."""
    daily = pd.DataFrame()
    daily["lreturn"] = lreturn.resample("B").sum().dropna()
    daily["returns"] = np.exp(daily["lreturn"]) - 1
    daily["balance"] = (daily["returns"] + 1).cumprod()
    return daily


# ---------------------------------------------------------------------------
# Internal calculator
# ---------------------------------------------------------------------------


class _PerfCalculator:
    """Internal computation class for orchestrating the :func:`perf` pipeline.

    This class prepares NumPy arrays from the ``TransactionFrame``, dispatches
    them to the appropriate execution engine, and formats the raw trade records
    and array responses back into Pyfolio-compatible DataFrames and summary stats.

    Not part of the public API.
    """

    def __init__(
        self,
        tx: _TransactionFrame,
        slippage: float,
        skip_last_open: bool,
        raise_exceptions: bool,
        use_numba: bool,
    ) -> None:
        self._tx = tx
        self._slippage = slippage
        self._skip_last_open = skip_last_open
        self._raise_exceptions = raise_exceptions
        self._use_numba = use_numba
        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: arrays
    # ------------------------------------------------------------------

    def _prepare_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Extract numpy arrays and compute cost from the transaction frame."""
        data = self._tx.data
        if self._slippage == 0.0:
            cost = 0.0
        else:
            min_tick = get_min_tick(data["bar_price"])
            cost = min_tick * self._slippage
        return (
            data["bar_price"].to_numpy(dtype=np.float64, copy=False),
            data["open_price"].to_numpy(dtype=np.float64, copy=False),
            data["close_price"].to_numpy(dtype=np.float64, copy=False),
            data["stop_price"].to_numpy(dtype=np.float64, copy=False),
            cost,
        )

    # ------------------------------------------------------------------
    # Step 2: engine
    # ------------------------------------------------------------------

    def _run_engine(
        self,
        bar_price: np.ndarray,
        open_price: np.ndarray,
        close_price: np.ndarray,
        stop_price: np.ndarray,
        cost: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch to Numba or Python engine.

        Returns (bar_log_returns, bar_net_pnl, trade_records).
        """
        engine = _perf_engine if self._use_numba else _perf_engine_python
        return engine(bar_price, open_price, close_price, stop_price, cost)

    # ------------------------------------------------------------------
    # Step 3: positions DataFrame from trade records
    # ------------------------------------------------------------------

    def _build_positions(
        self,
        trade_records: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        """Convert raw trade records to a per-position DataFrame.

        trade_records columns (7):
            entry_bar, exit_bar, entry_price, exit_price,
            gross_pnl, slippage (total round-trip), direction.

        Output columns: ``date_o``, ``open``, ``date_c``, ``close``,
        ``g_pnl``, ``pnl``, ``duration``.
        """
        if len(trade_records) == 0:
            return pd.DataFrame(
                columns=[
                    "date_o",
                    "open",
                    "date_c",
                    "close",
                    "g_pnl",
                    "pnl",
                    "duration",
                ]
            )

        entry_bars = trade_records[:, 0].astype(int)
        exit_bars = trade_records[:, 1].astype(int)
        entry_prices = trade_records[:, 2]
        exit_prices = trade_records[:, 3]
        gross_pnls = trade_records[:, 4]
        slippages = trade_records[:, 5]  # total round-trip slippage

        positions = pd.DataFrame(
            {
                "date_o": index[entry_bars],
                "open": entry_prices,
                "date_c": index[exit_bars],
                "close": exit_prices,
                "g_pnl": gross_pnls,
                "pnl": gross_pnls - slippages,
                "duration": index[exit_bars] - index[entry_bars],
            }
        ).reset_index(drop=True)

        return positions

    # ------------------------------------------------------------------
    # Step 4: bar-level df (for debugging)
    # ------------------------------------------------------------------

    def _build_bar_df(
        self,
        bar_log_returns: np.ndarray,
        bar_net_pnl: np.ndarray,
        cost: float,
    ) -> pd.DataFrame:
        """Assemble a bar-level DataFrame.

        Uses the engine's bar_net_pnl directly (already mark-to-market minus
        slippage) so that df.pnl.sum() == positions.pnl.sum() by construction.
        """
        data = self._tx.data
        df = data.copy()
        df["lreturn"] = bar_log_returns
        t_count = (
            (df["open_price"] != 0).astype(int)
            + (df["close_price"] != 0).astype(int)
            + (df["stop_price"] != 0).astype(int)
        )
        df["slippage"] = t_count * cost
        df["pnl"] = bar_net_pnl
        df["g_pnl"] = df["pnl"] + df["slippage"]
        return df

    # ------------------------------------------------------------------
    # Step 5: stats
    # ------------------------------------------------------------------

    def _build_stats(
        self,
        positions: pd.DataFrame,
        daily: pd.DataFrame,
        bar_df: pd.DataFrame,
        min_tick: float,
    ) -> pd.Series:
        """Compute all statistics."""
        warnings = self._warnings
        stats: pd.Series = pd.Series(dtype="O")

        win_pos = positions[positions["pnl"] > 0]
        loss_pos = positions[positions["pnl"] <= 0]

        try:
            num_pos = len(positions)
            days = daily.returns.count()

            avg_gain = win_pos.pnl.mean() if len(win_pos) else 0.0
            avg_loss = loss_pos.pnl.mean() if len(loss_pos) else 0.0
            stats["Payoff ratio"] = (
                abs(avg_gain / avg_loss) if avg_loss != 0 else float("nan")
            )
            stats["Profit factor"] = (
                abs(win_pos.pnl.sum() / loss_pos.pnl.sum())
                if loss_pos.pnl.sum() != 0
                else float("nan")
            )
            stats["Win ratio"] = len(win_pos) / num_pos if num_pos else float("nan")
            stats["Average gain"] = avg_gain
            stats["Average loss"] = avg_loss
            stats["Position EV"] = positions.pnl.mean() if num_pos else float("nan")
            stats["Position EV in ticks"] = stats["Position EV"] / min_tick

            stats["Long EV"] = positions[positions["open"] > 0].pnl.mean()
            stats["Short EV"] = positions[positions["open"] < 0].pnl.mean()

            sorted_gains = positions["pnl"].sort_values().values
            if len(sorted_gains) > 0:
                total_pnl = positions["pnl"].sum()
                stats["Best trade"] = sorted_gains[-1]
                stats["Worst trade"] = sorted_gains[0]
                stats["Best trade as % of pnl"] = (
                    abs(sorted_gains[-1] / total_pnl) if total_pnl else float("nan")
                )
                stats["Worst trade as % of pnl"] = (
                    abs(sorted_gains[0] / total_pnl) if total_pnl else float("nan")
                )

            stats["Positions per day"] = num_pos / days if days else float("nan")
            stats["Days per position"] = days / num_pos if num_pos else float("nan")

            duration = positions["duration"].mean()
            stats["Actual avg. duration"] = (
                duration.round("min")
                if isinstance(duration, pd.Timedelta)
                else duration
            )

            stats["Days"] = days
            stats["Positions"] = num_pos
            stats["Monthly EV"] = stats["Positions per day"] * stats["Position EV"] * 21
            stats["Annual EV"] = 12 * stats["Monthly EV"]
        except (KeyError, ValueError, ZeroDivisionError) as e:
            warnings.append(str(e))

        # Pyfolio stats
        pyfolio_stats = perf_stats(daily["returns"])

        price = bar_df["bar_price"]
        bar_pnl_sum = bar_df["pnl"].sum()
        year_frac = (bar_df.index[-1] - bar_df.index[0]) / pd.Timedelta(days=365)

        stats["Simple annual return"] = (
            (bar_pnl_sum / price.iloc[0]) / year_frac if year_frac else float("nan")
        )
        stats["Sum of returns annualised"] = (
            daily["returns"].sum() / year_frac if year_frac else float("nan")
        )

        def efficiency(strategy_return: float) -> float:
            mkt = abs(price.iloc[-1] / price.iloc[0] - 1)
            return strategy_return / mkt if mkt else float("nan")

        def efficiency_1(strategy_return: float) -> float:
            # Fixed from max/min to max/min - 1 (perfect buy-low sell-high return)
            mkt = abs(price.max() / price.min() - 1)
            return strategy_return / mkt if mkt else float("nan")

        def efficiency_2(strategy_return: float) -> float:
            mkt = price.pct_change().abs().sum()
            return strategy_return / mkt if mkt else float("nan")

        stats["Efficiency"] = efficiency(pyfolio_stats["Cumulative returns"])
        stats["Efficiency (Simple)"] = efficiency(bar_pnl_sum / price.iloc[0])
        stats["Efficiency_1 (Max/Min)"] = efficiency_1(
            pyfolio_stats["Cumulative returns"]
        )
        stats["Efficiency_2 (Path Length)"] = efficiency_2(
            pyfolio_stats["Cumulative returns"]
        )

        stats["Position Sharpe"] = (
            stats["Position EV"] / positions.pnl.std()
            if positions.pnl.std() != 0
            else float("nan")
        )
        stats["Annualized position Sharpe"] = stats["Position Sharpe"] * np.sqrt(
            stats["Positions per day"] * 252
        )

        return pd.concat([pyfolio_stats, stats])

    # ------------------------------------------------------------------
    # Duration warning
    # ------------------------------------------------------------------

    def _duration_warning(self, positions: pd.DataFrame, bar_df: pd.DataFrame) -> None:
        """Append duration warnings to self._warnings."""
        if len(positions) == 0:
            return
        indices = pd.DataFrame(
            {"n": np.arange(1, len(bar_df.index) + 1)}, index=bar_df.index
        )
        locs = positions[["date_o", "date_c"]].copy()
        locs["i_o"] = indices.loc[locs["date_o"], "n"].to_numpy()
        locs["i_c"] = indices.loc[locs["date_c"], "n"].to_numpy()
        locs["dur"] = locs["i_c"] - locs["i_o"]
        total = len(locs)
        for d in (0, 1):
            frac = (locs["dur"] == d).sum() / total
            if frac > 0.05:
                self._warnings.append(
                    f"Warning: {frac:.1%} positions with duration of {d} candle."
                )

    # ------------------------------------------------------------------
    # Last-open warning
    # ------------------------------------------------------------------

    def _last_open_warning(self, positions: pd.DataFrame) -> None:
        if len(positions) == 0:
            self._warnings.append("No positions")
            return
        total_pnl = positions.pnl.sum()
        if total_pnl == 0:
            return
        pnl_frac = abs(positions.iloc[-1].pnl / total_pnl)
        if pnl_frac > 0.3:
            durations = positions["date_c"] - positions["date_o"]
            avg = durations.iloc[:-1].mean()
            last = durations.iloc[-1]
            mean_time_frac = (
                last / avg if pd.Timedelta(avg).total_seconds() > 0 else float("nan")
            )
            self._warnings.append(
                f"Warning: last open position represents {pnl_frac:.1%} of total pnl "
                f"and is {mean_time_frac:.1f}x average duration ({last} vs {avg})"
            )

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def run(self) -> Results:
        """Execute the complete performance evaluation pipeline.

        Returns:
            Results: A named tuple containing stats, daily returns, positions,
            the modified bar DataFrame, and any raised warnings.
        """
        data = self._tx.data

        if len(data[data["position"] != 0]) == 0:
            self._warnings.append("No positions")

        bar_price, open_price, close_price, stop_price, cost = self._prepare_arrays()
        min_tick = cost / self._slippage if self._slippage else 1.0

        bar_log_returns, bar_net_pnl, trade_records = self._run_engine(
            bar_price, open_price, close_price, stop_price, cost
        )

        positions = self._build_positions(trade_records, data.index)

        bar_df = self._build_bar_df(bar_log_returns, bar_net_pnl, cost)

        # skip_last_open: if the last trade's exit_bar == last bar index and
        # position was still open, exclude it from stats.
        if self._skip_last_open and len(positions) > 0:
            last_exit = data.index[int(trade_records[-1, 1])]
            last_bar = data.index[-1]
            if last_exit == last_bar and data["position"].iloc[-1] != 0:
                positions = positions.iloc[:-1]
                if len(positions) > 0:
                    bar_df = bar_df.loc[: positions["date_c"].iloc[-1]]
                else:
                    bar_df = bar_df.iloc[:0]
        else:
            self._last_open_warning(positions)

        daily = _daily_returns(pd.Series(bar_log_returns, index=data.index))

        stats = self._build_stats(positions, daily, bar_df, min_tick)

        self._duration_warning(positions, bar_df)

        return Results(
            stats=stats,
            daily=daily,
            positions=positions,
            df=bar_df,
            warnings=self._warnings,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def perf(
    data: pd.DataFrame,
    slippage: float = 1.5,
    skip_last_open: bool = False,
    raise_exceptions: bool = True,
    use_numba: bool = True,
) -> Results:
    """Return performance statistics for a strategy.

    Args:
        data:             Output of :func:`no_stop` or
                          :func:`~haymaker.research.stop.interface.stop_loss`
                          with ``bar_price`` added.  Must have columns:
                          ``bar_price``, ``open_price``, ``close_price``,
                          ``stop_price``, ``position``.
        slippage:         Transaction cost expressed as a multiple of min tick.
        skip_last_open:   If ``True``, exclude the last trade if the position
                          was still open at the final bar (force-closed at
                          bar_price).
        raise_exceptions: If ``False``, catch internal assertion errors and
                          append them to ``Results.warnings`` instead.
        use_numba:        If ``True`` (default), use the Numba JIT engine.
                          Set to ``False`` to use the reference Python engine.

    Returns:
        :class:`Results` NamedTuple.
    """
    tx = _TransactionFrame(data)
    return _PerfCalculator(
        tx, slippage, skip_last_open, raise_exceptions, use_numba
    ).run()
