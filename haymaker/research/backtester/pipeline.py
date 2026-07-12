"""User-facing preparation and performance pipeline for the vector backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd

from haymaker.research.signal_converters import blip_sig, sig_pos

from .engine import _perf_engine, _perf_engine_python
from .metrics import build_performance_frames, build_stats


class Results(NamedTuple):
    """Strategy performance returned by :func:`perf` and :func:`auto_perf`.

    Attributes:
        stats: Account, fixed-capital, PnL, and trade statistics indexed by
            compact ``snake_case`` names.
        daily: Reporting-day PnL, account returns, fixed returns, equity, and
            normalized balance.
        positions: One row per completed trade, with entry/exit timestamps,
            signed prices, gross/net PnL, and duration.
        df: Prepared input bars enriched with slippage, gross/net PnL, equity,
            balance, and drawdown paths.
        warnings: Non-fatal conditions affecting metric availability or
            interpretation. Invalid input and accounting failures raise instead.
    """

    stats: pd.Series
    daily: pd.DataFrame
    positions: pd.DataFrame
    df: pd.DataFrame
    warnings: list[str]


@dataclass
class _TransactionFrame:
    """Validate the transaction DataFrame accepted by :func:`perf`."""

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
        if self.data.empty:
            raise ValueError("perf() requires at least one bar.")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("perf() requires a DatetimeIndex.")
        if not self.data.index.is_monotonic_increasing:
            raise ValueError("perf() requires a monotonically increasing index.")
        if not self.data.index.is_unique:
            raise ValueError("perf() requires unique bar timestamps.")

        try:
            values = self.data[list(self.REQUIRED_COLUMNS)].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("Transaction-frame columns must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Transaction-frame columns must contain finite values.")
        positions = set(self.data["position"].unique())
        if not positions.issubset({-1, 0, 1}):
            raise ValueError("position values must be -1, 0, or 1.")


def no_stop(
    df: pd.DataFrame,
    price_column: str = "open",
) -> pd.DataFrame:
    """Convert positions or entry events into input accepted by :func:`perf`.

    A literal ``position`` column is used directly. Otherwise, ``blip`` and an
    optional ``close_blip`` are converted into a next-bar position series.
    This helper is appropriate when exact intrabar stop prices are not needed;
    use :func:`haymaker.research.stop.stop_loss` when they are.

    Args:
        df: Bar-indexed market data containing ``price_column`` and either
            ``position`` or ``blip``.
        price_column: Execution and mark-to-market price column.

    Returns:
        Transaction DataFrame containing ``bar_price``, ``position``,
        ``open_price``, ``close_price``, and ``stop_price``.

    Raises:
        ValueError: If the input is empty or required columns are absent.
    """
    if df.empty:
        raise ValueError("df must contain at least one bar.")
    if price_column not in df.columns:
        raise ValueError(
            f"'{price_column}' indicated as price_column but not found in df"
        )

    price = df[price_column]
    if "position" in df.columns:
        position: pd.Series = df["position"].astype(int)
    elif "blip" in df.columns:
        signal = (
            blip_sig(df[["blip", "close_blip"]])
            if "close_blip" in df.columns
            else blip_sig(df["blip"])
        )
        position = sig_pos(signal)
    else:
        raise ValueError("df must have either a 'position' column or a 'blip' column")

    price_arr = price.to_numpy(dtype=np.float64, copy=False)
    pos_arr = position.to_numpy(dtype=np.int8, copy=False)
    prev_pos_arr = np.empty_like(pos_arr)
    prev_pos_arr[0] = 0
    prev_pos_arr[1:] = pos_arr[:-1]

    changed = pos_arr != prev_pos_arr
    opening = changed & (pos_arr != 0)
    closing = changed & (pos_arr == 0)
    reversal = changed & (pos_arr != 0) & (prev_pos_arr != 0)

    return pd.DataFrame(
        {
            "bar_price": price_arr,
            "position": pos_arr,
            "open_price": np.where(opening, price_arr * pos_arr, 0.0),
            "close_price": np.where(closing | reversal, price_arr * -prev_pos_arr, 0.0),
            "stop_price": np.zeros(len(df), dtype=np.float64),
        },
        index=df.index,
    )


def get_min_tick(data: pd.Series) -> float:
    """Estimate the most frequently observed positive price increment.

    Args:
        data: Instrument prices.

    Returns:
        Estimated tick size, or ``0.0`` when it cannot be inferred.
    """
    arr = np.sort(data.to_numpy(dtype=np.float64, copy=False))
    diffs = np.abs(np.diff(arr))
    diffs = diffs[diffs > 1e-10]
    if len(diffs) == 0:
        return 0.0
    values, counts = np.unique(np.round(diffs, 8), return_counts=True)
    return float(values[np.argmax(counts)])


class _PerfCalculator:
    """Orchestrate accounting output and performance reporting."""

    def __init__(
        self,
        tx: _TransactionFrame,
        *,
        slippage: float,
        skip_last_open: bool,
        capital: float | None,
        min_tick: float | None,
        sunday_to_monday: bool,
        use_numba: bool,
    ) -> None:
        self._tx = tx
        self._slippage = self._validate_nonnegative(slippage, "slippage")
        self._skip_last_open = skip_last_open
        self._capital = self._resolve_capital(capital)
        self._min_tick = self._resolve_min_tick(min_tick)
        self._sunday_to_monday = sunday_to_monday
        self._use_numba = use_numba
        self._warnings: list[str] = []

    @staticmethod
    def _validate_nonnegative(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")
        return result

    def _resolve_capital(self, capital: float | None) -> float:
        value = (
            float(self._tx.data["bar_price"].iloc[0])
            if capital is None
            else float(capital)
        )
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "capital must be positive; pass it explicitly when the first "
                "bar price is not a valid capital base."
            )
        return value

    def _resolve_min_tick(self, min_tick: float | None) -> float:
        if min_tick is not None:
            value = float(min_tick)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("min_tick must be a finite positive number.")
            return value
        value = get_min_tick(self._tx.data["bar_price"])
        if value == 0 and self._slippage > 0:
            raise ValueError(
                "Minimum tick could not be inferred; pass min_tick explicitly "
                "when slippage is non-zero."
            )
        return value

    def _prepare_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        data = self._tx.data
        return (
            data["bar_price"].to_numpy(dtype=np.float64, copy=False),
            data["open_price"].to_numpy(dtype=np.float64, copy=False),
            data["close_price"].to_numpy(dtype=np.float64, copy=False),
            data["stop_price"].to_numpy(dtype=np.float64, copy=False),
            self._min_tick * self._slippage,
        )

    def _run_engine(
        self,
        bar_price: np.ndarray,
        open_price: np.ndarray,
        close_price: np.ndarray,
        stop_price: np.ndarray,
        cost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        engine = _perf_engine if self._use_numba else _perf_engine_python
        return engine(bar_price, open_price, close_price, stop_price, cost)

    @staticmethod
    def _build_positions(
        trade_records: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        if len(trade_records) == 0:
            return pd.DataFrame(
                columns=(
                    "date_o",
                    "open",
                    "date_c",
                    "close",
                    "g_pnl",
                    "pnl",
                    "duration",
                )
            )

        entry_bars = trade_records[:, 0].astype(int)
        exit_bars = trade_records[:, 1].astype(int)
        return pd.DataFrame(
            {
                "date_o": index[entry_bars],
                "open": trade_records[:, 2],
                "date_c": index[exit_bars],
                "close": trade_records[:, 3],
                "g_pnl": trade_records[:, 4],
                "pnl": trade_records[:, 4] - trade_records[:, 5],
                "duration": index[exit_bars] - index[entry_bars],
            }
        ).reset_index(drop=True)

    def _build_bar_df(self, bar_net_pnl: np.ndarray, cost: float) -> pd.DataFrame:
        df = self._tx.data.copy()
        transaction_count = (
            df["open_price"].ne(0).astype(int)
            + df["close_price"].ne(0).astype(int)
            + df["stop_price"].ne(0).astype(int)
        )
        if df["position"].iloc[-1] != 0:
            transaction_count.iloc[-1] += 1
        df["slippage"] = transaction_count * cost
        df["pnl"] = bar_net_pnl
        df["g_pnl"] = df["pnl"] + df["slippage"]
        return df

    @staticmethod
    def _assert_reconciliation(
        positions: pd.DataFrame,
        bar_df: pd.DataFrame,
    ) -> None:
        checks = (
            ("net", positions["pnl"].sum(), bar_df["pnl"].sum()),
            ("gross", positions["g_pnl"].sum(), bar_df["g_pnl"].sum()),
        )
        for label, trade_total, bar_total in checks:
            if not np.isclose(trade_total, bar_total, rtol=1e-10, atol=1e-10):
                raise RuntimeError(
                    f"{label} PnL reconciliation failed: "
                    f"trades={trade_total}, bars={bar_total}."
                )

    def _exclude_last_open(
        self,
        positions: pd.DataFrame,
        bar_df: pd.DataFrame,
        trade_records: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = self._tx.data
        if not self._skip_last_open or positions.empty:
            return positions, bar_df
        last_exit = data.index[int(trade_records[-1, 1])]
        if last_exit != data.index[-1] or data["position"].iloc[-1] == 0:
            return positions, bar_df

        positions = positions.iloc[:-1]
        if positions.empty:
            return positions, bar_df.iloc[:0]
        return positions, bar_df.loc[: positions["date_c"].iloc[-1]]

    def _duration_warning(
        self,
        positions: pd.DataFrame,
        bar_df: pd.DataFrame,
    ) -> None:
        if positions.empty or bar_df.empty:
            return
        bar_numbers = pd.Series(np.arange(1, len(bar_df) + 1), index=bar_df.index)
        duration_bars = (
            bar_numbers.loc[positions["date_c"]].to_numpy()
            - bar_numbers.loc[positions["date_o"]].to_numpy()
        )
        for duration in (0, 1):
            fraction = float(np.mean(duration_bars == duration))
            if fraction > 0.05:
                self._warnings.append(
                    f"{fraction:.1%} of trades have a duration of {duration} bar(s)."
                )

    def _last_open_warning(self, positions: pd.DataFrame) -> None:
        if positions.empty:
            return
        total_pnl = float(positions["pnl"].sum())
        if total_pnl == 0:
            return
        pnl_fraction = abs(float(positions.iloc[-1]["pnl"]) / total_pnl)
        if pnl_fraction <= 0.3:
            return

        durations = positions["date_c"] - positions["date_o"]
        last_duration = pd.Timedelta(durations.iloc[-1])
        previous_average = (
            pd.Timedelta(durations.iloc[:-1].mean()) if len(durations) > 1 else pd.NaT
        )
        if pd.isna(previous_average) or previous_average <= pd.Timedelta(0):
            comparison = "without a positive prior average duration"
        else:
            comparison = f"{last_duration / previous_average:.1f}x the prior average"
        self._warnings.append(
            f"The final forced-close trade represents {pnl_fraction:.1%} of "
            f"total PnL and is {comparison}."
        )

    def run(self) -> Results:
        """Execute accounting, reconciliation, and performance reporting."""
        arrays = self._prepare_arrays()
        bar_net_pnl, trade_records = self._run_engine(*arrays)
        positions = self._build_positions(trade_records, self._tx.data.index)
        bar_df = self._build_bar_df(bar_net_pnl, arrays[-1])
        self._assert_reconciliation(positions, bar_df)

        positions, bar_df = self._exclude_last_open(positions, bar_df, trade_records)
        self._assert_reconciliation(positions, bar_df)

        if positions.empty:
            self._warnings.append("No trades.")
        elif not self._skip_last_open:
            self._last_open_warning(positions)

        bar_df, daily = build_performance_frames(
            bar_df,
            capital=self._capital,
            sunday_to_monday=self._sunday_to_monday,
        )
        stats = build_stats(
            positions,
            daily,
            bar_df,
            capital=self._capital,
            min_tick=self._min_tick,
            warnings=self._warnings,
        )
        self._duration_warning(positions, bar_df)
        return Results(stats, daily, positions, bar_df, self._warnings)


def perf(
    data: pd.DataFrame,
    slippage: float = 1.5,
    skip_last_open: bool = False,
    *,
    capital: float | None = None,
    min_tick: float | None = None,
    sunday_to_monday: bool = True,
    use_numba: bool = True,
) -> Results:
    """Evaluate a prepared strategy and return performance analysis.

    Use :func:`no_stop` to prepare an ordinary position or event strategy, or
    :func:`haymaker.research.stop.stop_loss` when exact stop and take-profit
    prices are required. The result always includes two views of the same
    one-unit PnL: conventional account returns based on beginning daily equity
    and non-compounding fixed returns based on unchanged initial capital.

    Args:
        data: Output of :func:`no_stop` or
            :func:`haymaker.research.stop.stop_loss`, indexed by unique,
            increasing timestamps. Required columns are ``bar_price``,
            ``open_price``, ``close_price``, ``stop_price``, and ``position``.
        slippage: Non-negative transaction cost in multiples of ``min_tick``,
            charged for every entry and exit leg.
        skip_last_open: Exclude a position that remains open on the final input
            bar instead of reporting the engine's synthetic final close.
        capital: Positive initial capital in the same units as point PnL.
            Defaults to the first ``bar_price`` for an initially unlevered
            one-unit assumption. For futures, convert a dollar funding or
            margin amount to point-equivalent capital before passing it; the
            backtester does not apply contract multipliers.
        min_tick: Positive instrument tick size. When omitted, infer it from
            observed prices. It must be supplied when inference is impossible
            and ``slippage`` is non-zero.
        sunday_to_monday: Combine Sunday observations with the following
            Monday. This is useful for Sunday-evening futures data. Disable it
            for markets where Sunday is an independent reporting day.
        use_numba: Use the optimized Numba engine. ``False`` selects the Python
            reference implementation.

    Returns:
        A :class:`Results` object. Inspect ``stats`` for summary metrics,
        ``daily`` for return/equity series, ``positions`` for completed trades,
        ``df`` for bar-level PnL/drawdowns, and ``warnings`` before accepting a
        result.

    Raises:
        TypeError: If the transaction input is not numeric or does not use a
            :class:`pandas.DatetimeIndex`.
        ValueError: If input data or performance parameters are invalid.
        RuntimeError: If bar-level and trade-level PnL fail to reconcile.
    """
    return _PerfCalculator(
        _TransactionFrame(data),
        slippage=slippage,
        skip_last_open=skip_last_open,
        capital=capital,
        min_tick=min_tick,
        sunday_to_monday=sunday_to_monday,
        use_numba=use_numba,
    ).run()


def auto_perf(
    data: pd.DataFrame,
    price_column: str = "open",
    slippage: float = 1.5,
    skip_last_open: bool = False,
    *,
    capital: float | None = None,
    min_tick: float | None = None,
    sunday_to_monday: bool = True,
    use_numba: bool = True,
) -> Results:
    """Evaluate either prepared transactions or an ordinary strategy frame.

    This convenience entrypoint is useful in notebooks where ``data`` may
    already contain the transaction columns required by :func:`perf`. If it
    does not, ``auto_perf()`` passes it through :func:`no_stop` using
    ``price_column``. Use :func:`perf` directly when explicit preparation is
    preferable or when using :func:`haymaker.research.stop.stop_loss`.

    Args:
        data: A transaction DataFrame accepted by :func:`perf`, or a raw frame
            containing ``position`` or ``blip`` accepted by :func:`no_stop`.
        price_column: Execution and mark-to-market price used only when raw
            strategy data needs conversion.
        slippage: Non-negative cost per transaction leg in multiples of
            ``min_tick``.
        skip_last_open: Exclude a position still open on the final input bar.
        capital: Positive initial capital in the same units as point PnL.
            Defaults to the first selected price.
        min_tick: Positive instrument tick size. Supply it explicitly when
            using sparse data or when it cannot be inferred.
        sunday_to_monday: Combine Sunday observations with the following
            Monday reporting day.
        use_numba: Use the optimized engine. Set to ``False`` for the Python
            reference implementation.

    Returns:
        The same :class:`Results` object returned by :func:`perf`.

    Raises:
        ValueError: If ``data`` is neither prepared transaction data nor valid
            input for :func:`no_stop`.
    """
    try:
        _TransactionFrame(data)
    except (TypeError, ValueError) as transaction_error:
        try:
            prepared = no_stop(data, price_column=price_column)
            return perf(
                prepared,
                slippage=slippage,
                skip_last_open=skip_last_open,
                capital=capital,
                min_tick=min_tick,
                sunday_to_monday=sunday_to_monday,
                use_numba=use_numba,
            )
        except Exception as conversion_error:
            raise ValueError(
                "auto_perf() could not interpret data as a transaction frame "
                "or convert it with no_stop(). "
                f"Transaction-frame error: {transaction_error}. "
                f"no_stop error: {conversion_error}."
            ) from conversion_error
    return perf(
        data,
        slippage=slippage,
        skip_last_open=skip_last_open,
        capital=capital,
        min_tick=min_tick,
        sunday_to_monday=sunday_to_monday,
        use_numba=use_numba,
    )
