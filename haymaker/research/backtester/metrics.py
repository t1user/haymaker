"""Performance metrics derived from reconciled backtester PnL."""

from __future__ import annotations

from collections.abc import Callable, MutableSequence
from typing import Any, cast

import numpy as np
import pandas as pd
from empyrical import (  # type: ignore[import-untyped]
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

SESSIONS_PER_YEAR = 252
SESSIONS_PER_MONTH = 21
ROLLING_DRAWDOWN_SESSIONS = 21


def _warn(warnings: MutableSequence[str], message: str) -> None:
    """Append a warning once."""
    if message not in warnings:
        warnings.append(message)


def reporting_dates(
    index: pd.DatetimeIndex,
    *,
    sunday_to_monday: bool,
) -> pd.DatetimeIndex:
    """Return reporting dates for observed bars.

    Dates are derived only from timestamps present in ``index``. When
    ``sunday_to_monday`` is enabled, Sunday observations are assigned to the
    following Monday; no other dates are inserted or shifted.

    Args:
        index: Ordered bar timestamps.
        sunday_to_monday: Whether Sunday observations belong to Monday's
            reporting day.

    Returns:
        Normalized reporting dates aligned one-for-one with ``index``.
    """
    dates = index.normalize()
    if sunday_to_monday:
        offsets = pd.to_timedelta((dates.dayofweek == 6).astype(int), unit="D")
        dates = dates + offsets
    return pd.DatetimeIndex(dates, name="session")


def _drawdown_paths(
    equity: pd.Series,
    capital: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return peak-relative, fixed-capital, and absolute drawdown paths."""
    if equity.empty:
        empty = pd.Series(index=equity.index, dtype=float)
        return (
            empty.rename("drawdown"),
            empty.rename("fixed_drawdown"),
            empty.rename("drawdown_pnl"),
        )

    running_peak = pd.Series(
        np.maximum.accumulate(
            np.concatenate(([capital], equity.to_numpy(dtype=float)))
        )[1:],
        index=equity.index,
    )
    drawdown_pnl = equity - running_peak
    drawdown = equity / running_peak - 1.0
    fixed_drawdown = drawdown_pnl / capital
    return (
        drawdown.rename("drawdown"),
        fixed_drawdown.rename("fixed_drawdown"),
        drawdown_pnl.rename("drawdown_pnl"),
    )


def build_performance_frames(
    bar_df: pd.DataFrame,
    *,
    capital: float,
    sunday_to_monday: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add equity paths and aggregate bar PnL into reporting sessions.

    Account returns use beginning-of-session equity. Fixed returns use the
    unchanged initial capital. Neither return series is calculated at bar
    frequency.

    Args:
        bar_df: Bar-level result containing a net ``pnl`` column.
        capital: Positive starting account capital in the same units as PnL.
        sunday_to_monday: Whether to combine Sunday observations with Monday.

    Returns:
        A pair containing the enriched bar frame and the session-level frame.
    """
    bar_df = bar_df.copy()
    if bar_df.empty:
        for column in (
            "equity",
            "balance",
            "drawdown",
            "fixed_drawdown",
            "drawdown_pnl",
        ):
            bar_df[column] = pd.Series(index=bar_df.index, dtype=float)
        daily = pd.DataFrame(
            columns=("pnl", "returns", "lreturn", "equity", "balance", "fixed_return"),
            index=pd.DatetimeIndex([], name="session"),
            dtype=float,
        )
        return bar_df, daily

    bar_df["equity"] = capital + bar_df["pnl"].cumsum()
    bar_df["balance"] = bar_df["equity"] / capital
    (
        bar_df["drawdown"],
        bar_df["fixed_drawdown"],
        bar_df["drawdown_pnl"],
    ) = _drawdown_paths(bar_df["equity"], capital)

    dates = reporting_dates(
        pd.DatetimeIndex(bar_df.index), sunday_to_monday=sunday_to_monday
    )
    session_pnl = bar_df["pnl"].groupby(dates, sort=True).sum()
    daily = pd.DataFrame({"pnl": session_pnl})
    daily["equity"] = capital + daily["pnl"].cumsum()
    beginning_equity = daily["equity"].shift(fill_value=capital)
    daily["returns"] = daily["pnl"] / beginning_equity.where(beginning_equity > 0)
    equity_values = daily["equity"].to_numpy(dtype=float)
    insolvent_sessions = np.flatnonzero(equity_values <= 0)
    if len(insolvent_sessions):
        after_insolvency = np.arange(len(daily)) > insolvent_sessions[0]
        daily["returns"] = daily["returns"].mask(after_insolvency)
    valid_returns = daily["returns"].where(daily["returns"] > -1)
    daily["lreturn"] = np.log1p(valid_returns)
    daily["balance"] = daily["equity"] / capital
    daily["fixed_return"] = daily["pnl"] / capital
    return (
        bar_df,
        daily[["pnl", "returns", "lreturn", "equity", "balance", "fixed_return"]],
    )


def _rolling_drawdowns(
    daily_equity: pd.Series,
    capital: float,
    *,
    window: int = ROLLING_DRAWDOWN_SESSIONS,
) -> tuple[float, float]:
    """Return median standard and fixed drawdowns over rolling sessions."""
    if len(daily_equity) < window:
        return float("nan"), float("nan")

    equity = np.concatenate(([capital], daily_equity.to_numpy(dtype=float)))
    standard: list[float] = []
    fixed: list[float] = []
    for end in range(window, len(equity)):
        sample = equity[end - window : end + 1]
        peaks = np.maximum.accumulate(sample)
        pnl_drawdown = sample - peaks
        standard.append(float(np.min(sample / peaks - 1.0)))
        fixed.append(float(np.min(pnl_drawdown / capital)))
    return float(np.median(standard)), float(np.median(fixed))


def _safe_empyrical_metric(
    metric: Callable[..., Any],
    returns: pd.Series,
) -> float:
    """Call a scalar Empyrical metric with the project's annualization."""
    return float(metric(returns, annualization=SESSIONS_PER_YEAR))


def return_stream_metrics(returns: pd.Series) -> pd.Series:
    """Return the lean standard metric set for periodic simple returns.

    This helper is for an already constructed return stream, such as the
    equal-weight stream produced by ``GridSearchResult.combined_returns()``.
    Its drawdown is necessarily measured at the input return frequency.

    Args:
        returns: Non-cumulative periodic simple returns.

    Returns:
        Compact standard performance statistics.
    """
    returns = returns.dropna().astype(float)
    stats = pd.Series(dtype="object")
    stats["total_return"] = (
        float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)
        if len(returns)
        else float("nan")
    )
    annual_return_value = _safe_empyrical_metric(annual_return, returns)
    stats["annual_return"] = annual_return_value
    stats["annual_volatility"] = _safe_empyrical_metric(annual_volatility, returns)
    stats["sharpe_ratio"] = _safe_empyrical_metric(sharpe_ratio, returns)
    stats["sortino_ratio"] = _safe_empyrical_metric(sortino_ratio, returns)
    max_drawdown_value = float(max_drawdown(returns))
    stats["max_drawdown"] = max_drawdown_value
    stats["calmar_ratio"] = (
        annual_return_value / abs(max_drawdown_value)
        if max_drawdown_value < 0
        else float("nan")
    )
    equity = (1.0 + returns).cumprod()
    stats["median_21d_drawdown"] = _rolling_drawdowns(equity, 1.0)[0]
    stats["skew"] = float(cast(float, returns.skew()))
    stats["kurtosis"] = float(cast(float, returns.kurt()))
    if len(returns) and not bool((returns < 0).any()):
        stats["sortino_ratio"] = float("nan")
    return stats


def return_metrics(
    daily: pd.DataFrame,
    bar_df: pd.DataFrame,
    *,
    capital: float,
    warnings: MutableSequence[str],
) -> pd.Series:
    """Calculate account and fixed-capital performance statistics."""
    stats = pd.Series(dtype="object")
    total_pnl = float(daily["pnl"].sum()) if not daily.empty else 0.0
    stats["total_return"] = total_pnl / capital

    if bar_df.empty:
        account_drawdown = fixed_drawdown = drawdown_pnl = float("nan")
    else:
        account_drawdown = float(bar_df["drawdown"].min())
        fixed_drawdown = float(bar_df["fixed_drawdown"].min())
        drawdown_pnl = float(bar_df["drawdown_pnl"].min())
    stats["max_drawdown"] = account_drawdown
    stats["fixed_max_drawdown"] = fixed_drawdown
    stats["max_drawdown_pnl"] = drawdown_pnl

    rolling, fixed_rolling = _rolling_drawdowns(daily["equity"], capital)
    insolvent = bool((bar_df["equity"] <= 0).any()) if not bar_df.empty else False
    if insolvent:
        rolling = float("nan")
    stats["median_21d_drawdown"] = rolling
    stats["fixed_median_21d_drawdown"] = fixed_rolling
    if len(daily) < ROLLING_DRAWDOWN_SESSIONS:
        _warn(
            warnings,
            "Fewer than 21 sessions; rolling drawdown metrics are undefined.",
        )

    fixed_returns = daily["fixed_return"].dropna()
    stats["fixed_annual_return"] = (
        float(fixed_returns.mean() * SESSIONS_PER_YEAR)
        if len(fixed_returns)
        else float("nan")
    )
    stats["fixed_annual_volatility"] = _safe_empyrical_metric(
        annual_volatility, fixed_returns
    )
    stats["fixed_sharpe_ratio"] = _safe_empyrical_metric(sharpe_ratio, fixed_returns)
    stats["fixed_sortino_ratio"] = _safe_empyrical_metric(sortino_ratio, fixed_returns)

    returns = daily["returns"].dropna()
    account_metric_names = (
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "skew",
        "kurtosis",
    )
    if insolvent:
        _warn(
            warnings,
            "Account equity became nonpositive; compounded account metrics "
            "are undefined.",
        )
        for name in account_metric_names:
            stats[name] = float("nan")
    else:
        annual_return_value = _safe_empyrical_metric(annual_return, returns)
        stats["annual_return"] = annual_return_value
        stats["annual_volatility"] = _safe_empyrical_metric(annual_volatility, returns)
        stats["sharpe_ratio"] = _safe_empyrical_metric(sharpe_ratio, returns)
        stats["sortino_ratio"] = _safe_empyrical_metric(sortino_ratio, returns)
        stats["calmar_ratio"] = (
            annual_return_value / abs(account_drawdown)
            if account_drawdown < 0
            else float("nan")
        )
        stats["skew"] = float(cast(float, returns.skew()))
        stats["kurtosis"] = float(cast(float, returns.kurt()))

    if len(daily) < 2:
        _warn(
            warnings,
            "Fewer than two sessions; volatility and ratio metrics are undefined.",
        )
    elif float(fixed_returns.std(ddof=1)) == 0:
        _warn(warnings, "Session returns have zero volatility; ratios are undefined.")

    if len(fixed_returns) and not bool((fixed_returns < 0).any()):
        stats["fixed_sortino_ratio"] = float("nan")
        _warn(warnings, "No negative fixed returns; fixed Sortino ratio is undefined.")
    if len(returns) and not bool((returns < 0).any()):
        stats["sortino_ratio"] = float("nan")
        _warn(warnings, "No negative account returns; Sortino ratio is undefined.")

    return stats


def trade_metrics(
    positions: pd.DataFrame,
    daily: pd.DataFrame,
    bar_df: pd.DataFrame,
    *,
    min_tick: float,
    warnings: MutableSequence[str],
) -> pd.Series:
    """Calculate completed-trade and point-PnL statistics."""
    stats = pd.Series(dtype="object")
    pnl = positions["pnl"].astype(float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    trade_count = len(positions)
    session_count = len(daily)

    gross_pnl = float(positions["g_pnl"].sum()) if trade_count else 0.0
    net_pnl = float(pnl.sum()) if trade_count else 0.0
    stats["gross_pnl"] = gross_pnl
    stats["net_pnl"] = net_pnl
    stats["monthly_pnl"] = (
        float(daily["pnl"].mean() * SESSIONS_PER_MONTH)
        if session_count
        else float("nan")
    )
    stats["annual_pnl"] = (
        float(daily["pnl"].mean() * SESSIONS_PER_YEAR)
        if session_count
        else float("nan")
    )
    stats["trade_count"] = trade_count
    stats["session_count"] = session_count
    stats["win_rate"] = len(winners) / trade_count if trade_count else float("nan")
    stats["avg_win"] = float(winners.mean()) if len(winners) else float("nan")
    stats["avg_loss"] = float(losers.mean()) if len(losers) else float("nan")
    stats["payoff_ratio"] = (
        abs(stats["avg_win"] / stats["avg_loss"])
        if len(winners) and len(losers)
        else float("nan")
    )
    stats["profit_factor"] = (
        float(winners.sum() / abs(losers.sum()))
        if len(winners) and len(losers)
        else float("nan")
    )
    expectancy = float(pnl.mean()) if trade_count else float("nan")
    stats["trade_expectancy"] = expectancy
    stats["trade_expectancy_ticks"] = (
        expectancy / min_tick if trade_count and min_tick > 0 else float("nan")
    )
    if min_tick <= 0:
        _warn(
            warnings, "Minimum tick could not be inferred; tick metrics are undefined."
        )

    long_pnl = positions.loc[positions["open"] > 0, "pnl"]
    short_pnl = positions.loc[positions["open"] < 0, "pnl"]
    stats["long_expectancy"] = float(long_pnl.mean()) if len(long_pnl) else float("nan")
    stats["short_expectancy"] = (
        float(short_pnl.mean()) if len(short_pnl) else float("nan")
    )
    stats["trades_per_session"] = (
        trade_count / session_count if session_count else float("nan")
    )

    durations = positions["duration"]
    stats["avg_duration"] = durations.mean() if trade_count else pd.NaT
    stats["median_duration"] = durations.median() if trade_count else pd.NaT
    stats["p90_duration"] = durations.quantile(0.9) if trade_count else pd.NaT
    stats["max_duration"] = durations.max() if trade_count else pd.NaT

    if bar_df.empty:
        stats["time_in_market"] = float("nan")
    else:
        exposed = (
            bar_df["position"].ne(0)
            | bar_df["open_price"].ne(0)
            | bar_df["close_price"].ne(0)
            | bar_df["stop_price"].ne(0)
        )
        stats["time_in_market"] = float(exposed.mean())

    if trade_count:
        sorted_pnl = pnl.sort_values(ascending=False)
        stats["best_trade"] = float(sorted_pnl.iloc[0])
        stats["worst_trade"] = float(sorted_pnl.iloc[-1])
        stats["net_pnl_ex_best"] = net_pnl - stats["best_trade"]
        gross_profit = float(winners.sum())
        stats["best_trade_share"] = (
            max(stats["best_trade"], 0.0) / gross_profit
            if gross_profit > 0
            else float("nan")
        )
        stats["top5_trade_share"] = (
            float(winners.nlargest(5).sum() / gross_profit)
            if gross_profit > 0
            else float("nan")
        )
    else:
        for name in (
            "best_trade",
            "worst_trade",
            "net_pnl_ex_best",
            "best_trade_share",
            "top5_trade_share",
        ):
            stats[name] = float("nan")

    return stats


def build_stats(
    positions: pd.DataFrame,
    daily: pd.DataFrame,
    bar_df: pd.DataFrame,
    *,
    capital: float,
    min_tick: float,
    warnings: MutableSequence[str],
) -> pd.Series:
    """Return the complete ordered performance-statistics series."""
    return pd.concat(
        (
            return_metrics(daily, bar_df, capital=capital, warnings=warnings),
            trade_metrics(
                positions,
                daily,
                bar_df,
                min_tick=min_tick,
                warnings=warnings,
            ),
        )
    )
