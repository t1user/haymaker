"""Backtest result analysis helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtester import Results

__all__ = ["always_on", "long_short_returns", "paths"]


def long_short_returns(r: Results) -> pd.DataFrame:
    """Return cumulative long-only and short-only trade-return paths.

    This is a hypothetical decomposition of completed trades in
    ``Results.positions``. Long trades are included only in the ``long`` path
    and short trades only in the ``short`` path. Per-trade returns are computed
    as log returns, ``log1p(pnl / abs(open))``, then accumulated and converted
    back to cumulative simple-return paths.
    """
    pos = r.positions.copy()
    pos["return"] = np.log1p(pos["pnl"] / pos["open"].abs())
    pos = pos.set_index("date_c")
    long = pos[pos["open"] > 0]
    short = pos[pos["open"] < 0]
    combined = pd.DataFrame({"long": long["return"], "short": short["return"]})
    return combined.fillna(0).cumsum().apply(np.exp)


def paths(r: Results, cumsum: bool = True, log_return: bool = False) -> pd.DataFrame:
    """Return chart-ready strategy, long, short, and underlying paths.

    This helper is meant for quick visual comparison between the strategy,
    the underlying asset movement, and the parts of strategy performance
    contributed by long and short exposure.

    Args:
        r: Backtester result returned by :func:`haymaker.research.backtester.perf`.
        cumsum: If ``True``, return running sums suitable for a path chart.
            With ``log_return=True``, these are cumulative log returns. With
            ``log_return=False``, these are cumulative price-point/PnL values.
        log_return: If ``True``, use bar log returns from the strategy and
            underlying asset. If ``False``, use absolute price-point movement
            for the underlying and absolute PnL for the strategy.

    Returns:
        DataFrame with four columns:

        - ``price``: movement of the underlying asset.
        - ``longs``: strategy bar result attributed to long exposure.
        - ``shorts``: strategy bar result attributed to short exposure.
        - ``strategy``: total strategy bar result.

        For current ``perf()`` output, exit rows are attributed to the side
        being closed. This matters on reversal bars, where ``position`` already
        contains the new side but the closing mark-to-market PnL belongs to the
        previous side.
    """

    rdf = r.df.copy()
    price_column = "price" if "price" in rdf.columns else "bar_price"
    if log_return:
        field = "lreturn"
        price = np.log(rdf[price_column].pct_change() + 1)  # type: ignore
    else:
        field = "pnl"
        price = rdf[price_column].diff()  # type: ignore

    if {"transaction", "curr_price"}.issubset(rdf.columns):
        half_return = rdf[field] / rdf["transaction"].abs()
        rdf["_return"] = half_return.mask(
            half_return.replace([-np.inf, np.inf], np.nan).isna(), rdf[field]
        )
        longs = rdf[(rdf["curr_price"] > 0) | (rdf["position"] == 1)]
        shorts = rdf[(rdf["curr_price"] < 0) | (rdf["position"] == -1)]
    else:
        rdf["_return"] = rdf[field]
        direction = pd.Series(0.0, index=rdf.index)
        direction = direction.mask(rdf["position"] != 0, rdf["position"])
        if "open_price" in rdf.columns:
            direction = direction.mask(
                rdf["open_price"] != 0,
                np.sign(rdf["open_price"]),  # type: ignore
            )
        if "close_price" in rdf.columns:
            direction = direction.mask(
                rdf["close_price"] != 0,
                -np.sign(rdf["close_price"]),  # type: ignore
            )
        if "stop_price" in rdf.columns:
            direction = direction.mask(
                rdf["stop_price"] != 0,
                -np.sign(rdf["stop_price"]),  # type: ignore
            )
        longs = rdf[direction > 0]
        shorts = rdf[direction < 0]
    df = pd.DataFrame(
        {
            "price": price,
            "longs": longs["_return"],
            "shorts": shorts["_return"],
            "strategy": rdf[field],
        }
    ).fillna(0)

    if cumsum:
        return df.cumsum()
    else:
        return df


def always_on(series: pd.Series) -> bool:
    """Return whether a position series stays in-market after first entry.

    Leading flat rows are ignored. A series with only zero positions, or an
    empty series, returns ``False``.
    """
    non_zero_positions = np.flatnonzero(series.ne(0).to_numpy())
    if len(non_zero_positions) == 0:
        return False
    return bool(series.iloc[non_zero_positions[0] :].ne(0).all())
