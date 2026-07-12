"""Backtest result analysis helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .backtester import Results

__all__ = [
    "excursions",
    "factor_extractor",
    "long_short_returns",
    "paths",
    "winning_trade_adverse_excursions",
]


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    """Raise when a dataframe does not contain its required columns.

    Args:
        frame: Dataframe to validate.
        required: Column names required by the caller.
        frame_name: User-facing dataframe name for the error message.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def _validate_excursion_inputs(
    high_low: pd.DataFrame,
    positions: pd.DataFrame,
    divisor: pd.Series | None,
) -> pd.Series:
    """Validate excursion inputs and return the per-bar scaling series.

    Args:
        high_low: Bar-indexed high and low prices.
        positions: Trade records produced by the backtester.
        divisor: Optional positive scale aligned exactly with ``high_low``.

    Returns:
        Scaling values aligned with ``high_low``.

    Raises:
        TypeError: If ``divisor`` is not a Series.
        ValueError: If required data is missing, misaligned, or invalid.
    """
    _require_columns(high_low, {"high", "low"}, "high_low")
    _require_columns(
        positions,
        {"date_o", "open", "date_c", "close", "g_pnl"},
        "positions",
    )

    if not high_low.index.is_unique:
        raise ValueError("high_low index must be unique")
    if not high_low.index.is_monotonic_increasing:
        raise ValueError("high_low index must be sorted in increasing order")
    if high_low[["high", "low"]].isna().any().any():
        raise ValueError("high_low must not contain missing high or low values")
    if (high_low["high"] < high_low["low"]).any():
        raise ValueError("high_low contains a high price below its low price")

    dates = pd.concat([positions["date_o"], positions["date_c"]])
    missing_dates = pd.Index(dates[~dates.isin(high_low.index)].unique())
    if len(missing_dates):
        raise ValueError(
            "position entry and exit dates must exist in the high_low index: "
            f"{missing_dates.tolist()}"
        )
    if (positions["date_c"] < positions["date_o"]).any():
        raise ValueError("position exit dates must not precede entry dates")
    if positions[["open", "close"]].eq(0).any().any():
        raise ValueError("position open and close prices must be non-zero")

    if divisor is None:
        return pd.Series(1.0, index=high_low.index)
    if not isinstance(divisor, pd.Series):
        raise TypeError("divisor must be a pandas Series")
    if not divisor.index.equals(high_low.index):
        raise ValueError("divisor must have exactly the same index as high_low")
    if divisor.isna().any() or (divisor <= 0).any():
        raise ValueError("divisor values must be positive and non-missing")
    return divisor.astype(float)


def excursions(
    high_low: pd.DataFrame,
    positions: pd.DataFrame,
    divisor: pd.Series | None = None,
) -> pd.DataFrame:
    """Measure how far completed trades moved for and against the strategy.

    An excursion describes the price path between entry and exit:

    - Maximum favorable excursion (``fav`` or MFE) is the largest unrealized
      gain the trade offered before it closed.
    - Maximum adverse excursion (``adv`` or MAE) is the largest unrealized loss
      the trade had to withstand before it closed.

    These measurements help diagnose whether winning trades commonly endure a
    material drawdown, whether trades offer substantially more profit than the
    strategy captures, and whether stop distances or profit targets deserve
    further investigation. Comparing excursions across market regimes can also
    reveal when a strategy's trade management is unusually loose or restrictive.

    Excursions are descriptive, not a simulation of alternative exits. Bar data
    does not reveal whether a bar's favorable or adverse extreme occurred first,
    so these values alone cannot prove how a different stop or target would have
    changed the result.

    Args:
        high_low: Bar-indexed dataframe containing numeric ``high`` and ``low``
            columns. Its index must be unique and increasing.
        positions: Completed trades containing ``date_o``, ``open``, ``date_c``,
            ``close``, and ``g_pnl``. Prices use the signed convention returned
            by :func:`haymaker.research.backtester.perf`.
        divisor: Optional positive scale aligned exactly with ``high_low``. Its
            value at entry is used to normalize ``fav``, ``adv``, and gross PnL.
            For example, passing ATR expresses the results in entry-time ATR
            units, making trades from different volatility regimes easier to
            compare.

    Returns:
        Dataframe indexed like ``positions`` containing:

        - ``fav``: Maximum favorable excursion in price or divisor units.
        - ``adv``: Maximum adverse excursion in price or divisor units.
        - ``eff``: Gross PnL divided by the trade's observed price range. A
          positive value means the trade finished profitably; a larger positive
          value means it captured more of the available range.
        - ``pnl_mul``: Gross PnL in divisor units. Present only when ``divisor``
          is supplied.

    Raises:
        TypeError: If ``divisor`` is not a Series.
        ValueError: If required columns, dates, index alignment, or price data
            are invalid.

    Example:
        Measure trades in entry-time ATR units and attach the measurements to
        the backtester position table::

            metrics = excursions(
                bars[["high", "low"]],
                result.positions,
                divisor=bars["atr"],
            )
            trades = result.positions.join(metrics)
    """
    scale = _validate_excursion_inputs(high_low, positions, divisor)
    columns = ["fav", "adv", "eff"]
    if divisor is not None:
        columns.insert(0, "pnl_mul")
    if positions.empty:
        return pd.DataFrame(index=positions.index, columns=columns, dtype=float)

    entry_dates = positions["date_o"].tolist()
    exit_dates = positions["date_c"].tolist()
    entry_locations = high_low.index.get_indexer(entry_dates)
    exit_locations = high_low.index.get_indexer(exit_dates)
    open_prices = positions["open"].to_numpy(dtype=float)
    close_prices = positions["close"].to_numpy(dtype=float)
    gross_pnls = positions["g_pnl"].to_numpy(dtype=float)

    rows: list[dict[str, float]] = []
    for (
        entry_date,
        entry_location,
        exit_location,
        open_price,
        close_price,
        gross_pnl,
    ) in zip(
        entry_dates,
        entry_locations,
        exit_locations,
        open_prices,
        close_prices,
        gross_pnls,
        strict=True,
    ):
        entry_location = int(entry_location)
        exit_location = int(exit_location)
        observed = high_low.iloc[entry_location:exit_location]

        entry_price = abs(float(open_price))
        exit_price = abs(float(close_price))
        observed_high = max(
            [entry_price, exit_price, *observed["high"].astype(float).tolist()]
        )
        observed_low = min(
            [entry_price, exit_price, *observed["low"].astype(float).tolist()]
        )
        entry_scale = float(scale.loc[entry_date])

        if open_price > 0:
            favorable = observed_high - entry_price
            adverse = entry_price - observed_low
        else:
            favorable = entry_price - observed_low
            adverse = observed_high - entry_price

        price_range = observed_high - observed_low
        row = {
            "fav": round(favorable / entry_scale, 2),
            "adv": round(adverse / entry_scale, 2),
            "eff": (
                round(float(gross_pnl) / price_range, 2)
                if price_range
                else float("nan")
            ),
        }
        if divisor is not None:
            row["pnl_mul"] = round(float(gross_pnl) / entry_scale, 2)
        rows.append(row)

    return pd.DataFrame(rows, index=positions.index)[columns]


def winning_trade_adverse_excursions(
    df: pd.DataFrame,
    results: Results,
    divisor: pd.Series | None = None,
    full: bool = False,
) -> pd.Series | pd.DataFrame:
    """Analyze the adverse movement experienced by profitable trades.

    This helper answers a practical stop-research question: how far did eventual
    winners typically move against the strategy before becoming profitable? The
    distribution can identify stop distances that would have removed many past
    winners and can highlight unusually volatile winning trades for closer
    inspection. It should be treated as diagnostic evidence, not proof that a
    wider stop would improve future performance.

    A trade is profitable when its net ``pnl`` is positive. Excursion efficiency
    uses gross ``g_pnl`` because it describes captured price movement rather than
    transaction costs.

    Args:
        df: Bar-indexed market dataframe containing ``high`` and ``low``.
        results: Backtester results whose ``positions`` contain completed trades.
        divisor: Optional positive scale aligned exactly with ``df``. Passing a
            volatility measure such as ATR makes the adverse excursions
            comparable across different volatility levels.
        full: If ``True``, return profitable trade rows with excursion columns.
            If ``False``, return descriptive statistics for their adverse
            excursions.

    Returns:
        Profitable trades with excursion columns, or a descriptive Series for
        their ``adv`` values.

    Example:
        Inspect the adverse-excursion distribution in ATR units::

            winning_trade_adverse_excursions(
                bars,
                result,
                divisor=bars["atr"],
            )
    """
    positions = results.positions
    metrics = excursions(df[["high", "low"]], positions, divisor)
    metric_columns = ["pnl_mul", "fav", "adv", "eff"]
    enriched = positions.drop(columns=metric_columns, errors="ignore").join(metrics)
    profitable = enriched[enriched["pnl"] > 0]
    if full:
        return profitable
    return profitable["adv"].describe()


def factor_extractor(
    positions: pd.DataFrame,
    data: pd.DataFrame,
    field: str | list[str],
    shift: bool = True,
) -> pd.DataFrame:
    """Enrich completed trades with market information available at entry.

    The function looks up one or more columns from bar-indexed ``data`` for each
    trade's ``date_o`` and appends those values to the trade table. A "factor"
    can be any variable useful for explaining results, such as a forecast, ATR,
    volume, volatility regime, trend strength, or session label; this function
    does not calculate the factor itself.

    The enriched table can be grouped or filtered to investigate questions such
    as whether stronger forecasts produce better trades, whether performance
    deteriorates in high-volatility regimes, or whether losses cluster around a
    particular entry condition. Trade PnL and timing are not modified.

    Args:
        positions: Trade dataframe containing ``date_o`` entry timestamps.
        data: Bar-indexed dataframe containing the requested factor fields. Its
            index must be unique and contain every entry timestamp.
        field: Factor column name or list of column names to attach. Existing
            columns of the same names in ``positions`` are replaced.
        shift: If ``True``, use each field's value from the observation preceding
            the entry row. This is appropriate when the current row's value is
            only known after that bar completes. Set it to ``False`` when values
            are already known at the entry timestamp. On an irregular index,
            "preceding" means the preceding row, not a fixed time interval.

    Returns:
        A new dataframe with one row per input trade, in the original order,
        and the requested factor columns appended.

    Raises:
        ValueError: If required columns, fields, entry timestamps, or index
            uniqueness constraints are not satisfied.

    Example:
        Attach the forecast and ATR known before each entry, then compare mean
        trade PnL by forecast quartile::

            trades = factor_extractor(
                result.positions,
                bars,
                ["forecast", "atr"],
            )
            quartile = pd.qcut(trades["forecast"], 4)
            pnl_by_forecast = trades.groupby(quartile)["pnl"].mean()
    """
    _require_columns(positions, {"date_o"}, "positions")
    fields = [field] if isinstance(field, str) else list(field)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("field must contain one or more unique column names")
    _require_columns(data, set(fields), "data")
    if not data.index.is_unique:
        raise ValueError("data index must be unique")

    missing_dates = pd.Index(
        positions.loc[~positions["date_o"].isin(data.index), "date_o"].unique()
    )
    if len(missing_dates):
        raise ValueError(
            "position entry dates must exist in the data index: "
            f"{missing_dates.tolist()}"
        )

    source = data.shift() if shift else data
    entry_factors = source.reindex(pd.Index(positions["date_o"]))[fields]
    output = positions.drop(columns=fields, errors="ignore").reset_index(drop=True)
    for name in fields:
        output[name] = entry_factors[name].to_numpy()
    return output


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
        log_return: If ``True``, use changes in log account balance for the
            strategy and log returns for the underlying. If ``False``, use
            absolute price-point movement and strategy PnL.

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
        if "balance" in rdf.columns:
            valid_balance = rdf["balance"].where(rdf["balance"] > 0)
            rdf["_account_log_return"] = np.log(valid_balance).diff()
            if len(rdf) and pd.notna(valid_balance.iloc[0]):
                rdf.at[rdf.index[0], "_account_log_return"] = np.log(
                    valid_balance.iloc[0]
                )
            field = "_account_log_return"
        else:
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
