from typing import Literal, NamedTuple

import pandas as pd

from . import Results, no_stop, perf

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def v_backtester(
    indicator: pd.Series,
    threshold: float = 0,
    signal_or_position: Literal["signal", "position", "both"] = "position",
) -> pd.Series | pd.DataFrame:
    """
    Vector backtester.

    Run a quick and dirty backtest for a system that goes long when
    indicator is above threshold and short when indicator is below
    minus threshold.
    """
    assert signal_or_position in ("signal", "position", "both")

    df = pd.DataFrame({"indicator": indicator})
    df["signal"] = ((df["indicator"] > threshold) * 1) + (
        (df["indicator"] < -threshold) * -1
    )
    df["position"] = (df["signal"].shift(1).fillna(0)).astype("int")

    if signal_or_position == "position":
        return df["position"]
    elif signal_or_position == "signal":
        return df["signal"]
    else:
        return df


Out = NamedTuple(
    "Out",
    [
        ("stats", pd.DataFrame),
        ("dailys", pd.DataFrame),
        ("returns", pd.DataFrame),
        ("positions", dict[float, pd.DataFrame]),
        ("dfs", dict[float, pd.DataFrame]),
    ],
)


def summary(
    data: pd.Series | pd.DataFrame,
    indicator: pd.Series | None = None,
    slip: float = 0,
    threshold: list[float] | float | None = None,
    price_field_name: str = "open",
) -> Out:
    """Return stats summary of strategy for various thresholds."""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Data must be either Series or DataFrame")

    if isinstance(data, pd.DataFrame):
        if price_field_name not in data.columns:
            raise ValueError(f"'{price_field_name}' not in columns")
        price = data[price_field_name]

        if indicator is None:
            if "forecast" in data.columns:
                indicator = data["forecast"]
            else:
                raise KeyError("Indicator must be passed or df must have 'forecast'")
    else:
        price = data

    if threshold is None:
        threshold = [0, 1, 3, 5, 6, 10, 15, 17, 19, 20]
    elif isinstance(threshold, (int, float)):
        threshold = [threshold]

    stats = pd.DataFrame()
    dailys = pd.DataFrame()
    returns = pd.DataFrame()
    positions = {}
    dfs = {}

    for i in threshold:
        try:
            assert indicator is not None
            b = v_backtester(indicator, i)
            assert isinstance(b, pd.Series)
            tx = no_stop(
                pd.DataFrame({"open": price, "position": b}), price_column="open"
            )
            r = perf(tx, slippage=slip)
        except ZeroDivisionError:
            continue
        stats[i] = r.stats
        dailys[i] = r.daily["balance"]
        returns[i] = r.daily["returns"]
        positions[i] = r.positions
        dfs[i] = r.df

    return Out(stats, dailys, returns, positions, dfs)


def excursions(
    high_low: pd.DataFrame, positions: pd.DataFrame, divisor: pd.Series | None = None
) -> pd.DataFrame:
    """Calculate maximum adverse and favourable excursions."""
    if not {"high", "low"}.issubset(set(high_low.columns)):
        raise ValueError("high_low must have columns named 'high' and 'low'")

    high_low = high_low.copy()

    if divisor is None:
        high_low["divisor"] = 1
    elif isinstance(divisor, pd.Series) and (divisor.index == high_low.index).all():
        high_low["divisor"] = divisor
    else:
        raise ValueError(
            "divisor, if given must be a pd.Series with the same index as high_low"
        )

    def extremes(
        high_low: pd.DataFrame, positions: pd.DataFrame
    ) -> list[tuple[float, float, float]]:
        data = []
        for p in positions[["date_o", "date_c"]].itertuples():
            h_l = high_low.loc[p.date_o : p.date_c]  # type: ignore[misc]
            if len(h_l) > 1:
                h_l = h_l.iloc[:-1]
            data.append((h_l["high"].max(), h_l["low"].min(), h_l["divisor"].iloc[0]))
        return data

    out = positions.join(
        pd.DataFrame(extremes(high_low, positions), columns=["high", "low", "divisor"])
    )

    out["close_"] = out["close"].abs()
    out["high"] = out[["close_", "high"]].max(axis=1)
    out["low"] = out[["close_", "low"]].min(axis=1)

    out["_fav"] = ((out["open"] > 0) * out["high"]) + ((out["open"] < 0) * out["low"])
    out["_adv"] = ((out["open"] > 0) * out["low"]) + ((out["open"] < 0) * out["high"])

    out["fav"] = ((out["open"].abs() - out["_fav"]).abs() / out["divisor"]).round(2)
    out["adv"] = ((out["open"].abs() - out["_adv"]).abs() / out["divisor"]).round(2)
    out["eff"] = (out["g_pnl"] / (out["high"] - out["low"])).round(2)

    if divisor is None:
        return out[["fav", "adv", "eff"]]
    else:
        out["pnl_mul"] = (out["g_pnl"] / out["divisor"]).round(2)
        return out[["pnl_mul", "fav", "adv", "eff"]]


def profitable_excursions(
    df: pd.DataFrame,
    results: Results,
    divisor: pd.Series | None = None,
    full: bool = False,
) -> pd.Series | pd.DataFrame:
    positions = results.positions
    exc = positions.join(excursions(df[["high", "low"]], positions, divisor))
    out = exc[exc["pnl"] > 0]
    if full:
        return out
    else:
        return out.adv.describe()


adverse_excursions = profitable_excursions


def blip_extractor(signal: pd.Series) -> pd.Series:
    return (signal.shift() != signal) * signal


def factor_extractor(
    pos: pd.DataFrame,
    df: pd.DataFrame,
    field: str | list[str],
    shift: bool = True,
) -> pd.DataFrame:
    pos = pos.set_index("date_o")
    if shift:
        df = df.shift()

    if isinstance(field, str):
        field = [field]

    overlap = list(set(pos.columns).intersection(set(field)))
    cols = [c for c in pos.columns if c not in overlap]
    pos = pos[cols]

    return pos.join(df.loc[pos.index, field]).reset_index()
