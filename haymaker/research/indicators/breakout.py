"""Breakout and rolling-extreme indicators."""

import numpy as np
import pandas as pd

__all__ = [
    "breakout",
    "breakout_blip",
    "min_max_blip",
    "min_max_index",
]


def min_max_blip(
    price: pd.Series, period: int, buff: float | pd.Series = 0
) -> pd.Series:
    """Return breakout blips from the previous rolling high or low.

    Args:
        price: Price series.
        period: Lookback window for the previous rolling high and low.
        buff: Price-unit buffer added to the upper breakout threshold and
            subtracted from the lower breakout threshold. A positive value makes
            breakouts harder to trigger. If passed as a Series, it must have the
            same index as ``price``.

    Returns:
        Series with ``1`` when price breaks above the previous rolling high plus
        ``buff``, ``-1`` when price breaks below the previous rolling low minus
        ``buff``, and ``0`` otherwise. Blips are recorded on the bar where they
        are detected.
    """
    if isinstance(buff, pd.Series) and not buff.index.equals(price.index):
        raise ValueError("buff index must match price index")

    prior_max = price.rolling(period, closed="left").max()
    prior_min = price.rolling(period, closed="left").min()
    df = pd.DataFrame(
        {
            "max": (price > prior_max + buff) * 1,
            "min": (price < prior_min - buff) * 1,
        }
    )
    df["blip"] = df["max"] - df["min"]
    return df["blip"].rename(None)


def breakout(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
    always_on: bool = False,
) -> pd.Series:
    """Return stateful breakout signal from rolling high/low blips.

    Args:
        price: Price series, typically close prices.
        lookback: Rolling high/low lookback.
        stop_frac: Fraction of ``lookback`` used for opposite-side close blips.
            Must be in ``(0, 1]``. If ``stop_frac == 1``, opposite breakout
            blips are interpreted directly by ``always_on``.
        always_on: Relevant when ``stop_frac == 1``. If ``True``, an opposite
            breakout blip reverses the signal immediately. If ``False``, an
            opposite breakout blip closes the existing signal to zero.

    Returns:
        Series where ``1`` means long signal, ``-1`` short signal, and ``0`` no
        desired market exposure at that bar. The signal is recorded on the bar
        where the breakout is detected; for next-bar execution, shift the output
        before deriving executable positions.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    if stop_frac == 1:
        from ..numba_tools import _blip_to_signal_converter

        df["break"] = _blip_to_signal_converter(
            df["in"].to_numpy(), always_on=always_on
        )
    else:
        from ..numba_tools import _in_out_blip_unifier

        df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
        df["break"] = _in_out_blip_unifier(df[["in", "out"]].to_numpy())
    return df["break"]


def breakout_blip(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
) -> pd.Series:
    """Return breakout event blips instead of a stateful breakout signal.

    Blips are raw generated events recorded on the bar where the breakout
    information becomes known. Do not pre-shift them before passing them to
    blip-aware consumers such as ``stop_loss`` or ``no_stop``. For direct
    next-bar execution, shift the returned blips before deriving positions.

    Args:
        price: Price series, typically close prices.
        lookback: Rolling high/low lookback.
        stop_frac: Fraction of ``lookback`` used for opposite-side close blips.

    Returns:
        Series with ``1`` and ``-1`` breakout blips, and ``0`` otherwise.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
    df["break"] = (df["in"] + df["out"]).clip(-1, 1)
    return df["break"]


def min_max_index(
    price: pd.Series, lookback: int, cutoff_value: int = 1, binary: bool = True
) -> pd.Series:
    """Return relative position of recent rolling min and max.

    Args:
        price: Price series.
        lookback: Rolling window for finding the most recent min and max.
        cutoff_value: Absolute index difference below this value is treated as
            unchanged and forward-filled.
        binary: If ``True``, return only the sign of the index: ``1`` or ``-1``.

    Returns:
        Signed trend indicator. Negative values mean the rolling max is more
        recent than the rolling min; positive values mean the rolling min is
        more recent than the rolling max.
    """
    from ..numba_tools import rolling_min_max_index

    df = pd.DataFrame(
        rolling_min_max_index(price.to_numpy(), lookback),
        columns=["min", "max"],
        index=price.index,
    )
    df["ind"] = df["min"] - df["max"]
    df.loc[df[df["ind"].abs() < cutoff_value].index, "ind"] = 0
    df = df.replace(0, np.nan).ffill()
    if binary:
        return np.sign(df["ind"])  # type: ignore
    else:
        return df["ind"]
