"""Performance and return metrics for research workflows."""

import numpy as np
import pandas as pd

__all__ = ["true_sharpe"]


def true_sharpe(ret: pd.Series) -> pd.Series:
    """Compare simple-return and log-return Sharpe calculations.

    Args:
        ret: Period returns as simple returns, for example ``P1 / P0 - 1``.

    Returns:
        A series with cumulative return, annualized return, annualized means,
        annualized volatility, and Sharpe ratios calculated from both simple
        and log returns.

    Raises:
        ValueError: If ``ret`` is empty.
    """
    if ret.empty:
        raise ValueError("ret must contain at least one return.")

    r = pd.Series(dtype=float)
    df = pd.DataFrame({"returns": ret.astype(float)})
    df["cumulative_return"] = (df["returns"] + 1).cumprod()
    df["log_returns"] = np.log(df["returns"] + 1)
    r["cumulative_return"] = df["cumulative_return"].iloc[-1] - 1
    r["annual_return"] = ((r["cumulative_return"] + 1) ** (252 / len(df.index))) - 1
    r["mean"] = df["returns"].mean() * 252
    r["mean_log"] = df["log_returns"].mean() * 252
    r["vol"] = df["returns"].std() * np.sqrt(252)
    r["vol_log"] = df["log_returns"].std() * np.sqrt(252)
    r["sharpe"] = r["mean"] / r["vol"]
    r["sharpe_log"] = r["mean_log"] / r["vol_log"]
    return r
