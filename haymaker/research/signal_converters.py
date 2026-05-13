from typing import Literal, Optional, Union

import numpy as np
import pandas as pd  # type: ignore

from .numba_tools import _blip_to_signal_converter, _in_out_blip_unifier

"""
This module allows for most frequent conversions between various types
of signal series.

Throughout the package it is assumed that strategy information is
generated after the bar is completed (typically on close price). It is
not possible to execute a new strategy transaction within the same bar
where the information was generated. Stop-loss and take-profit orders
are treated differently because they are standing orders whose trigger
can occur inside the bar.

Following is an explanation of various terms used throughout
documentation in BINARY signal series (continuous signals are treated
separately):

indicator - computation series that the strategy is based on, ie.
difference of moving averages, macd, stochastic, etc.; actual value of
the indicator

signal - usually based on an indicator; at every bar shows where the
strategy wants to be after seeing that bar. A signal is generated on
that bar and is not itself an executable position.

blip - a sparse event recorded on the bar where the strategy learns
that it wants to execute a trade. It is not immediately actionable;
the transaction will typically be executed one bar after the blip.
Zero means no trade event was generated on that bar.

transaction - the point where actual transaction is executed; for
performance calculation typically this price bar should be used (in
most typical scenario, signals are generated on a bar 'close' and
transactions executed on next bar's 'open'); zero means no trade is
required; transaction results in change in position on the same bar

position - at every price bar, indicates actual executable/held state
after timing conversion (direction only; none of these series are
concerned with sizing).

'signal', 'blip', 'transaction', 'position' must only have one of
three values [-1, 0, 1] where -1 is short, 1 long and 0 flat.

Neither of those series are required in all dfs.  They are used on as
needed basis.  This terminology should be used consistently
throughout the package.

"""


def sig_blip(
    signal: pd.Series,
    clip: bool = True,
    side: Optional[Literal["open", "close"]] = None,
) -> pd.Series:
    """Convert a stateful signal series to sparse blip events.

    Args:
        signal: Desired strategy state indexed by bar.
        clip: If ``True``, clip generated blips to ``-1``/``0``/``1``.
            If ``False``, direct reversals in always-in-market systems may
            produce ``-2`` or ``2``.
        side: Optional side filter. ``"open"`` keeps only opening events,
            ``"close"`` keeps only closing events, and ``None`` keeps all
            events.

    Returns:
        Sparse blip series indexed like ``signal``.
    """
    if signal.isna().any():
        raise ValueError("signal series must not have any n/a values")

    o = (signal - signal.shift()).fillna(0)

    if side == "open":
        o = o * (signal.shift() == 0)
    elif side == "close":
        o = o * (signal.shift() != 0)

    if clip:
        return o.clip(-1, 1)
    else:
        return o


def pos_trans(position: pd.Series, clip=True) -> pd.Series:
    """
    Position to transaction converter.

    if clip == False in case of 'always-in-the-market' systems, ie.
    position is never zero, will return -2 or 2.
    """
    # is it the same as sig_blip with side=None???
    # (position.shift() != position) is redundant?
    # return sig_blip(position, clip, None)
    o = (position.shift() != position) * (position - position.shift()).fillna(0)
    if clip:
        return o.clip(-1, 1)
    else:
        return o


def pos_trans_array(position: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    NumPy position to transaction converter.

    Intended for internal use in array-oriented research code paths.
    """
    arr = np.asarray(position)
    out = np.zeros(arr.shape, dtype=np.int8)

    if arr.size <= 1:
        return out

    diff = arr[1:] - arr[:-1]
    changed = arr[1:] != arr[:-1]
    out[1:] = diff * changed

    if clip:
        np.clip(out, -1, 1, out=out)

    return out


def pos_trans_numpy(position: pd.Series, clip: bool = True) -> pd.Series:
    """
    Position to transaction converter using NumPy instead of pandas ops.
    """
    values = position.to_numpy(copy=False)
    out = pos_trans_array(values, clip=clip)
    return pd.Series(out, index=position.index, name=position.name)


def sig_pos(signal: pd.Series) -> pd.Series:
    """
    Convert signal to executable position.

    Position changes one bar after the signal was generated.
    """
    return signal.shift().fillna(0).astype(int)


def blip_sig(blip: Union[pd.Series, pd.DataFrame], always_on=True) -> pd.Series:
    """Convert sparse blip events to a stateful signal series.

    This does not produce an executable position. It produces the same-row
    desired state implied by generated blips. To backtest without stops, use
    the blip-aware ``no_stop`` path or apply the proper next-bar timing
    conversion after any frequency alignment.

    Args:
        blip: Sparse blip input. If a ``pd.Series`` is given, the same series
            represents both open and close events. If a ``pd.DataFrame`` is
            given, the first column is used for open events and the second
            column is used for close events. In the dataframe case, open blips
            are ignored while a position is active; only ``close_blip`` can
            close an existing position.
        always_on: Relevant only for ``pd.Series`` input. If ``True``, an
            opposite blip reverses the desired state immediately. If ``False``,
            an opposite blip first moves the desired state to flat.

    Returns:
        Stateful signal series indexed like ``blip``.
    """

    def verify(series: pd.Series) -> None:
        if not set(series.fillna(0)).issubset(set({-1, 0, 1})):
            raise ValueError(
                "Data passed to blip_sig contain values other than -1, 0, 1"
            )

    if isinstance(blip, pd.Series):
        verify(blip)
        return pd.Series(
            _blip_to_signal_converter(blip.to_numpy(), always_on), index=blip.index
        )
    elif isinstance(blip, pd.DataFrame):
        verify(blip.iloc[:, 0])
        verify(blip.iloc[:, 1])
        return pd.Series(
            _in_out_blip_unifier(blip.to_numpy()).flatten(),
            index=blip.index,
        )
    else:
        raise TypeError(f"Passed data must be a Series or DataFrame, not {type(blip)}")
