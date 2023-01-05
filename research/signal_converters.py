import pandas as pd  # type: ignore
from typing import Optional, Literal, Union
from numba_tools import _blip_to_signal_converter, _in_out_signal_unifier


"""
This module allows for most frequent conversions between various types
of signal series.

Throughout the package it is assumed that transaction signals are
generated after price is completed (typically on close price) and it
is not possible to execute transaction within the same bar where
signal has been generated (except for stop-loss and take-profit order,
which are treated differently due to their nature).

Following is an explanation of various terms used throughout
documentation in BINARY signal series (continuous signals are treated
separately):

indicator - computation series that the strategy is based on, ie.
difference of moving averages, macd, stochastic, etc.; actual value of
the indicator

signal - usually, based on indicator; at every bar shows where the
strategy 'wants' to be; as transactions cannot be executed on the same
bar where signal is generated, transaction will typically happen one
bar after 'signal' changed

blip - indicates that the strategy 'wants' to execute a trade; the
transaction will typically be executed one bar after the blip; zero
means no trade is required

transaction - the point where actual transaction is executed; for
performance calculation typically this price bar should be used (in
most typical scenario, signals are generated on a bar 'close' and
transactions executed on next bar's 'open'); zero means no trade is
required; transaction results in change in position on the same bar

position - at every price bar, indicates actual holding (direction
only, neither of those series are concerned with sizing)

'signal', 'blip', 'transaction', 'position' must only have one of
three values [-1, 0, 1] where -1 is short, 1 long and 0 flat.

Neither of those series are required in all dfs.  They are used on as
needed basis.  This terminology should be used consistently
throughout the package.

"""


def sig_blip(
    signal: pd.Series, clip=True, side: Optional[Literal["open", "close"]] = None
) -> pd.Series:
    """
    Signal to blip converter.

    if clip == False in case of 'always-in-the-market' systems, ie.
    signal never is zero will return -2 or 2.

    NOT TESTED
    """
    if signal.isna().any():
        raise ValueError("n/a values in signal")

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


def sig_pos(signal: pd.Series) -> pd.Series:
    """
    Convert signal to position (position is changed one bar after signal
    was generated).
    """
    return signal.shift().fillna(0).astype(int)


def blip_sig(blip: Union[pd.Series, pd.DataFrame], always_on=True) -> pd.Series:
    """
    Blip to signal converter. Numba optimized.

    Parameters:
    ----------
    blip:
        if pd.Series - the series represents both open and close signals

        if pd.DataFrame - first column is open signals, second column is close signals

    always_on:
        relevant only if blis is a Series;
        if True - close blip is simultanously an open blip for a reverse position.
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
            _in_out_signal_unifier(blip.to_numpy()).flatten(), index=blip.index
        )
    else:
        raise TypeError(f"Passed data must be a Series or DataFrame, not {type(blip)}")
