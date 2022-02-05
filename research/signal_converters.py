import pandas as pd  # type: ignore
from numba_tools import _blip_to_signal_converter

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
needed basis.  However, this terminology is being used consistently
throughout the package to ensure consistent meaning in all files.

"""


def sig_blip(signal: pd.Series, clip=True) -> pd.Series:
    """
    Signal to blip converter.

    if clip == False in case of 'always-in-the-market' systems, ie.
    signal is never zero will return -2 or 2.
    """

    o = (signal - signal.shift()).fillna(0)
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
    o = (position.shift() != position) * \
        (position - position.shift()).fillna(0)
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


def blip_sig(blip: pd.Series) -> pd.Series:
    """Blip to signal converter. Numba optimized."""
    return pd.Series(_blip_to_signal_converter(blip.to_numpy()),
                     index=blip.index)
