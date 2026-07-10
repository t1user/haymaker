===============
Result Analysis
===============

Result-analysis helpers operate on completed trades returned by the research
backtester. They do not alter transaction timing or recalculate strategy PnL.

Trade Excursions
================

:func:`haymaker.research.excursions` measures each trade's maximum favorable
excursion (the largest unrealized gain offered while the trade was open) and
maximum adverse excursion (the largest unrealized loss experienced while it was
open). These measurements are useful when investigating stop distances, profit
targets, and how efficiently the strategy turns available price movement into
realized PnL.

Excursions describe completed trades; they do not simulate alternative exits.
In particular, bar data does not establish whether the favorable or adverse
extreme occurred first within a bar.

An optional positive ``divisor`` can express PnL and excursions in volatility,
risk, or other entry-time units. The divisor must have exactly the same index as
the OHLC dataframe; its value on the trade's entry bar is used.

.. code-block:: python

   from haymaker.research import excursions

   metrics = excursions(
       bars[["high", "low"]],
       result.positions,
       divisor=bars["atr"],
   )
   trades = result.positions.join(metrics)

.. autofunction:: haymaker.research.excursions

.. autofunction:: haymaker.research.winning_trade_adverse_excursions

Entry-Time Factors
==================

:func:`haymaker.research.factor_extractor` attaches entry-time context such as a
forecast, ATR, volume, or regime label to each completed trade. The resulting
table can be grouped or filtered to determine which entry conditions are
associated with stronger or weaker results.

By default the source fields are shifted by one row so values calculated from
the entry bar cannot leak into the analysis. Use ``shift=False`` only when a
value was already available at the entry timestamp.

.. code-block:: python

   from haymaker.research import factor_extractor

   trades = factor_extractor(
       result.positions,
       bars,
       ["forecast", "atr"],
   )

.. autofunction:: haymaker.research.factor_extractor
