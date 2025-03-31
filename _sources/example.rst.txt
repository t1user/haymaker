Example Usage
=============

.. literalinclude:: includes/example.py
   :language: python

.. warning::
   **NOT INVESTMENT ADVICE**

   This example is only meant to illustrate how to use the Haymaker framework. It is unlikely to produce favorable investment outcomes.

Example Walk-Through
--------------------

This is a simple example implementing a `moving average crossover <https://en.wikipedia.org/wiki/Moving_average_crossover>`_ strategy with a stop-loss.

The strategy:

* Buys 1 `e-mini S&P futures contract ('ES') <https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html>`_ whenever the faster exponential moving average (EMA) crosses above the slower one.
* Sells 1 contract when the faster EMA crosses below the slower EMA.
* The moment a position-opening order is filled, places a `trailing stop-loss order <https://www.investopedia.com/terms/t/trailingstop.asp>`_ with a distance based on the current instrument's `Average True Range <https://www.investopedia.com/terms/a/atr.asp>`_.
* Whenever the stop-loss is hit, prevents reopening a position in the same direction until an opposite position is opened and closed. This protects against repeated transactions in a volatile, non-trending market.
* Reverses the position when a signal indicates a direction opposite to the current position held.

.. code-block:: python
   :caption: Defining the EMA crossover strategy

   from dataclasses import dataclass
   import pandas as pd
   import numpy as np
   from haymaker import brick, indicators
   import ib_insync as ibi

   @dataclass
   class EMACrossStrategy(brick.AbstractDfBrick):
       strategy: str
       contract: ibi.Contract
       fast_lookback: int
       slow_lookback: int
       atr_lookback: int

       def df(self, df: pd.DataFrame) -> pd.DataFrame:
           df["fast_ema"] = df["close"].ewm(self.fast_lookback).mean()
           df["slow_ema"] = df["close"].ewm(self.slow_lookback).mean()
           df["signal"] = np.sign(df["fast_ema"] - df["slow_ema"])
           df["atr"] = indicators.atr(df, self.atr_lookback)
           return df

This defines the trading signals using :class:`haymaker.brick.AbstractDfBrick`. It requires a :py:class:`dataclasses.dataclass` with the strategy name, contract, and parameters. The :meth:`haymaker.brick.AbstractDfBrick.df` method must be overridden to process a :class:`pandas.DataFrame` containing market data (e.g., Open, High, Low, Close, Volume, AveragePrice), depending on the connected streamers and processors.

The data received via `onData` is wrapped into a :class:`pandas.DataFrame` with column names matching the keys in the `data` dictionary. Users must ensure upstream components provide all required data for signal generation.

The :meth:`haymaker.brick.AbstractDfBrick.df` method must return a :class:`pandas.DataFrame` with a ``signal`` column: 1 for long, 0 for no position, -1 for short. Additional columns (e.g., ``atr``) can be included for downstream components.

.. code-block:: python
   :caption: Defining the ES futures contract

   es_contract = ibi.ContFuture("ES", "CME")

The :class:`ib_insync.contracts.ContFuture` contract is not directly tradable. Haymaker replaces it with the current on-the-run futures contract and rolls it to the next contract near expiration. Refer to other documentation sections for customization details.

.. code-block:: python
   :caption: Setting a fixed portfolio size

   from haymaker import portfolio
   portfolio.FixedPortfolio(1)

Typically, a :class:`haymaker.portfolio.FixedPortfolio` would include more logic. Here, it trades one contract regardless of circumstances—a simplistic approach not recommended for real use.

.. code-block:: python
   :caption: Assembling the pipeline

   from haymaker import base, streamers, aggregators, signals, execution_models, portfolio, bracket_legs

   pipe = base.Pipe(
       streamers.HistoricalDataStreamer(es_contract, "10 D", "1 hour", "TRADES"),
       aggregators.BarAggregator(aggregators.NoFilter()),
       EMACrossStrategy("ema_cross_ES", es_contract, 12, 48, 24),
       signals.BinarySignalProcessor(),
       portfolio.PortfolioWrapper(),
       execution_models.EventDrivenExecModel(
           stop=bracket_legs.TrailingStop(3, vol_field="atr")
       ),
   )

The :class:`haymaker.base.Pipe` connects components into an event-driven pipeline. Market data from the streamer triggers processing, potentially resulting in broker orders.

Components used:

- .. code-block:: python
     :caption: Historical data streamer

     streamers.HistoricalDataStreamer(es_contract, "10 D", "1 hour", "TRADES")

  Pulls 10 days of 1-hour "TRADES" data for ``es_contract`` from the broker, updating with new data points. The framework automatically fills gaps if disruptions occur.

- .. code-block:: python
     :caption: No-op aggregator

     aggregators.BarAggregator(aggregators.NoFilter())

  No aggregation or processing is applied (using :class:`haymaker.aggregators.NoFilter`). An aggregator is currently required with historical data streamers to track history.

- .. code-block:: python
     :caption: EMA crossover strategy instance

     EMACrossStrategy("ema_cross_ES", es_contract, 12, 48, 24)

  Instantiates the strategy with arbitrary parameters: 12-hour and 48-hour EMAs, 24-hour ATR.

  .. note::
     These parameters are illustrative and not optimized.

- .. code-block:: python
     :caption: Binary signal processor

     signals.BinarySignalProcessor()

  The :class:`haymaker.signals.BinarySignalProcessor` ensures:
  * Repeated signals in the same direction are ignored if a position exists.
  * Post-stop-loss, prevents reopening in the same direction until an opposite position is cycled.
  * Reverses positions on opposing signals.

- .. code-block:: python
     :caption: Portfolio wrapper

     portfolio.PortfolioWrapper()

  Connects to a single global :class:`haymaker.portfolio.FixedPortfolio` instance, ensuring strategy-specific data flows correctly to downstream components.

- .. code-block:: python
     :caption: Event-driven execution model

     execution_models.EventDrivenExecModel(
         stop=bracket_legs.TrailingStop(3, vol_field="atr")
     )

  Sends a :class:`ib_insync.order.MarketOrder` to the broker. Upon fill, places a trailing stop-loss order with a distance of 3×ATR (requires an ``atr`` column from the :class:`haymaker.brick.AbstractDfBrick`).

.. code-block:: python
   :caption: Running the application

   from haymaker import app

   if __name__ == "__main__":
       app.App().run()

Run this as a script (e.g., ``strategy.py``):

.. code-block:: bash
   :caption: Starting the strategy

   python strategy.py

Ensure ``python`` is the correct interpreter. Long-running scripts should be managed as processes (see process management documentation—link TBD).

Conclusion
----------

In real-world strategies, you’d trade multiple instruments and parameter sets. Using the patterns above with Python data structures, you can create pipelines for as many combinations as needed.