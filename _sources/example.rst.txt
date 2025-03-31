Example Usage 
=============

.. literalinclude:: includes/example.py 


.. warning::
    **NOT AN INVESTMENT ADVICE**

    This example is only meant to illustrate how to use `Haymaker` framework. It is unlikely to produce a favourable investment outcomes.


Example walk-through
--------------------

This is a simple example implementing `moving average crossover <https://en.wikipedia.org/wiki/Moving_average_crossover>`_ strategy with stop-loss.

The strategy:
* buys 1 `e-mini S&P futures contract ('ES') <https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html>`_ whenever faster exponential moving average crosses above slower one,
* sells 1 contract when faster ema crosses below slower ema.
* the moment position openning order is filled, strategy places `trailing stop loss order <https://www.investopedia.com/terms/t/trailingstop.asp>`_ with distance based on current instrument `Average True Range <https://www.investopedia.com/terms/a/atr.asp>`_.
* whenever stop-loss is hit, position will not be opened in the same direction as previously, an opposite position will have to openned and closed first. It's a protection against repeated openning and closing transaction in a volatitle non-trending market.
* position will be reversed whenever there's an signal with direction opposing the direction of the position held


.. code:: python

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


This is a definition of trading signals. We use :class:`brick.AbstractDfBrick`, which requires that we define a :py:`dataclasses.dataclass` with all strategy name, contract and all parameters that maybe required by the strategy. We need to override :meth:`df` method so that it accepts a dataframe with market data.

Typically market data are: Open, High, Low, Close, Volume, AveragePrice, but it depends on which Streamers and Processor we're using. Any data received via by `onData`, the class will wrap into  :pandas:`DataFrame` with collumn names corresponding to keys in `data` dictionary received by `onData` method. It's up to the user to connect this object to objects that will supply it with all data required to generate signals`

:meth:`df` must return a :pandas:`DataFrame` with a column `signal` containing values: 1 for long position, 0 for no position and -1 for short position. It should also contain any other columns as may be needed by other strategy components.

.. code:: python

          es_contract = ibi.ContFuture("ES", "CME")



:class:ib_insync.contract:`ContFuture` contract is not tradeable, however it can be used in defining strategies, as the framework will replace it current on-the-run future contract, when this contract is close to expiration, it will be rolled into next contract. How and when it exactly happens can be customized, refer to other parts of documentation.


.. code:: python
           
           portfolio.FixedPortfolio(1)

Typically, way more logic would be used in :class:`Portfolio` object, here we will just trade one contract whatever the circumstances. Don't do that at home.

.. code:: python

          
            pipe = base.Pipe(
                streamers.HistoricalDataStreamer(es_contract, "10 D", "1 hours", "TRADES"),
                aggregators.BarAggregator(processors.NoFilter()),
                EMACrossStrategy("ema_cross_ES", es_contract, 12, 48, 24),
                signals.BinarySignalProcessor(),
                portfolio.PortfolioWrapper(),
                execution_models.EventDrivenExecModel(
                    stop=bracket_legs.TrailingStop(3, vol_field="atr")
                ),
            )

:class:`base.Pipe` object connects all streams into an event-driven processing pipeline. Market data received by the streamer will be passed on to other components and every such event may or may not result in producing an order sent to broker.

So components used are:

.. code:: python
          
          streamers.HistoricalDataStreamer(es_contract, "10 D", "1 hours", "TRADES"),

Streamer will pull 10 days of 1 hour "TRADES" data for es_contract from the broker and this data will be subsequently updated whenever new data point arrives. If a disconnect or any other disruption occurs, the framework will strive to automatically fill in the missing data.


.. code:: python

          aggregators.BarAggregator(processors.NoFilter()),

No aggregating (nor other processing) is being done here (i.e. `NoFilter`), but in at the moment a aggregator is always required with a historical data streamer to keep record of historical data.


.. code:: python

          EMACrossStrategy("ema_cross_ES", es_contract, 12, 48, 24),

That's the instance of the class defined above instantiated with required parameters (picked absolutely arbitrarily here, i.e. we're using 12- and 48-hour ema and 24-hour atr).

.. code:: python

          signals.BinarySignalProcessor(),

:class:`BinarySignalProcessor` ensures that whenever it receives a signal, it:
       * make sure repeated signals in the same direction are ignored if the position is already in the market
       * after stop-loss is hit, will prevent strategy from openning another position in the same direction
       * when holding a position in the market, an opposing signal will issue a 'reverse' signal, which will close existing position and open an opposing one

.. code:: python

          portfolio.PortfolioWrapper(),

Since there can be globally only one Portfolio component, what we use here is not :class:`Portfolio` directly but a wrapper that will connect all strategies using it to one :class:`Portfolio` object, the wrapper will also ensure that any data returned from Porfolio concerning this strategy will be correctly connected to subsequent components. This is a necessary pattern at the moment, components have to use :class:`PortfolioWrapper` in places where they wish to connect to the Portfolio.
          
          
.. code:: python
          
          execution_models.EventDrivenExecModel(
              stop=bracket_legs.TrailingStop(3, vol_field="atr")

This particular implementation of :class:`BaseExecModel` will send a :ib_insync.order:`MarketOrder` to the broker and when this order is filled will send a trailing stop order with a distance to openning price of 3xATR ('atr' column has to be present in the data passed from :class:`Brick`).
  
   
.. code:: python


          if __name__ == "__main__":
              app.App().run()

The assumption here is that this will be run as a script directly, i.e. if all this code is in `strategy.py` file then we would start it with:

.. code:: bash

          python strategy.py

          
...where `python` obviously is the correct interpreter that we wish to run the script with.

But long-running scripts should be run as processes as explained <here: link to come>


Conclusion
----------

In real-life strategies, we would not trade just one instrument or one set of parameters, but using patterns illustrated above and typical python data structures, we can create pipelines for as many instruments and parameter sets as required.
