***************
Execution Module
***************

.. toctree::
    :maxdepth: 3

Haymaker trading algorithms consist of a series of Atom components, each implementing a step in a trading algorithm. These components are piped together in an event-driven fashion.

Common trading steps include:

* Receiving price data.
* Processing, aggregating, or filtering data.
* Generating trading signals.
* Managing the portfolio.
* Controlling risk.
* Managing execution.

Each processing component (called an "Atom") inherits from :class:`haymaker.base.Atom`.

Atom Object
===========

.. autoclass:: haymaker.base.Atom
    :members:

Auxiliary Objects
-----------------

.. autoclass:: haymaker.base.Pipe
    :members:

.. autoclass:: haymaker.base.Details
    :members:

Strategy Building Components
============================

Building on :class:`haymaker.base.Atom`, Haymaker offers skeletons of several components addressing typical requirements in building trading strategies:

* **Streamer**: Connects to the broker and pipes market data.
* **Aggregator**: Custom aggregation or processing of market data before signal generation.
* **Block**: Generates trading signals; this is the core building block of strategies (like blocks in a house).

  .. note:: If you have an idea for a better name, email me! :)

* **Signal Processor**: Filters or processes signals based on strategy state or auxiliary data, determining whether signals should trigger orders.
* **Portfolio**: A global object that receives processed signals and translates them into allocations (e.g., amounts of instruments to trade). It uses data like account value, holdings, volatility, risk targets, and concentration limits.
* **Execution Model**: Issues actual broker orders based on target instrument amounts.

Below is a review of how to use pre-built component modules:

Dataframe Persistence
---------------------

Configure dataframe persistence while building the strategy module. Use
``base.Atom.runtime.frame_store_provider`` and supply the Arctic library name
and symbol-naming policy for each datastore.

Choose the provider method according to how the datastore will be used:

``datastore()``
   Use the returned :class:`haymaker.datastore.AsyncDataStore` with
   :class:`haymaker.streamers.HistoricalDataStreamer` and
   :class:`haymaker.dfaggregator.DfAggregator`. Its methods are asynchronous;
   after a mutation returns from ``await``, the datastore operation has
   completed and any failure has been reported at that call.

``queued_sink()``
   Use the returned :class:`haymaker.datastore.QueuedDataSink` for optional
   dataframe-block output. Its ``enqueue_*`` methods return after accepting the
   write rather than after saving it. These writes are best-effort and pending
   writes may be discarded during shutdown.

Choose the symbol namer when constructing each store. Use
:class:`haymaker.datastore.BarSizeSymbolNamer` for market data shared by a
streamer and aggregator, and :class:`haymaker.datastore.StrategySymbolNamer`
for strategy block output. Naming cannot be changed after construction, so
create a separate store for every distinct naming policy, even when the stores
use the same Arctic library.

Streamer
--------

Haymaker provides streamers corresponding to all `ib_insync` market data subscriptions:

   +---------------------------+--------------------------+
   | ib_insync Method          | Streamer                 |
   +===========================+==========================+
   | reqHistoricalDataAsync    | HistoricalDataStreamer   |
   +---------------------------+--------------------------+
   | reqMktData                | MktDataStreamer          |
   +---------------------------+--------------------------+
   | reqRealTimeBars           | RealTimeBarsStreamer     |
   +---------------------------+--------------------------+
   | reqTickByTickData         | TickByTickStreamer       |
   +---------------------------+--------------------------+

All streamers extend :class:`haymaker.streamers.Streamer`.

.. autoclass:: haymaker.streamers.Streamer
   :members:

Implementations
^^^^^^^^^^^^^^^

Every implementation accepts the same arguments as the respective `ib_insync` method it wraps, plus standard :class:`haymaker.streamers.Streamer` parameters.

.. autoclass:: haymaker.streamers.HistoricalDataStreamer
   :members:

To make historical startup incremental, construct one bar-size-configured
datastore during strategy composition and inject it into the streamer. The
same datastore can be shared with a :class:`haymaker.dfaggregator.DfAggregator`
using the same bar size:

.. code-block:: python

   from haymaker import base, dfaggregator, streamers
   from haymaker.datastore import BarSizeSymbolNamer

   market_data = base.Atom.runtime.frame_store_provider.datastore(
       "market_data",
       symbol_namer=BarSizeSymbolNamer("30 secs"),
   )
   source = streamers.HistoricalDataStreamer(
       contract,
       "10 D",
       "30 secs",
       "TRADES",
       datastore=market_data,
   )
   aggregator = dfaggregator.DfAggregator(datastore=market_data)

When ``datastore`` is ``None``, the streamer requests history without reading a
saved boundary. Boolean datastore shortcuts are not supported.

.. autoclass:: haymaker.streamers.MktDataStreamer
   :members:

.. autoclass:: haymaker.streamers.RealTimeBarsStreamer
   :members:

.. autoclass:: haymaker.streamers.TickByTickStreamer
   :members:

Aggregator
----------

.. autoclass:: haymaker.aggregators.BarAggregator
   :members:

Available Filters
^^^^^^^^^^^^^^^^^

.. autoclass:: haymaker.aggregators.CountBars
   :members:

.. autoclass:: haymaker.aggregators.VolumeBars
   :members:

.. autoclass:: haymaker.aggregators.TickBars
   :members:

.. autoclass:: haymaker.aggregators.TimeBars
   :members:

.. autoclass:: haymaker.aggregators.NoFilter
   :members:

Block
-----

.. autoclass:: haymaker.block.AbstractBaseBlock
   :members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: haymaker.block.AbstractDfBlock
   :members:

Override :meth:`haymaker.block.AbstractDfBlock.df` when creating a concrete `AbstractDfBlock`.

Pass a fully configured :class:`haymaker.datastore.QueuedDataSink` through the
keyword-only ``datastore`` argument when a block needs an explicit persistence
dependency. Configure its symbol naming when constructing the store, for
example with :class:`haymaker.datastore.StrategySymbolNamer`; blocks
do not mutate injected stores. Construct the sink from
``base.Atom.runtime.frame_store_provider`` while composing the strategy module.
When ``datastore`` is omitted, block persistence is disabled.

For example, a strategy module can define a small helper and use one configured
sink per block:

.. code-block:: python

   from haymaker import base
   from haymaker.datastore import QueuedDataSink, StrategySymbolNamer

   def block_store(strategy: str) -> QueuedDataSink:
       return base.Atom.runtime.frame_store_provider.queued_sink(
           "block_data",
           symbol_namer=StrategySymbolNamer(strategy),
       )

   block = MyDataframeBlock(
       strategy="momentum_ES",
       contract=contract,
       datastore=block_store("momentum_ES"),
   )

Here ``MyDataframeBlock`` represents the strategy's concrete
:class:`haymaker.block.AbstractDfBlock` subclass.

Signal Processor
----------------

To better separate concerns, filter signals received from :class:`haymaker.block.AbstractBaseBlock` based on strategy state (e.g., avoid repeated signals or re-entering after a stop-out). While this could be done in :class:`haymaker.block.AbstractBaseBlock`, using a :class:`haymaker.signals.BinarySignalProcessor` is more modular.

Available processors are designed for binary signals (on/off switches). For discrete signals (e.g., 0–10 strength), these implementations aren’t suitable, but you can develop custom ones.

It’s easiest to create a binary signal processor by implementing :class:`haymaker.signals.AbstractBaseBinarySignalProcessor`.

.. autoclass:: haymaker.signals.AbstractBaseBinarySignalProcessor
   :members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: haymaker.signals.BinarySignalProcessor
   :members:

.. autoclass:: haymaker.signals.BlipBinarySignalProcessor
   :members:
      
.. autoclass:: haymaker.signals.LockableBinarySignalProcessor
   :members:

.. autoclass:: haymaker.signals.LockableBlipBinarySignalProcessor
   :members:

.. autoclass:: haymaker.signals.AlwaysOnLockableBinarySignalProcessor
   :members:

.. autoclass:: haymaker.signals.AlwaysOnBinarySignalProcessor
   :members:

Factory Function
^^^^^^^^^^^^^^^^

.. autofunction:: haymaker.signals.binary_signal_processor_factory

Portfolio
---------

.. autoclass:: haymaker.portfolio.AbstractBasePortfolio
   :members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: haymaker.portfolio.FixedPortfolio
   :members:

Wrapper
^^^^^^^

.. autoclass:: haymaker.portfolio.PortfolioWrapper
   :members:

An instance of :class:`haymaker.portfolio.AbstractBasePortfolio` should never be directly included in a processing pipeline, as there should be only one portfolio for all strategies. Instead, include an instance of :class:`haymaker.portfolio.PortfolioWrapper` in a pipeline. As long as an instance of :class:`haymaker.portfolio.AbstractBasePortfolio` exists in your package, :class:`haymaker.portfolio.PortfolioWrapper` ensures it’s connected.

Execution Model
---------------

It’s easiest to create an execution model by extending :class:`haymaker.execution_models.AbstractExecModel`.

.. autoclass:: haymaker.execution_models.AbstractExecModel
   :members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: haymaker.execution_models.BaseExecModel
   :members:
   :show-inheritance:

.. autoclass:: haymaker.execution_models.EventDrivenExecModel
   :members:
   :show-inheritance:

This model automatically places stop-loss orders (and potentially take-profit orders) when the original order is filled.

.. include:: example.rst
