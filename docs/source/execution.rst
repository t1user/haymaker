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
* **Brick**: Generates trading signals; this is the core building block of strategies (like bricks in a house).

  .. note:: If you have an idea for a better name, email me! :)

* **Signal Processor**: Filters or processes signals based on strategy state or auxiliary data, determining whether signals should trigger orders.
* **Portfolio**: A global object that receives processed signals and translates them into allocations (e.g., amounts of instruments to trade). It uses data like account value, holdings, volatility, risk targets, and concentration limits.
* **Execution Model**: Issues actual broker orders based on target instrument amounts.

Below is a review of how to use pre-built component modules:

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

Brick
-----

.. autoclass:: haymaker.brick.AbstractBaseBrick
   :members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: haymaker.brick.AbstractDfBrick
   :members:

Override :meth:`haymaker.brick.AbstractDfBrick.df` when creating a concrete `AbstractDfBrick`.

Signal Processor
----------------

To better separate concerns, filter signals received from :class:`haymaker.brick.AbstractBaseBrick` based on strategy state (e.g., avoid repeated signals or re-entering after a stop-out). While this could be done in :class:`haymaker.brick.AbstractBaseBrick`, using a :class:`haymaker.signals.BinarySignalProcessor` is more modular.

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
