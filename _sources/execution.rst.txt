****************
Execution Module
****************

.. toctree::
    :maxdepth: 3

Haymaker trading algorithms consist of a series of Atom components, each implementing a step in a trading algorithm. These components are piped together in an event-driven fashion.

Common trading steps include:

* Receiving price data.

* Processing/aggregating/filtering data.

* Generating trading signals.

* Portfolio management.

* Risk control.

* Execution management.

Each processing component (which is called: Atom) inherits from :class:`haymaker.base.Atom` .

Atom Object 
===========

.. autoclass:: haymaker.base::Atom
    :members:


Auxiliary Objects
-----------------

.. autoclass:: haymaker.base::Pipe
    :members:


.. autoclass:: haymaker.base::Details
    :members:


Strategy Building Components
============================

Building on :class:`Atom` `Haymaker` offers skeletons of several components addressing typical requirements in building trading strategies:

* Streamer - it's a connection to the broker that pipes market data
* Aggregator - component for custom aggregation or processing of market data required before they are used to generate signals
* Brick - trading signal generating component; this is what strategies are built with (like houses are built with bricks; if you have an idea for a better name let me know by email :)
* Signal Processor - not all signals must be used to generate orders; sometimes signals need filtering or further processing, this is what `Signal Processor` is for; it has access to information about state of strategies and potentially to any other auxiliary data required to determine whether signals should be acted on 
* Portfolio - whereaus all other comonents are meant to be created separately for every strategy/instrument, portfolio is a global object that receives all processed signals and translates them into allocations, i.e. actual amounts of instruments to be trades; `Portfolio` typically uses data such as current account value, current holdings, instrument volatility, risk targets, concentration limits, predetermined instrument weights and possibly many others
* Execution Model - given target instrument amount, issue actual order (or orders) that can be passed to broker

Below is a review how to use pre-built component modules:

Streamer
--------
There are objects corresponding to all `ib_insync` market data subsriptions, i.e.


   =========================     =========================
   ib_insync                     streamer
   =========================     =========================
   reqHistoricalDataAsync        HistoricalDataStreamer
   reqMktData                    MktDataStreamer
   reqRealTimeBars               RealTimeBarsStreamer
   reqTickByTickData             TickByTickStreamer
   =========================     =========================

All streamers extend :class:`Streamer`

.. autoclass:: haymaker.streamers::Streamer
   :members:

       
Implementations

Every implementation accepts the same arguments as respective `ib_insync` method, which it wraps plus standard :class:`Streamer` parameters.

.. autoclass:: haymaker.streamers::HistoricalDataStreamer
   :members:

.. autoclass:: haymaker.streamers::MktDataStreamer
   :members:

.. autoclass:: haymaker.streamers::RealTimeBarsStreamer
   :members:

.. autoclass:: haymaker.streamers::TickByTickStreamer
   :members:

      

       
Aggregator
----------

.. autoclass:: haymaker.aggregators::BarAggregator
   :members:

Available Filters
^^^^^^^^^^^^^^^^^
.. autoclass:: haymaker.aggregators::CountBars
   :members:
      
.. autoclass:: haymaker.aggregators::VolumeBars
   :members:
      
.. autoclass:: haymaker.aggregators::TickBars
   :members:
      
.. autoclass:: haymaker.aggregators::TimeBars
   :members:

.. autoclass:: haymaker.aggregators::NoFilter
   :members:



Brick
-----

.. autoclass:: haymaker.brick::AbstractBaseBrick
   :members:

Implementations

.. autoclass:: haymaker.brick::AbstractDfBrick
   :members:

Override :meth:`df` while creating a concrete `DfBrick`
      

Signal Processor
----------------
In order to better separate concerns, it might be useful to filter signals recevied from :class:`Brick` based on the state of the strategy, e.g. don't use repeated signals in the same direction or don't re-enter the same signal if the last one was stopped out. Of course it could be done directly in :class:`Brick`, but that may require frequently adding the same code to many Bricks. It's more modular to use :class:`SignalProcessor`.

Available processors are meant for binary signals - those that act on/off switches, if you have a discrete signal, where a number say in the range 0...10, signifies how strong the signal is, available implementations are not appropriate, but you're of course free to develop your own.

It's easies to create a binary signal processor by implementing :class:`AbstractBaseBinarySignalProcessor`

.. autoclass:: haymaker.signals::AbstractBaseBinarySignalProcessor
   :members:

Implementations

.. autoclass:: haymaker.signals::BinarySignalProcessor
   :members:


.. autoclass:: haymaker.signals::LockableBinarySignalProcessor
   :members:

      
.. autoclass:: haymaker.signals::LockableBlipBinarySignalProcessor
   :members:


.. autoclass:: haymaker.signals::AlwaysOnLockableBinarySignalProcessor
   :members:


.. autoclass:: haymaker.signals::AlwaysOnBinarySignalProcessor
   :members:


Factory Function
^^^^^^^^^^^^^^^^
.. autofunction:: haymaker.signals.binary_signal_processor_factory

          

Portfolio
---------

.. autoclass:: haymaker.portfolio::AbstractBasePortfolio
   :members:

Implementations

.. autoclass:: haymaker.portfolio::FixedPortfolio
   :members:

Wrapper
^^^^^^^

.. autoclass:: haymaker.portfolio::PortfolioWrapper
   :members:

An instance of `Portoflio` should never be directly included in a processing pipeline, because by it's nature, there should be only one portfolio for all strategies. Instead, you should include and instace of :class:`PortfolioWrapper` in a pipeline, and as long as there exists an instance of `Portfolio` in your package, :class:`PortfolioWrapper` will ensure that it's connected.

Execution Model
---------------
It's easiest to create an execution model by extending :class:`AbstractExecModel`

.. autoclass:: haymaker.execution_models.AbstractExecModel
   :members:

:class:`AbastractExecModel` Implementations


.. autoclass:: haymaker.execution_models.BaseExecModel
   :members:
   :show-inheritance:

.. autoclass:: haymaker.execution_models.EventDrivenExecModel
   :members:
   :show-inheritance:

This model will automatically place stop-loss orders and potentially
take-profit orders when the original order is filled.

      
.. include:: example.rst


