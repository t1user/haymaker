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

Each processing component (which is called: Atom) inherits from :class:`Atom` .

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

Building on :class:`Atom` `Haymaker` offers skeletons of several components addressing typical requirements in building trading strategies:

* Streamer - it's a connection to the broker that pipes market data
* Data Processor - component for custom aggregation or processing of market data required before they are used to generate signals
* Brick - trading signal generating component; this is what strategies are built with (like houses are built with bricks; if you have an idea for a better name let me know by email :)
* Signal Processor - not all signals must be used to generate orders; sometimes signals need filtering or further processing, this is what `Signal Processor` is for; it has access to information about state of strategies and potentially to any other auxiliary data required to determine whether signals should be acted on 
* Portfolio - whereaus all other comonents are meant to be created separately for every strategy/instrument, portfolio is a global object that receives all processed signals and translates them into allocations, i.e. actual amounts of instruments to be trades; `Portfolio` typically uses data such as current account value, current holdings, instrument volatility, risk targets, concentration limits, predetermined instrument weights and possibly many others
* Execution Model - given target instrument amount, issue actual order (or orders) that can be passed to broker

Below is a review how to use pre-built component modules:

Streamer
--------


Data Processor
--------------


Brick
-----


Signal Processor
----------------


Portfolio
---------


Execution Model
---------------


.. include:: example.rst


