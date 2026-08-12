======================================
Event-driven Backtester (Experimental)
======================================

.. warning::

   ``haymaker.backtester`` is an experimental extra. It is not fully functional,
   its API may change, and its results should not be used as production evidence.

The event-driven backtester replays saved broker bars through Haymaker strategy
components. Its purpose is to exercise the same Atom graph, messages, portfolio,
execution models, controller order path, and Book accounting used by a live
strategy while replacing Interactive Brokers and external persistence with
simulation adapters.

This is distinct from the :doc:`research/backtester`, which evaluates a prepared
pandas transaction frame without running the live-style component graph.

Basic usage
===========

Construct a fresh strategy graph for each run and pass its factory to
``Backtester.run``:

.. code-block:: python

   from datetime import datetime, timezone

   from haymaker.backtester import BacktestDataStore, Backtester

   store: BacktestDataStore = ...

   def strategy_factory() -> None:
       # Construct and connect the strategy's Haymaker components here.
       ...

   result = await Backtester(
       store,
       start=datetime(2025, 1, 1, tzinfo=timezone.utc),
       end=datetime(2025, 2, 1, tzinfo=timezone.utc),
       initial_cash=100_000.0,
       slippage_ticks=1.0,
   ).run(strategy_factory)

``Backtester`` installs the simulated runtime before it invokes
``strategy_factory``. ``start`` and ``end`` restrict the replay range;
``initial_cash`` sets the simulated opening balance; and ``slippage_ticks``
applies signed slippage to each fill. The API is experimental and may change.

The supplied ``store`` must implement the smaller, read-only
``BacktestDataStore`` protocol: awaited ``keys()``, ``read()``, and
``read_metadata()`` methods. A normal
:class:`~haymaker.datastore.AsyncDataStore` satisfies this protocol. Use a store
configured for the same library naming as the dataloader: ``whatToShow`` plus
bar size with spaces replaced by underscores, for example ``TRADES_1_min`` or
``MIDPOINT_30_secs``. The normal collection key is
``<localSymbol>_<secType>`` (for example ``ESH5_FUT``), but the reader first
enumerates string keys and therefore preserves a store's configured naming
policy. A run does not mutate the supplied store.

Disable live persistence when adapting a strategy factory. In particular,
``HistoricalDataStreamer(datastore=True)``,
``PandasSignalModel(persistence=True)``, and strategy-created frame stores fail
explicitly during a replay. The datastore passed to ``Backtester`` is the sole
market-data input and all Book state remains in memory.

Data and metadata
=================

The datastore is the authority for both bars and contract-specific facts. Each
series must retain enough metadata to reconstruct its exact qualified
``ib_insync.Contract``, including a non-zero ``conId``. In particular, futures
expiry comes from an exact ``YYYYMMDD``
``lastTradeDateOrContractMonth`` in metadata; the simulator does not query IB to
fill missing contract details.

The following execution inputs also come from series metadata:

* ``multiplier`` scales realized and unrealized PnL and defaults to one.
* ``commission`` is interpreted as cost per filled unit and defaults to zero.
* A positive ``minTick`` is required when tick-based slippage or fixed bracket
  prices are enabled. The run should fail clearly rather than guess a tick size.

Saved frames must contain the standard dataloader ``open``, ``high``, ``low``,
``close``, ``volume``, ``average``, and ``barCount`` columns. Intraday indexes
must be timezone-aware UTC datetimes; daily indexes must contain calendar dates.
``start`` limits the replay clock but does not discard earlier rows needed for
component warmup.

Trading sessions are inferred from the saved bars, not from exchange-hours
metadata. At a replay timestamp, a Contract with a bar is considered to have a
session; a Contract without a bar is considered closed. Its pending orders wait
for its next saved bar.

Timing and fills
================

Bars are treated as completed observations. An order created while processing a
bar cannot use that bar's open, high, low, or close. It first becomes eligible on
the Contract's next saved bar. This is the core no-lookahead rule.
An order deliberately created by ``onStart`` predates the first bar and may fill
on that first observed Contract session.

The experimental simulator supports complete fills for:

* ``MKT`` orders at the next eligible bar's open;
* ``LMT`` and ``STP`` orders when a later eligible bar reaches their price; and
* fixed stop-loss/take-profit brackets linked through one-cancels-all (OCA),
  where filling one exit cancels the remaining linked exit.

Limit and stop results use bar OHLC data, so they cannot reconstruct the path of
prices within a bar. The simulator treats the open as occurring first. A market
close therefore precedes a protective stop touched later in the same bar. If a
fixed stop and take-profit both touch, the stop wins. Limit fills are capped at
their limit even when slippage is configured. Avoid interpreting a backtest as
tick-accurate execution.

Only omitted or ``GTC`` time-in-force is accepted. Broker algorithms,
conditions, parent/child and staged-transmission flags, what-if orders, deferred
activation, timed expiry, and advanced quantity/display modifiers fail
explicitly because their semantics are not modeled. In particular, do not pass
the live Adaptive/``DAY`` opening-order defaults to a backtest. ``outsideRth``
has no independent effect: the presence of a stored bar is the only session
signal.

Results
=======

``BacktestResult`` exposes final ``contracts``, ``positions``, every submitted
``order``, still-active ``working_orders``, and completed ``fills``. A market
order submitted by the final bar remains visible in ``working_orders`` rather
than disappearing from a zero-fill result.

``initial_cash`` is a reporting baseline, not a risk control. The simulator does
not enforce affordability, margin, or buying power. ``ending_cash`` applies
fill notional to non-futures and a simple realized-PnL variation-margin model to
futures; ``equity`` remains opening cash plus marked net PnL. These account
figures do not model an IB account's margin rules.

Current limitations
===================

The following live behaviors are not simulated:

* trailing or adjustable orders;
* partial fills;
* rolling an already-open futures position;
* live broker connection supervision, startup synchronization, reconciliation,
  or recovery; and
* broker schedule or contract-detail requests.

One strategy and one simulated runtime per process is the supported shape.
``Signal.as_of`` and fill times use replay chronology, while message creation and
Book audit timestamps still use the wall clock of the backtest run.

Use focused test runs and inspect orders, fills, positions, and warnings before
relying on any aggregate result.
