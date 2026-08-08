*****************
Live Components
*****************

Haymaker strategies are ordinary Python modules that compose
:class:`~haymaker.base.Atom` objects after the live runtime has installed a
ready ``RuntimeContext``. ``Atom`` is intentionally
general: custom Atoms may pass any object. The built-in trading toolbox is
available from ``haymaker.components`` and uses structured immutable
messages.

Atom composition
================

Every concrete Atom implements ``onData`` and explicitly emits through
``dataEvent``. Base ``onData`` raises ``NotImplementedError``. Startup remains
synchronous and receives arbitrary mutable data through
``onStart(data, source)``; Haymaker does not add strategy names, timestamps, or
reserved fields.

``source.connect(*targets)`` validates every target before changing any event
connections. A component overrides ``validate_source`` only when an upstream
class is structurally incompatible. The override returns normally for a valid
source and raises ``TypeError`` or a more specific domain exception for an
invalid source; returning a boolean does not reject a connection.
Value-dependent checks happen in ``onData``.

Fan-out sends the same object reference to every branch. Immutable standard
messages are safe to share. A custom branch that mutates its input must copy it
first.

Atoms are stateful graph nodes rather than value records. A dataclass-based
Atom must therefore use ``@dataclass(eq=False)``. Repeat ``eq=False`` on every
subclass decorated with ``@dataclass`` because a new dataclass decoration would
otherwise generate value equality again.

.. autoclass:: haymaker.base.Atom
   :members: connect, disconnect, clear, pipe, validate_source, onStart, onData, onFeedback

.. autoclass:: haymaker.base.Pipe
   :members:

Trading messages
================

The standard message sequence is:

.. code-block:: text

   Signal -> PositionProposal -> PositionTarget

:class:`~haymaker.components.Signal` identifies one logical source, a Contract,
a finite scalar value or :class:`~haymaker.components.SignalPair`, and
mandatory :class:`~haymaker.components.SignalType`.
``STATE`` replaces the source's previous desired state. Each ``EVENT`` is a new
event; an EVENT zero means that no event occurred. ``as_of`` is the optional
effective market-observation time, while ``created_at`` records local Signal
creation. They deliberately remain distinct: IB bars are left-labelled, so a
bar's ``as_of`` identifies the start of its interval rather than the time at
which the completed bar was processed.
``SignalPair`` carries separate entry and exit values without duplicating the
Signal's identity, Contract, timestamps, or metadata.

:class:`~haymaker.components.PositionProposal` is used only by one-to-one
processors. It preserves the original Signal, chooses short/flat/long
direction, and includes mandatory OPEN/CLOSE/REVERSE intent.

:class:`~haymaker.components.PositionTarget` is an absolute signed setpoint for
a concrete Contract. It never contains a captured current quantity or a
proposed order delta. The numeric target is authoritative. Intent is optional
for general direct execution and mandatory only for the one-to-one bracket
path.

.. autoclass:: haymaker.components.Signal

.. autoclass:: haymaker.components.SignalPair

.. autoclass:: haymaker.components.SignalType

.. autoclass:: haymaker.components.PositionProposal

.. autoclass:: haymaker.components.PositionTarget

.. autoclass:: haymaker.components.PositionIntent

.. autoclass:: haymaker.components.StandardOrderRole

Market-data components
======================

Streamers own Interactive Brokers subscriptions and emit broker objects while
the runtime is connected. Historical bars can feed the bar-object aggregation
family or the separate DataFrame aggregation family. Streamers are registered
process-wide during strategy import and started by the runtime.

.. autoclass:: haymaker.components.Streamer

.. autoclass:: haymaker.components.HistoricalDataStreamer

.. autoclass:: haymaker.components.MktDataStreamer

.. autoclass:: haymaker.components.RealTimeBarsStreamer

.. autoclass:: haymaker.components.TickByTickStreamer

Event inactivity monitoring
---------------------------

:class:`~haymaker.components.EventTimeout` monitors any ``eventkit.Event`` and
calls a user callback once when one inactivity interval expires. Every source
emission moves the deadline. After a timeout, fresh source data rearms the next
episode. General event timeouts are user-owned: create positive intervals on
the running asyncio loop and call ``cancel()`` when the owning object ends.
Haymaker does not cancel them during a supervised workload restart.

:class:`~haymaker.components.MarketDataTimeout` adds a Contract's trading
session and supervisor restart policy. Streamers install it automatically.
Custom market-data Atoms should call ``MarketDataTimeout.from_atom()`` from
``onStart()`` or later, once Contract details and the supervisor callback are
available. A closed session pauses monitoring until the next open and then
starts a complete interval. Log-only mode waits for fresh data before rearming;
restart mode requests one workload rebuild and stays disarmed while that
lifecycle transition is handled.

Generic timeout in a custom Atom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This pass-through Atom monitors its own output event. The positive timeout is
created from ``onStart()``, when the asyncio loop is running. It is created
only once because a general ``EventTimeout`` belongs to the component owner
and is not reset by supervised workload restarts:

.. code-block:: python

   import logging
   from typing import Any

   from haymaker.base import Atom
   from haymaker.components import EventTimeout


   log = logging.getLogger(__name__)


   class HeartbeatGuard(Atom):
       """Pass data through and report when output becomes inactive."""

       def __init__(self, seconds: float) -> None:
           super().__init__()
           self.seconds = seconds
           self._timeout: EventTimeout | None = None

       def onStart(
           self,
           data: Any,
           source: Atom | None = None,
       ) -> None:
           if self._timeout is None:
               self._timeout = EventTimeout(
                   self.dataEvent,
                   self.seconds,
                   callback=self.on_stale,
                   name=f"{type(self).__name__} output",
               )
           super().onStart(data, source)

       def onData(self, data: Any, *args: Any) -> None:
           self.dataEvent.emit(data)

       def on_stale(self) -> None:
           log.warning("No output for %s seconds", self.seconds)

       def close(self) -> None:
           if self._timeout is not None:
               self._timeout.cancel()
               self._timeout = None


   guard = HeartbeatGuard(seconds=30)

Call ``guard.close()`` when the application-specific owner releases the
component. A callback may also be an ``async def`` function. After one timeout,
the callback is not repeated until ``dataEvent`` emits again and begins a new
inactivity episode.

Market-aware timeout in a custom Atom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``MarketDataTimeout.from_atom()`` when stale data should be interpreted
using the Atom's Contract session and Haymaker's supervisor policy. Construct
it on every ``onStart()``: Haymaker cancels the previous workload's
market-data timeouts before the next workload generation starts.

.. code-block:: python

   from typing import Any

   import eventkit as ev
   import ib_insync as ibi

   from haymaker.base import Atom
   from haymaker.components import MarketDataTimeout


   class CustomMarketFeed(Atom):
       """Forward one external market-data event into an Atom pipeline."""

       def __init__(
           self,
           contract: ibi.Contract,
           update_event: ev.Event,
       ) -> None:
           super().__init__()
           self.contract = contract
           self.update_event = update_event
           self.update_event += self.onUpdateEvent
           self._timeout: MarketDataTimeout | None = None

       def onStart(
           self,
           data: Any,
           source: Atom | None = None,
       ) -> None:
           self._timeout = MarketDataTimeout.from_atom(
               self,
               self.update_event,
               key="quotes",
           )
           super().onStart(data, source)

       def onUpdateEvent(self, update: Any) -> None:
           self.dataEvent.emit(update)


   updates = ev.Event("vendor quotes")
   feed = CustomMarketFeed(
       ibi.Future("ES", exchange="CME"),
       updates,
   )

Omitting ``seconds`` uses the runtime ``timeout:`` configuration. Passing
``seconds=15`` overrides only the interval; the configured ``restart`` or
``log`` action still applies. ``from_atom()`` requires qualified Contract
details and the bound supervisor callback, so call it from ``onStart()`` or
later rather than during module-level pipeline construction.

.. autoclass:: haymaker.components.EventTimeout
   :members: arm, cancel

.. autoclass:: haymaker.components.MarketDataTimeout
   :members: from_atom, cancel

Bar-object aggregation
----------------------

:class:`~haymaker.components.BarAggregator` incrementally feeds completed
``BarData`` objects through an eventkit bar filter and emits the resulting
``BarDataList``. Use this family when downstream components should continue to
operate on IB bar objects.

.. autoclass:: haymaker.components.BarAggregator

.. autoexception:: haymaker.components.aggregators.WrongStreamer

Bar filters
~~~~~~~~~~~

The filter objects below group source bars and retain their output
``BarDataList``. ``NoFilter`` preserves one output bar per input bar.

.. autoclass:: haymaker.components.CountBars

.. autoclass:: haymaker.components.VolumeBars

.. autoclass:: haymaker.components.TickBars

.. autoclass:: haymaker.components.TimeBars

.. autoclass:: haymaker.components.NoFilter

DataFrame aggregation
---------------------

:class:`~haymaker.components.FuturesPandasAggregator` is the parallel,
futures-only DataFrame pipeline. It combines current ``BarDataList`` snapshots
with stored history, acquires missing previous futures contracts, stitches a
continuous series, periodically saves it, and emits the complete DataFrame. By
default it resolves the runtime store configured by
``market_data_store.library`` after connecting to its streamer. Set
``HistoricalDataStreamer(datastore=True)`` to use that same cached store for
persisted-endpoint lookup and shorter subsequent IB requests. The aggregator
adopts the request settings from its connected streamer, so both components
resolve the same store.

The default store identity includes the Contract, bar size, ``whatToShow``, and
``useRTH`` policy, so incompatible historical series cannot share one symbol.
``FuturesPandasAggregator`` does not support disabling its datastore; use
``save_frequency=0`` only to disable periodic saves, not history reads or
broker-backfill writes.

The DataFrame components are public from both
``haymaker.components.dataframe_aggregators`` and the package-level
``haymaker.components`` toolbox.

.. autoclass:: haymaker.components.FuturesPandasAggregator

.. autoclass:: haymaker.components.VolumeGrouper

.. autoexception:: haymaker.components.dataframe_aggregators.MissingStreamerParam

Signal models
=============

:class:`~haymaker.components.SignalModel` is the general structured Signal
producer. A custom model implements ``calculate_signal(data)`` and returns a
:class:`~haymaker.components.SignalCalculation` containing only its calculated
value, optional metadata, and optional observation time. The framework supplies
the configured source, resolved Contract, SignalType, and ``created_at`` when it
constructs the immutable Signal. Override ``validate_signal_value()`` to impose
model-specific restrictions beyond the standard finite-value checks.

.. code-block:: python

   class ThresholdModel(SignalModel):
       threshold = 100.0

       def calculate_signal(self, observation):
           return SignalCalculation(
               value=1 if observation.price > self.threshold else 0,
               metadata={"threshold": self.threshold},
               as_of=observation.time,
           )

:class:`~haymaker.components.PandasSignalModel` provides dataframe
conveniences: implement ``df(data)`` and return the complete calculated
dataframe. By default, ``signal_fields="signal"`` selects a scalar from the
last row. A two-field tuple such as ``signal_fields=("in", "out")`` creates
``SignalPair(entry=..., exit=...)``. The index becomes ``as_of`` when
datetime-like. ``metadata_fields=None`` copies all non-signal fields into
metadata, an empty collection copies none, and an explicit collection selects
only those fields. The last returned row is authoritative: the user calculation
owns ordering, duplicate handling, and correctness.

Override ``row_to_calculation(row)`` when the standard field selection cannot
express the required conversion. The override returns only
``SignalCalculation``; it cannot replace framework-owned Signal identity.
``persistence=False`` disables calculation history. ``persistence=True`` uses
the runtime default configured by ``signal_persistence.library``. The standard
implementation records history under
``{source_key}_{ACTIVE.localSymbol}_{run_started_at}``: the first accepted save
writes the complete frame and later saves append only new rows. A NEXT-only
futures change does not rotate history.

Persistence work is queued before Signal emission so the accepted
``audit_symbol`` can be included in metadata; no storage I/O is awaited. If the
queue cannot accept the work, Haymaker logs the failure and emits the Signal
without that reference. Calling ``create_signal()`` directly performs no
persistence. When saving is enabled, the calculated dataframe must not be
mutated after queue acceptance.

Custom market-data stores and Signal persistence policies are described in
:ref:`advanced-storage-configuration`.

.. autoclass:: haymaker.components.SignalCalculation

.. autoclass:: haymaker.components.SignalModel
   :members: calculate_signal, create_signal, validate_signal_value

.. autoclass:: haymaker.components.PandasSignalModel
   :members: df, row_to_calculation, calculate_signal, create_signal

One-to-one processing
=====================

Binary processors consume Signal and query Book's effective quantity,
including relevant working orders. Values must be exactly ``-1``, ``0``, or
``1``. For scalar input, STATE zero requests flat and EVENT zero is ignored.
Matching direction is suppressed. ``OpposingSignalPolicy.CLOSE`` closes first;
``REVERSE`` reverses immediately.

``BinaryEntryExitSignalProcessor`` consumes a SignalPair. It consults only the
entry value while flat and only the exit value while positioned. An opposing
exit closes and never reverses directly. STATE exit zero closes; EVENT exit
zero is ignored.

Either processor can set ``respect_blocked_direction=True``. A protective
STOP_LOSS or TAKE_PROFIT that completely flattens a position establishes the
block. Opening in the blocked direction is suppressed, opening in the opposite
direction is allowed, and its first actual fill clears the old block. Ordinary
CLOSE and ROLL fills do not alter it.

.. autoclass:: haymaker.components.BinarySignalProcessor
   :members: process

.. autoclass:: haymaker.components.BinaryEntryExitSignalProcessor
   :members: process

.. autoclass:: haymaker.components.OpposingSignalPolicy

Portfolio boundaries
====================

The dedicated one-to-one flow uses a processor, allocator wrapper, and bracket
model:

.. code-block:: text

   SignalModel
       -> BinarySignalProcessor
       -> PortfolioWrapper(FixedSizeAllocator)
       -> BracketExecutionModel

``PortfolioWrapper`` accepts only PositionProposal, calls the narrow
``PositionAllocator.target_for()`` protocol, and emits at most one target while
preserving source, Contract, metadata, and intent.

.. autoclass:: haymaker.components.PositionAllocator

.. autoclass:: haymaker.components.FixedSizeAllocator
   :members: target_for

.. autoclass:: haymaker.components.PortfolioWrapper
   :members: onData

Account-wide allocation uses a direct Portfolio:

.. code-block:: text

   multiple Signal paths
       -> Portfolio
       -> ExecutionRouter
       -> SerialTargetExecutionModel

Implement ``process(signal)`` to update Portfolio state and return zero or more
absolute targets. The optional ``sources`` collection rejects unknown source
keys. The base class deliberately does not batch, debounce, time out, or
interpret ``as_of``; concrete policies own those decisions.

.. autoclass:: haymaker.components.Portfolio
   :members: process

Execution and routing
=====================

Execution models have stable configured names used for recovery. They persist
the newest target, inspect Book quantity and working orders, and derive the
next order rather than replaying stale intent.

:class:`~haymaker.components.SerialTargetExecutionModel` manages each concrete
Contract independently, supports arbitrary same-side resizing, and permits at
most one active TARGET_ADJUSTMENT order per Contract.

:class:`~haymaker.components.BracketExecutionModel` manages one configured
``source_key``. Initial targets require consistent intent, same-side non-zero
resizing is rejected, and each opening episode gets a fresh ``position_id``.
Protective brackets are attached only after the entry is completely filled.
The stop-loss is critical protection for every established episode; a
take-profit is optional and its absence is not a synchronization failure.
Regular CLOSE orders join the active brackets' OCA group, allowing IB to cancel
the unfilled protective orders when any exit fills.

.. autoclass:: haymaker.components.ExecutionModel

.. autoclass:: haymaker.components.SerialTargetExecutionModel

.. autoclass:: haymaker.components.BracketExecutionModel

Router rules are fixed and evaluated in declaration order; first match wins.
Without a default, unmatched targets fail closed. Working orders retain their
persisted model affinity until terminal and require that named model during
recovery. Held quantity and idle targets do not pin an old model: current rules
take ownership, and startup reassigns each idle direct target before model
recovery.

.. autoclass:: haymaker.components.ExecutionRule

.. autoclass:: haymaker.components.ExecutionRouter

.. autofunction:: haymaker.components.contract_is

.. autofunction:: haymaker.components.symbol_is

.. autofunction:: haymaker.components.security_type_is

.. autofunction:: haymaker.components.exchange_is

.. autofunction:: haymaker.components.where

Bracket execution
=================

``BracketExecutionModel`` and its bracket legs form one dedicated execution
family. The legs convert a completely filled entry Trade and validated Signal
metadata into IB stop or take-profit order fields. Their ``vol_field`` defaults
to ``atr`` and must be present in PositionTarget metadata.

.. autoclass:: haymaker.components.AbstractBracketLeg

.. autoclass:: haymaker.components.FixedStop

.. autoclass:: haymaker.components.TrailingStop

.. autoclass:: haymaker.components.AdjustableTrailingFixedStop

.. autoclass:: haymaker.components.AdjustableFixedTrailingStop

.. autoclass:: haymaker.components.AdjustableTrailingStop

.. autoclass:: haymaker.components.TakeProfitAsStopMultiple

.. autoclass:: haymaker.components.FlexibleTakeProfitAsStopMultiple

Book and Controller ownership
=============================

:class:`~haymaker.book.Book` owns typed order, Fill, PositionState, TargetState,
Portfolio recovery state, stopped-direction state, and blotter access. It uses
one ordered critical ``DRAIN`` queue and performs no broker calls or Portfolio
calculation.

:class:`~haymaker.controller.Controller` owns broker submission/cancellation,
immediate OrderInfo registration, status and rejection handling, Fill and
commission processing, Trade rebinding, blotter attribution, aggregate broker
reconciliation, target verification, and futures rolling. Target verification
waits only for OPEN, CLOSE, and TARGET_ADJUSTMENT work; protective stops and
take-profits remain active without delaying the check. A superseded target is
not checked or compared with the broker.

.. autoclass:: haymaker.book.Book

.. autoclass:: haymaker.book.OrderInfo

.. autoclass:: haymaker.book.FillRecord

.. autoclass:: haymaker.book.PositionState

.. autoclass:: haymaker.book.TargetState

.. autoclass:: haymaker.controller.Controller
   :members: trade, cancel

State conversion
================

The standalone converter is dry-run by default and never modifies its source:

.. code-block:: bash

   python scripts/migrate_components_state.py \
       --source-db legacy_haymaker \
       --target-db fresh_components

Inspect the count, P&L, identifier, active-state, and episode report. Add
``--apply`` only after selecting an empty target database. Compatible reruns
are idempotent; mixed or foreign target data is refused.
