*****************
Live Components
*****************

Haymaker strategies are ordinary Python modules that compose
:class:`~haymaker.base.Atom` objects after the live runtime has installed a
ready ``RuntimeContext``. ``Atom`` is intentionally
general: custom Atoms may pass any object. The built-in trading toolbox is
available from ``haymaker.components`` and uses structured frozen message
envelopes.

Atom composition
================

Every concrete Atom implements ``onData`` and explicitly emits through
``dataEvent``. Base ``onData`` raises ``NotImplementedError``. Startup remains
synchronous and receives arbitrary mutable data through
``onStart(data, source)``; Haymaker does not add strategy names, timestamps, or
reserved fields.

``source.connect(*targets)`` validates every target before changing any event
connections. A component overrides ``validate_source`` only when an upstream
capability is structurally required, such as the API supplied by a Streamer.
The override returns normally for a valid source and raises ``TypeError`` or a
more specific domain exception for an invalid source; returning a boolean does
not reject a connection. Atoms do not declare input or output message types.
Each component validates messages when they arrive in ``onData``.

Fan-out sends the same object reference to every branch. Standard messages
prevent field reassignment and copy metadata at the top level, but Contracts
and nested metadata values remain shared mutable objects. Consumers must treat
them as immutable or copy them before mutation. Signal, PositionProposal, and
PositionTarget are intentionally unhashable.

Atoms are stateful graph nodes rather than value records. A dataclass-based
Atom must therefore use ``@dataclass(eq=False)``. Repeat ``eq=False`` on every
subclass decorated with ``@dataclass`` because a new dataclass decoration would
otherwise generate value equality again.

.. autoclass:: haymaker.base.Atom
   :members: connect, disconnect, clear, pipe, validate_source, onStart, onData, onFeedback

.. autoclass:: haymaker.base.Pipe
   :members:

Blueprints and concrete Contracts
---------------------------------

.. autoclass:: haymaker.enums.ActiveNext

.. autoclass:: haymaker.contract_registry.ContractRegistry
   :members: blueprint_key, blueprint_for, contracts_for, get_selector

.. autoclass:: haymaker.contract_selector.AbstractBaseContractSelector

.. autoclass:: haymaker.contract_selector.AbstractBaseFutureWrapper
   :members: roll_day, last_trading_day

.. autoclass:: haymaker.contract_selector.FutureSelector
   :members: nth_contract, active_contract, next_contract, past_contracts

.. autoclass:: haymaker.details_processor.Details

Assign an ordinary ``ibi.Contract`` blueprint to ``atom.contract``. Reading
``atom.contract`` resolves its current ``which_contract`` role. Reading
``atom.contract_blueprint`` returns the original declaration as a safe copy;
``atom.contract_selector`` exposes the initialized selector and raises if it
is not ready.

For futures, ``selector.nth_contract(0).contract`` is ACTIVE and index one is
the following eligible expiry. Index one is not necessarily NEXT: NEXT stays
at ACTIVE until the configured margin. Explicit nth requests raise when the
requested maturity is unavailable; NEXT retains its last-available fallback.
Wrappers also expose ``roll_day`` and expiry information.

The registry's ``blueprint_for(contract)``, ``get_selector(contract)`` and
``contracts_for(contract)`` accept a declaration or qualified member. Membership
comes from qualification, not symbol guessing. Identical declarations share
registration; distinct declarations resolving to overlapping conIds raise.
The registry keeps declaration copies so qualification cannot mutate identity.

SignalModels normally emit ``self.contract``. Override
``select_signal_contract()`` to return another qualified chain member, or
``self.contract_blueprint`` for a custom direct Portfolio to resolve. The
framework still constructs source identity, value, metadata and timestamps.
The one-to-one allocator requires a qualified Contract. Selecting a different
Signal Contract does not change the selector's ACTIVE audit identity.

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
a concrete Contract with a non-zero ``conId``. It never contains a captured
current quantity or a proposed order delta. The numeric target is authoritative.
Direct Portfolio targets omit ``source_key`` and intent. Each is an absolute
setpoint for its exact concrete Contract; multiple expiries can have independent
targets. One-to-one targets instead carry ``source_key`` and the mandatory
proposal intent supplied by ``PortfolioWrapper``.

One-to-one Contract selection belongs to ``BracketExecutionModel``, not the
wrapper. OPEN uses the incoming target Contract. CLOSE uses the source's held
or pending-entry Contract in Book even if the incoming target names another
expiry. REVERSE completely closes that episode before opening the incoming
Contract under a new position ID. Book stores held and pending-target Contracts
and their bracket inputs separately; changing ACTIVE/NEXT or accepting a newer
target never changes the identity of an existing holding. Recovery resumes the
pending operation without requiring an old Trade callback to be replayed.

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
``TimeBars`` is a live timer-driven event operator and cannot be used as a
``BarAggregator`` filter because historical backfills do not follow wall-clock
timer boundaries.

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
broker-backfill writes. Periodic saves are best effort: a timer tick is skipped
while an earlier save remains in progress, and a failed save is logged before
the next complete-frame attempt retries the unsaved tail.

The DataFrame components are public from both
``haymaker.components.dataframe_aggregators`` and the package-level
``haymaker.components`` toolbox.

The DataFrame groupers recalculate their complete output from each cumulative
input frame and emit only when a new group has completed. ``CountGrouper``
uses a fixed number of source rows, ``VolumeGrouper`` and ``TickGrouper`` use
cumulative thresholds, and ``TimeGrouper`` uses pandas time buckets. The final
incomplete group is withheld. Time grouping also omits empty buckets and waits
for a row in the following bucket before treating the previous one as complete.
Grouped rows use first/max/min/last OHLC, summed volume and ``barCount``, a
volume-weighted ``average``, and the last value of any other column.
No pass-through grouper is needed: connect ``FuturesPandasAggregator`` directly
to the next component when no regrouping is required.

.. autoclass:: haymaker.components.FuturesPandasAggregator

.. autoclass:: haymaker.components.CountGrouper

.. autoclass:: haymaker.components.VolumeGrouper

.. autoclass:: haymaker.components.TickGrouper

.. autoclass:: haymaker.components.TimeGrouper

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
   :members: calculate_signal, create_signal, validate_signal_value, onData, select_signal_contract

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
``PositionAllocator.target_for()`` protocol, and emits at most one target. The
allocator preserves source, Contract, and metadata; the wrapper owns transfer
of the proposal's mandatory intent to the target.

.. autoclass:: haymaker.components.PositionAllocator
   :members: target_for

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
absolute targets for concrete Contracts. It must omit one-to-one
``source_key`` and intent. Portfolio owns allocation among expiries: a target
for NEXT never implicitly reduces a holding in ACTIVE. Emit a separate ACTIVE
target to reduce it. The optional ``sources`` collection rejects unknown
Signal source keys. Base Portfolio does not batch, debounce, time out, or
interpret ``as_of``; concrete policies own those decisions.

.. autoclass:: haymaker.components.Portfolio
   :members: process, positions_for_blueprint

Portfolio accounting and optional state
---------------------------------------

Use ``self.book.aggregate_quantity(contract)`` for filled quantity in an exact
Contract and ``self.book.active_orders(contract=contract)`` for working orders.
``self.positions_for_blueprint(signal.contract)`` returns filled quantities
across all registered members of that blueprint, including held past expiries.
The query requires initialized registry membership. Neither desired targets
nor working quantities are actual fills.

Both execution modes use the same maintained Contract balances. These queries
read current state; they do not scan historical executions or contact IB.
``self.book.logical_positions()`` returns a new mapping of all non-flat net
balances by concrete Contract. For individual one-to-one episodes, use
``self.book.position_state(source_key)`` instead.

In direct mode, allocation to each ``source_key`` belongs to your Portfolio.
Net fills cannot determine how much of a combined broker position belongs to
each input. Keep that allocation mapping yourself.

``PortfolioStateMixin`` optionally supplies explicit ``load_state()`` and
``save_state(mapping)`` methods backed by Book. Declare ``portfolio_key``, load
before consuming inputs, and save your normalized allocation mapping when it
changes. Base Portfolio has no mandatory persistence. For an independent store,
override those methods or omit the mixin entirely; see :doc:`storage` for
advanced storage ownership and lifecycle.

.. autoclass:: haymaker.components.PortfolioStateMixin
   :members: load_state, save_state

Execution completion feedback
-----------------------------

ExecutionModel's ``dataEvent`` reports acceptance, not completion.
``targetReachedEvent`` reports a converged PositionTarget after Book reflects
the fills. It also travels upstream through Atom's normal ``onFeedback`` path;
ExecutionRouter forwards model feedback to the connected Portfolio.

Override ``Portfolio.onFeedback(target, *args)`` to implement sequencing such
as closing A before requesting B. Check the concrete Contract, quantity and
target creation time against your own pending allocation. Notifications contain
persisted target fields, not arbitrary execution metadata. They may repeat
after restart, so feedback policy must be idempotent. Book queries remain the
recovery authority if a notification was missed. This is an extension point,
not a built-in cross-Contract allocation algorithm.

Execution and routing
=====================

Execution models have stable configured names used for recovery. They persist
the newest target, inspect Book quantity and working orders, and derive the
next order rather than replaying stale intent.

:class:`~haymaker.components.SerialTargetExecutionModel` manages each concrete
Contract independently, supports arbitrary same-side resizing, and permits at
most one active TARGET_ADJUSTMENT per conId. New targets supersede old targets
for that Contract, not for its whole futures series. Portfolio chooses which
Contracts to trade; execution does not substitute another held expiry.

:class:`~haymaker.components.BracketExecutionModel` manages one configured
``source_key``. Initial targets require consistent intent, same-side non-zero
resizing is rejected, and each opening episode gets a fresh ``position_id``.
Protective brackets are attached only after the entry is completely filled.
The stop-loss is critical protection for every established episode; a
take-profit is optional and its absence is not a synchronization failure.
Regular CLOSE orders join the active brackets' OCA group, allowing IB to cancel
the unfilled protective orders when any exit fills.

If an entry fills completely while Haymaker is offline, its model restores
missed initial brackets automatically after order/position reconciliation and
before the configured missing-bracket policy acts. It uses the held episode's
saved calculation inputs (for example ATR), quantity-weighted execution price,
and the same configured bracket legs as a live entry. No new Signal or commission
report is needed. Keep the model's bracket configuration recovery-compatible
across restarts.

Repeated recovery does not duplicate stops. A surviving take-profit supplies its
existing OCA group when the initial stop was never installed. An existing stop
is left untouched, and a missing optional take-profit alone is not repaired.
Partial entries still wait for complete filling; active closes and rolls keep
their own sequencing. An already closed episode receives no new brackets.

This is **initial installation recovery**, not general replacement of protection
that was previously active and then cancelled, rejected or otherwise lost.
In particular, the original entry price cannot reproduce a trailing stop's
later broker-maintained state. Those cases retain the configured
``controller.missing_brackets`` policy. Missing or inconsistent entry evidence,
unresolved broker-order attribution, missing Contract ticks, or failed required
stop submission prevents automatic initial installation; failed recovery during
Controller reconciliation disables trading and reports the affected source.

The model registers that source for Controller-owned futures rolling by
default. A strategy that intentionally manages its own one-to-one roll can opt
out at construction time:

.. code-block:: python

   execution = BracketExecutionModel(
       "intraday_es",
       stop=FixedStop(10),
       auto_roll_futures=False,
   )

Contradictory roll settings for the same ``source_key`` are rejected during
strategy construction.

.. autoclass:: haymaker.components.ExecutionModel

.. autoclass:: haymaker.components.SerialTargetExecutionModel

.. autoclass:: haymaker.components.BracketExecutionModel

Futures rolling
---------------

Controller owns the one daily UTC schedule, stale-holding discovery, and startup
recovery coordination. It delegates broker work to exactly one process-wide
:class:`~haymaker.components.FutureRollExecutor` family. Direct and bracket
accounting cannot be mixed in one process. This matches the supported deployment
model: a process uses either account-wide direct targets or independently
managed one-to-one episodes.

ContractRegistry supplies the futures-series identity from the registered
blueprint and qualified contract-detail chain. The default
:class:`~haymaker.components.PastToActiveRollPolicy` moves only
``selector.past_contracts`` into ACTIVE; NEXT and all later eligible expiries
are retained. FutureRoller persists ``RollState`` before any broker order and never infers
series membership from symbol, exchange, or multiplier alone.

Custom triggers and destinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply ``roll_policy=`` to either execution model to customize the trigger and
destination without replacing durable execution. A bracket policy applies to
its source; a direct policy applies to that model's concrete holdings.
``auto_roll_futures=False`` remains the one-to-one opt-out. Policies receive a
full selector evaluated at the current check time, without changing the graph's
ACTIVE/NEXT selection. For example, roll twenty business days before the
holding's roll day into its successor:

.. code-block:: python

   import pandas as pd
   from haymaker.components import FutureRollPolicy, RollDecision

   class EarlyRoll(FutureRollPolicy):
       def plan(self, holding, selector, *, now):
           chain = selector.all_contracts
           index = next(i for i, item in enumerate(chain)
                        if item.contract.conId == holding.contract.conId)
           held = chain[index]
           trigger = held.roll_day - pd.offsets.BusinessDay(20)
           if now.replace(tzinfo=None) < trigger:
               return None
           # A missing successor raises rather than selecting an arbitrary expiry.
           return RollDecision(
               destination=chain[index + 1].contract,
               occurrence=f"early-{held.roll_day:%Y-%m-%d}",
           )

Use your trading calendar instead of pandas' weekday calendar where holidays
matter. A fixed schedule should use a stable ``occurrence`` label for that
scheduled event. Completed occurrence markers survive restart and prevent
rolling its replacement again on every check. With no label, the trigger must
become false after rolling. Each decision moves a whole holding to another
qualified member of the same registered series; it is not a general spread
allocation API. When several expiries are due, later expiries are handled first.

Controller already checks daily. A user-owned scheduler on the event loop may
call ``controller.future_roller.roll()`` more often; do not create a new timer
on every reconnect. Accepted endpoints are persisted: recovery resumes them
without reevaluating the policy.

.. autoclass:: haymaker.components.FutureRollPolicy
   :members: plan

.. autoclass:: haymaker.components.PastToActiveRollPolicy

.. autoclass:: haymaker.components.RollDecision

Durable execution
~~~~~~~~~~~~~~~~~

:class:`~haymaker.components.DirectFutureRollExecutor` waits for active
TARGET_ADJUSTMENT work on both roll endpoints, reads the accounted balance,
and submits a calendar-spread BAG. A durable target-transfer snapshot makes the
old target zero and adds its target to the destination. Reapplying that snapshot
after recovery is idempotent; newer explicit targets supersede it. Holdings in
other expiries remain independent. Custom executors can override
``target_transfers()`` without replacing the broker execution sequence.

An optional Portfolio callback can subscribe to
``controller.future_roller.completedEvent``. It receives the completed
``RollState`` after Book has accounted for the movement and applied target
transfers. Use its concrete endpoints to update your source allocation model
and save your normalized Portfolio state if required. This notification is not
a durable message queue: on process recovery, reconcile allocations with Book's
latest targets and holdings instead of requiring an old callback to replay.
Ordinary upstream ``targetReachedEvent`` feedback remains a PositionTarget.
Target verification waits for roll sequencing (within its configured retry
limit), but does not wait for standing protective stop/take-profit orders.

:class:`~haymaker.components.BracketFutureRollExecutor` preserves
``source_key`` and ``position_id`` while processing logical episodes serially.
Because IB exposes only the account net, offsetting logical sources are assigned
deterministically to physical BAG work or to an observed spread-price
adjustment. After moving an episode, the executor cancels its old protection and
installs an active replacement stop before advancing. Take-profit replacement
is optional and best effort.

Before submitting work, bracket rolling waits for OPEN/CLOSE orders of all
unprocessed sources, then refreshes their quantities and net allocation from
Book. Episodes that closed while waiting are skipped; changed episode identity
blocks before submission. Completed logical offsets are preserved; if changed
remaining quantities cannot represent that offset safely, the roll blocks for
review instead of submitting an inconsistent BAG.

The built-ins are used automatically. Supply a custom, process-shared executor
only when its persisted stages and recovery behavior are intentionally
compatible:

.. code-block:: python

   roll_executor = DirectFutureRollExecutor(name="index_future_roll")
   execution = SerialTargetExecutionModel(
       name="index_targets",
       future_roll_executor=roll_executor,
   )

Synchronization back-reports Fill evidence before resuming an incomplete roll.
A missing executor, changed mode or executor name, lost roll evidence, failed
critical stop replacement, or explicit ``BLOCKED`` state fails closed. Process
shutdown warns about incomplete roll work; preserve the mode and executor name
until it completes or is reviewed.

.. autoclass:: haymaker.components.FutureRollExecutor
   :members: holdings, create_state, advance, recover

.. autoclass:: haymaker.components.DirectFutureRollExecutor

.. autoclass:: haymaker.components.BracketFutureRollExecutor

.. autoclass:: haymaker.components.FutureRollMode

.. autoclass:: haymaker.components.FutureRollStage

.. autoclass:: haymaker.components.RollHolding

.. autoclass:: haymaker.components.RollParticipant

.. autoclass:: haymaker.components.RollState

Router rules are fixed and evaluated in declaration order; first match wins.
Without a default, unmatched targets fail closed. Current rules always select
the model; persisted ownership never overrides them. Before recovery, every
active direct ``TARGET_ADJUSTMENT`` must still select the model that submitted
it. A mismatch, ambiguous ownership, missing target state, or unroutable target
blocks that Router locally. It does not cancel orders or disable Controller
trading, and one-to-one bracket roles are outside this check.

Held quantity and idle targets do not pin an old model. Startup computes all
idle direct-target reassignments from current rules before applying any of
them, then starts the models. Custom predicates used for recoverable execution
must therefore be deterministic from persisted TargetState fields:
Contract, target quantity, and target creation time. Metadata is
unavailable during reconstruction. Reusing a configured model name across deployments asserts
that the implementation and configuration remain recovery-compatible. Final
process shutdown logs unfinished ``TARGET_ADJUSTMENT`` orders so operators can
avoid changing routing or models until those orders finish.

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
RollState, ContractPosition balances, Portfolio recovery state, stopped-direction
state, and blotter access. Both modes read the same contract-level balances,
updated as episode, order and roll records change. During a bracket roll,
physical movement remains distinct from serial episode updates so quantities
are not counted twice.

Book persists balances in the existing ``state`` collection under
``balance:{conId}``, after the records establishing them in its critical
``DRAIN`` queue. Startup verifies the balances once against saved accounting
records, reconstructing missing or inconsistent totals after interrupted writes.
These startup repairs complete synchronously before the runtime starts its loop;
subsequent mutations use the configured persistence queue.
Saved one-to-one corrections remain authoritative; offline executions are still
accounted by Controller synchronization. No historical replay occurs in ordinary
position queries. Book performs no broker calls or Portfolio allocation.
Source positions persist their applied-fill checkpoint with their quantity.
Startup completes a received fill whose position write was interrupted without
reapplying older fills over a correction. Critical persistence stops on its first
failed write; new broker submissions are rejected while that writer is unavailable.
Records created without source fill checkpoints require the standalone converter;
conversion takes the selected saved position as the accounting baseline and does
not infer corrections from historical fills. Review/reconcile that baseline first.
An explicit state clear stores a per-target
Fill-evidence cutoff: historical orders and Fills remain available, but
pre-reset executions cannot recreate a cleared direct position after restart.
Fills that actually arrive after the clear are still accounted.

:class:`~haymaker.controller.Controller` owns broker submission/cancellation,
immediate OrderInfo registration, status and rejection handling, Fill and
commission processing, Trade rebinding, blotter attribution, aggregate broker
reconciliation, target verification, and futures-roll coordination. Target verification
waits only for OPEN, CLOSE, and TARGET_ADJUSTMENT work; protective stops and
take-profits remain active without delaying the check. A superseded target is
not checked or compared with the broker.

Each synchronization pass uses the successfully requested broker-position
snapshot as its authoritative input. Cached/fresh disagreement is retried
without reconnecting; request timeout or failure asks the supervisor to
recover broker state. Position correction is deferred for a Contract with
active one-to-one OPEN/CLOSE work. When correction is eventually applied, the
one-to-one target is aligned with the broker quantity so startup recovery does
not recreate a position that reconciliation intentionally removed.

.. autoclass:: haymaker.book.Book

.. autoclass:: haymaker.book.OrderInfo

.. autoclass:: haymaker.book.FillRecord

.. autoclass:: haymaker.book.PositionState

.. autoclass:: haymaker.book.TargetState

.. autoclass:: haymaker.book.ContractPosition

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

The converter accepts legacy strategy snapshots or component ``state`` records,
not a mixture of both. It preserves complete Trade diagnostics and authoritative
explicit Fill/CommissionReport evidence. Older one-to-one records without
separate held/pending Contracts are reconciled to episode order evidence;
missing or ambiguous evidence is refused. Direct keyed records convert only
when their held and working exposure belongs to the target's concrete Contract.
Multiple old targets mapping to one conId require an explicit allocation
decision before conversion. Finish pending rolls under the old implementation
before changing their schema. Existing non-empty databases produced by another
converter version are incompatible targets.

The report distinguishes optional blotter totals from deduplicated Fill-level
commission/P&L totals and includes source/episode order grouping even when
blotter writing was disabled. No live schema fallback or dual writes exist.
Derived balance documents are not copied by the converter: Book reconstructs
them from the converted accounting records at startup.
