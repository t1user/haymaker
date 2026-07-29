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
class is structurally incompatible. Value-dependent checks happen in
``onData``.

Fan-out sends the same object reference to every branch. Immutable standard
messages are safe to share. A custom branch that mutates its input must copy it
first.

.. autoclass:: haymaker.base.Atom
   :members: connect, pipe, union, validate_source, onStart, onData, onFeedback

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
effective market-observation time, while ``created_at`` records local creation.
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
the runtime is connected. Historical bars normally feed either
:class:`~haymaker.components.BarAggregator` for eventkit bar filters or
:class:`~haymaker.components.DfAggregator` for dataframe history. Streamers are
registered process-wide during strategy import and started by the runtime.

.. autoclass:: haymaker.components.Streamer

.. autoclass:: haymaker.components.HistoricalDataStreamer

.. autoclass:: haymaker.components.MktDataStreamer

.. autoclass:: haymaker.components.RealTimeBarsStreamer

.. autoclass:: haymaker.components.TickByTickStreamer

.. autoclass:: haymaker.components.BarAggregator

.. autoclass:: haymaker.components.DfAggregator

.. autoexception:: haymaker.components.aggregators.WrongStreamer

.. autoexception:: haymaker.components.aggregators.MissingStreamerParam

Bar filters
-----------

The filter objects below group source bars and retain their output
``BarDataList``. ``NoFilter`` preserves one output bar per input bar.

.. autoclass:: haymaker.components.CountBars

.. autoclass:: haymaker.components.VolumeBars

.. autoclass:: haymaker.components.TickBars

.. autoclass:: haymaker.components.TimeBars

.. autoclass:: haymaker.components.NoFilter

.. autoclass:: haymaker.components.VolumeGrouper

Signal models
=============

:class:`~haymaker.components.SignalModel` is the general structured Signal
producer. :class:`~haymaker.components.PandasSignalModel` retains the existing
dataframe conveniences: implement ``df(data)`` and return the complete
calculated dataframe. By default, ``signal_fields="signal"`` selects a scalar
from the latest row. A two-field tuple such as
``signal_fields=("in", "out")`` creates ``SignalPair(entry=..., exit=...)``.
The selected field or fields are excluded from metadata; the index becomes
``as_of`` when datetime-like, and all other row fields become metadata.

An optional custom row hook may build the Signal directly. An optional audit
sink must use ``DRAIN`` and records calculation history under
``{source_key}_{ACTIVE.localSymbol}_{run_started_at}``. The first successful
write stores the complete frame; later writes append only new rows. A NEXT-only
futures change does not rotate audit history.

.. autoclass:: haymaker.components.SignalModel
   :members: create_signal

.. autoclass:: haymaker.components.PandasSignalModel
   :members: df, create_signal

.. autofunction:: haymaker.components.read_signal_audit

.. autoclass:: haymaker.datastore.AsyncDataStore

.. autoclass:: haymaker.datastore.QueuedDataSink

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

Bracket legs
============

Bracket legs convert a completely filled entry Trade and validated Signal
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
