# Components architecture

This document records the implemented live-component architecture and its
invariants. It is the contract for future changes.

## Package boundary

`Atom` and `Pipe` remain general-purpose composition primitives in
`haymaker.base`. They accept arbitrary application messages. Base `onData`
raises `NotImplementedError`; every concrete Atom explicitly emits output.
`onStart(data, source)` receives arbitrary mutable startup data and reserves no
keys. Fan-out passes one shared object reference, so a mutating branch must
copy first. Dataclass-based Atom subclasses use `@dataclass(eq=False)` at every
decorated inheritance level so stateful graph nodes retain identity equality
and object hashing.

`source.connect(*targets)` asks every target to validate the source before any
connection changes. Built-ins use `validate_source()` only when they require a
concrete upstream capability visible at composition time. An override returns
normally for a compatible source and raises for an incompatible source; a
boolean return value is ignored. Atoms do not declare input or output message
types. Message envelopes, values, and conditional requirements remain runtime
validation in `onData()`.

The user-facing trading toolbox is exported explicitly from
`haymaker.components`. It contains messages, streamers, bar aggregators,
DataFrame aggregators, SignalModels, one-to-one processors, Portfolio
boundaries, routing, execution models, bracket legs, and event timeout helpers.
Routing, execution models, and bracket execution are organized under the
`haymaker.components.execution` subpackage and re-exported at the root.
Bar aggregation and DataFrame aggregation are separate public component
families: the former incrementally groups broker bar objects, while the latter
maintains complete DataFrames, restores persisted history, and supports
DataFrame transformations such as equal-volume grouping. Runtime, Book,
Controller, persistence, Trader, contract selection, and
`haymaker.config.TimeoutPolicy` stay outside. There are no forwarding modules
or aliases for former top-level component paths. Each public leaf module owns
its `__all__`; package initializers declare which modules participate and
aggregate their non-overlapping exports.

`FuturesPandasAggregator()` resolves a runtime-default awaited datastore at
startup from its connected streamer's bar size, `whatToShow`, and `useRTH`.
`HistoricalDataStreamer(datastore=True)` resolves the same cached store, while
the streamer uses `False` to disable persisted-endpoint lookup. Supplying the
same custom datastore to both components bypasses the runtime default.

## Runtime metadata

Every Atom accesses one ready `RuntimeContext`. Its `book` is the process-owned
typed accounting service. `run_started_at` is fixed for the process/component
graph. `workload_generation` increments immediately before each supervised
workload start and lets stateful models recover once per generation. Neither
value modifies generic startup data.

## Timeout ownership

`EventTimeout` monitors any `eventkit.Event` and invokes a synchronous or
asynchronous user callback once per inactivity episode. Emissions restart the
deadline; after firing, the monitor waits for a fresh emission before rearming.
The user owns cancellation, and supervised restarts never reach general event
timeouts.

`MarketDataTimeout` extends this mechanism with Contract trading sessions and
the runtime's `TimeoutPolicy`. Streamers create it through `from_atom()` after
contract qualification and supervisor binding. Closed sessions pause
monitoring until the next open, when a complete interval begins. An open-market
deadline either logs once or requests one supervised workload restart; a
rejected request remains disarmed because it indicates that another lifecycle
transition is already active. Market-data timeouts are workload-owned and are
all cancelled by `LiveRuntime` when the workload stops or exits.

## Trading messages

All standard messages are frozen, keyword-only dataclasses with aware
timestamps, finite numeric fields, and copied read-only top-level metadata.

`Signal` identifies one logical input by opaque `source_key`, carries the
applicable Contract and value, and declares `SignalType.STATE` or
`SignalType.EVENT`. STATE replaces the prior desired state. Every EVENT is a
new observation; zero EVENT means no event. `as_of` is the optional effective
market-observation time, while `created_at` is local creation time.

`PositionProposal` is the narrow one-to-one result of a built-in signal
processor. It preserves the original Signal, selects direction `-1`, `0`, or
`1`, and always has `PositionIntent.OPEN`, `CLOSE`, or `REVERSE`.

`PositionTarget` is an absolute signed setpoint for a concrete execution
Contract. It never captures current quantity or a proposed delta. Its numeric
target remains authoritative after acceptance. `source_key` and
`PositionIntent` are optional in the general message; the wrapper/bracket flow
requires both. Intent is checked at initial acceptance and is not a durable
execution command.

Order attribution uses open-ended strings. `StandardOrderRole` supplies OPEN,
CLOSE, TARGET_ADJUSTMENT, STOP_LOSS, TAKE_PROFIT, ROLL, LIQUIDATION,
RECONCILIATION, MANUAL, and UNKNOWN while custom non-empty strings remain
valid.

## Signal production and one-to-one flow

`SignalModel` is an identity-based dataclass that owns source identity,
Contract, SignalType, local creation time, and standard Signal emission. It
does not own futures-roll policy. Subclasses implement `calculate_signal(data)`
and return `SignalCalculation(value, metadata, as_of)`; they never reconstruct
framework-owned Signal identity. A model may override `validate_signal_value()`
for restrictions beyond the standard immutable Signal validation. A Signal
carries either one finite scalar value or a finite `SignalPair(entry, exit)`.
`as_of` remains the effective observation label while `created_at` records
local construction; these differ intentionally for IB's left-labelled bars.

`PandasSignalModel` retains dataframe, BarDataList, mapping, and compatible
input conversion; subclasses implement `df(data)`. `signal_fields` accepts
either one field name for a scalar or an `(entry, exit)` field-name tuple for a
SignalPair. The last returned row is authoritative and its index supplies
`as_of` when possible. `metadata_fields=None` copies every non-signal row field,
an empty collection copies none, and an explicit collection selects fields. A
subclass may override `row_to_calculation(row)` while returning the same
`SignalCalculation` boundary.

`PandasSignalModel.persistence` accepts `False`, `True`, or a custom
`SignalFramePersistence`. `True` creates a model-owned default from Runtime
configuration; `False` performs no persistence. Custom and default policies
must queue synchronously rather than wait for storage I/O. Queue acceptance
precedes emission so the reference can be attached, while enqueue failure is
logged and never suppresses the Signal. Direct `create_signal()` calls do not
persist.

Built-in binary processors consume Signal and query
`Book.effective_quantity(source_key)`. STATE zero requests flat; EVENT zero is
ignored. Matching non-zero direction is suppressed. `BinarySignalProcessor`
accepts scalar `-1/0/1` values and uses explicit `OpposingSignalPolicy.CLOSE`
or `REVERSE`. `BinaryEntryExitSignalProcessor` accepts SignalPair, consults
entry only while flat and exit only while positioned, and never reverses
directly. Either processor may respect Book's `blocked_direction`.

A completely filled STOP_LOSS or TAKE_PROFIT that flattens an episode sets the
block to the exited direction. A permitted opposite OPEN clears the old block
on its first actual fill. CLOSE and ROLL are indifferent, and partial
protective fills do not set the block.

The dedicated one-to-one composition is:

```text
PandasSignalModel
    -> one-to-one signal processor
    -> PositionProposal
    -> PortfolioWrapper(PositionAllocator)
    -> PositionTarget
    -> BracketExecutionModel
    -> Controller
```

`PortfolioWrapper` accepts only PositionProposal, calls
`PositionAllocator.target_for()`, and emits at most one target. It copies the
mandatory proposal intent. `FixedSizeAllocator` accepts one size, a mapping
keyed by source, or a callable; signed target equals proposal direction times
allocated size.

## Direct Portfolio flow

`Portfolio` is the abstract direct account-wide boundary. It accepts raw
Signal and emits every PositionTarget returned by `process(signal)`. An
optional explicit source collection rejects unknown source keys and exposes
the expected universe; `None` enables dynamic membership without completeness
policy.

The abstract base does not batch, debounce, time out, or interpret `as_of`.
Concrete implementations own input state, synchronization, duplicate/late
handling, EVENT accumulation, and recomputation. One input may produce zero,
one, or several targets, normally with `intent=None`.

The direct composition is:

```text
multiple Signal paths
    -> Portfolio
    -> ExecutionRouter (optional)
    -> SerialTargetExecutionModel
    -> Controller
```

## Book and persistence

One process-owned `Book` replaces whole-system strategy snapshots. Book
performs no Portfolio calculation and no broker API calls. It owns typed
queries and recovery for:

- `OrderInfo`: complete serialized IB Trade, actual broker identifiers,
  explicit normalized FillRecords, role, submission time, stable execution
  model name, and optional source/position episode attribution;
- `PositionState`: one-to-one fill-accounted quantity, latest target, Contract,
  model name, episode ID, stopped direction, and only validated bracket
  recovery inputs;
- `TargetState`: latest direct Contract target for one named model;
- normalized Portfolio recovery mappings keyed by `portfolio_key`.

FillRecord preserves the complete Execution, fill time and Contract, optional
CommissionReport, and a deduplication key normally equal to `execId`. Explicit
FillRecords are authoritative execution evidence; serialized Trade remains
diagnostic evidence. Fill application is conditional and idempotent, and late
commission callbacks update the normalized record whether or not optional
blotter output is configured.

Physical Mongo collections are only `orders`, `state`, and `blotter`. State
documents use:

```text
position:{source_key}
target:{execution_model_name}:{conId}
portfolio:{portfolio_key}
```

All Book mutations use one ordered critical `DRAIN` queue. Order evidence is
queued before the projection derived from it. Recovery loads completed as well
as working order evidence so direct logical quantities can be rebuilt from
fills; current live Trades are rebound by orderId with permId fallback.

## Controller and execution models

Controller owns trading-disable and market-hours checks, broker submission,
immediate OrderInfo registration, status/rejection handling, Fill and
commission processing, offline Fill accounting, Trade rebinding, blotter
attribution, aggregate Contract reconciliation, delayed target verification,
and Controller-owned futures rolling.

Execution models consume absolute PositionTargets and have stable configured
names. They validate a target before persistence or submission, retain only the
newest target for their natural identity, derive work from Book state and
working orders, and report accepted targets to Controller. Recovery rebinds
completion callbacks to the current live Trade objects. Delayed verification
waits only for target-converging OPEN, CLOSE, and TARGET_ADJUSTMENT orders and
silently abandons a check when a newer target supersedes it; protective orders
do not delay verification.

`SerialTargetExecutionModel` groups by concrete Contract, supports arbitrary
quantities and same-side resizing, ignores optional intent, and permits one
active TARGET_ADJUSTMENT order per Contract. It converges again after
completion and recovery.

`BracketExecutionModel` owns one configured source key. New targets require a
matching source and initially consistent PositionIntent. Numeric target is
authoritative afterward. It supports no-op, OPEN, CLOSE, and REVERSE, rejects
non-zero same-side resizing, creates a new `position_id` for each opening
episode, and preserves it on close, brackets, and roll orders. Protection is
attached only after a complete entry fill. A protective exit closes the
episode, zeroes the recovered target, and persists the exited direction as
blocked. Stop-loss protection is critical while take-profit is optional. A
regular close joins the active brackets' OCA group, so the first filled exit
causes IB to cancel the remaining exits.

Order keyword precedence is:

```text
built-in fallback < global order defaults < model constructor options
```

## Routing and recovery ownership

`ExecutionRouter` receives fixed ordered `ExecutionRule` objects and an
optional default model. First match wins; no match without a default fails
closed. Model instances are constructed before the Router and names must be
unique. Every configured model starts once per workload generation.

Current rules always select the model; persisted ownership never overrides
them. Before recovery, every active direct `TARGET_ADJUSTMENT` must still select
the model recorded on that order. A mismatch, ambiguous ownership, missing
TargetState, or unroutable target blocks that Router locally without cancelling
orders or disabling Controller trading. Other roles, including one-to-one
bracket work, are excluded. A stable model name is a promise that its
implementation and configuration remain recovery-compatible.

Held quantity does not pin a model. During recovery, all idle direct-target
reassignments must be routable under current rules before any are persisted,
then models resume convergence. Predicates used for recoverable routing must be
deterministic from persisted TargetState fields. Final process shutdown warns
when active adjustments remain so operators can defer routing or model changes.
There is no route key in messages or persistence.

## Futures and calculation audit

Futures rolling remains Controller-owned. It reads PositionState, preserves
source and `position_id`, submits role ROLL directly, and retains ACTIVE/NEXT
held-contract rules.

Direct-mode futures rolling remains deferred. The current roller accounts for
one-to-one PositionState holdings but does not yet define futures-series
identity, quantity convergence across contract expiries, or working-order and
target recovery for direct Portfolio/SerialTargetExecutionModel flows. A later
refactor must cover every supported work mode before direct futures-series
rolling is considered complete. Direct target identity is intentionally
unresolved: current execution uses concrete `conId`, while a later design must
decide how Portfolio communicates which targets belong to one logical
instrument and supersede each other. Router and ExecutionModel must not infer
that policy from Contract fields before this boundary is agreed.

PandasSignalModel audit symbols are:

```text
{source_key}_{ACTIVE.localSymbol}_{run_started_at}
```

ACTIVE comes from the selector independently of transaction Contract or NEXT.
The first successful calculation for a generation writes the complete
dataframe; later saves append only new rows. ACTIVE rotation starts a new
generation only after a successful calculation. A supervised restart continues
the process run, while a new process creates a new run. Default persistence uses
a dedicated ordered `DRAIN` sink created from
`signal_persistence.library`. Metadata records source, run start, ACTIVE
Contract. Futures history may be back-adjusted without a downstream
notification.

## Conversion and validation

`scripts/migrate_components_state.py` is the only legacy conversion path. It
is dry-run by default, requires distinct explicit source and target database
names, never modifies source data, and writes only with `--apply`. It preserves
Trade/Fill/Execution/CommissionReport evidence and broker IDs, converts only
the latest useful strategy snapshot, uses deterministic provenance for reruns,
and refuses mixed or foreign target data. It must never be run against a real
database as part of code implementation or tests.

Public exports, Atom, and Pipe require usage-focused Google-style,
Sphinx-compatible docstrings. Focused tests cover message validation,
connection atomicity, STATE/EVENT transitions, locks, Portfolio allocation,
target supersession, callback recovery, Router ownership validation, Fill
deduplication, Trade rebinding, bracket creation, audit generations, and
migration idempotence. Full validation includes pytest, mypy, Black, and Sphinx
reference checks.
