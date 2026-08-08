# Components package guidance

`haymaker.components` is the discoverable public toolbox for user-composed
trading pipelines. Keep `Atom`, `Pipe`, `Controller`, `Book`, runtime services,
persistence infrastructure, contract selection, and broker infrastructure
outside this package. Every public component module owns an explicit
module-level `__all__`. `haymaker.components.__init__` imports those declared
names and builds the package `__all__` from the ordered set of registered
public modules. A module's `__all__` therefore means both public within that
module and promoted to the package toolbox; keep submodule-only and private
helpers out. Export names must be unique across modules.

## Message boundaries

- `Signal` is a frozen, keyword-only input with a stable `source_key`, Contract,
  finite scalar or `SignalPair(entry, exit)` value, mandatory `SignalType`,
  aware timestamps, and copied read-only top-level metadata. `STATE` replaces a
  source's prior desired state; each `EVENT` is a new event and an EVENT zero
  is normally ignored by one-to-one processors.
- `PositionProposal` is the frozen one-to-one boundary. It preserves the
  original Signal, adds direction `-1`, `0`, or `1`, and always has
  `PositionIntent.OPEN`, `CLOSE`, or `REVERSE`.
- `PositionTarget` is a frozen absolute signed setpoint. It never carries a
  captured current quantity or proposed delta. Its numeric target remains
  authoritative after acceptance.
- `PositionIntent` is optional on general targets. It is mandatory only at the
  `PortfolioWrapper -> BracketExecutionModel` boundary and is an initial
  lifecycle assertion, not a lasting execution command.

Copy a message before changing it. Metadata is only shallowly protected, so a
branch that mutates nested objects must copy those objects explicitly.

## Supported flows

The dedicated one-to-one flow is:

```text
SignalModel -> one-to-one processor -> PositionProposal
    -> PortfolioWrapper(PositionAllocator) -> PositionTarget
    -> BracketExecutionModel -> Controller
```

The account-wide flow is:

```text
multiple SignalModels -> Portfolio -> PositionTarget(s)
    -> ExecutionRouter (optional) -> SerialTargetExecutionModel -> Controller
```

`PortfolioWrapper` calls `PositionAllocator.target_for()` and emits at most one
target. Direct `Portfolio` implementations own source state, synchronization,
`as_of`, duplicate/late input, timeout, and recomputation policies. Do not put
those policies in the abstract base.

## Market-data aggregation

Keep the two public aggregation families distinct. `aggregators.py` operates
on `ib_insync` bar objects through `BarAggregator` and its count, volume, tick,
time, and pass-through filters. `dataframe_aggregators.py` maintains complete
pandas DataFrames through `FuturesPandasAggregator` and contains
DataFrame-native transformations such as `VolumeGrouper`.

`FuturesPandasAggregator` is the futures-only persistence companion to
`HistoricalDataStreamer`. Its default datastore and
`HistoricalDataStreamer(datastore=True)` resolve the same runtime-cached store
for bar size, data type, and RTH policy. Supplying the same custom awaited
datastore to both bypasses the runtime default. The aggregator restores and
saves complete history; the streamer consults the persisted endpoint to
shorten its next IB request. `False` is valid only for the streamer;
FuturesPandasAggregator requires stored history.
Do not collapse the DataFrame components into `aggregators.py` or treat their
public module as an implementation detail.

SignalModels are identity-based dataclasses used by both flows. They own
`source_key`, the Signal Contract, SignalType, `created_at`, and standard
emission, but do not own futures-roll policy. User implementations return only
`SignalCalculation(value, metadata, as_of)`; do not make them reconstruct or
override the framework-owned Signal envelope. `as_of` is the observation label
and remains distinct from local creation time, notably for IB's left-labelled
bars. PandasSignalModel treats the last row returned by `df()` as authoritative;
user calculations own ordering, duplicates, and correctness. Its
`metadata_fields` distinguishes all non-signal fields (`None`), no fields (an
empty collection), and an explicit selection. Custom row conversion returns a
SignalCalculation by overriding `row_to_calculation(row)`.

`PandasSignalModel.persistence` is its only persistence option. `False`
disables it, `True` resolves a new model-owned default from RuntimeContext, and
a `SignalFramePersistence` object supplies custom non-blocking behavior.
Persistence queue acceptance precedes Signal emission so an accepted lookup
reference can be attached, but enqueue failure is logged and never suppresses
the Signal. Calling `create_signal()` directly has no persistence side effect.

`BinarySignalProcessor` accepts only scalar `-1/0/1` values and exposes
`OpposingSignalPolicy.CLOSE` or `REVERSE`. `BinaryEntryExitSignalProcessor`
accepts only SignalPair, uses entry while flat and exit while positioned, and
never reverses directly. Both may opt into blocked-direction checks. A complete
STOP_LOSS or TAKE_PROFIT fill that flattens the episode sets the block; the
first actual fill of a permitted opposite OPEN clears it. CLOSE and ROLL do not
change it.

## Event timeouts

`EventTimeout` is the general callback-based inactivity monitor for any
`eventkit.Event`. It is user-owned, independent of Atom and supervisor
lifecycle, fires once per stale episode, and rearms only after the source emits
again. A positive interval must be armed on a running asyncio loop. The owner
calls `cancel()`; ending the source event also cancels it.

`MarketDataTimeout` inherits the generic mechanism and adds Contract trading
hours plus `haymaker.config.TimeoutPolicy`. The policy remains configuration,
not a component export. Create `MarketDataTimeout` with `from_atom()` during
`onStart()` or later, after contract qualification and supervisor binding.
Closed markets pause until the next open and then start a full interval.
Log-only timeouts rearm after fresh data. Restart-enabled timeouts request one
workload rebuild and remain disarmed even when the request is rejected because
another lifecycle transition is active. `LiveRuntime` alone cancels all
market-data timeout instances when a workload stops; never include general
`EventTimeout` instances in that registry.

## Atom and validation

`Atom` accepts arbitrary messages. Base `onData` raises `NotImplementedError`;
components must emit explicitly. `onStart(data, source)` receives and forwards
arbitrary mutable startup data without reserving keys. `validate_source()` may
reject only structural incompatibility known before values arrive. It must
return normally for a compatible source and raise for an incompatible source;
returning a boolean has no effect. `connect()` validates every target before
changing any connection. Fan-out passes one shared object reference.
Dataclass-based Atoms use `@dataclass(eq=False)` at every decorated inheritance
level so stateful graph nodes retain identity equality and object hashing.

Prefer a real class, ABC, or minimal runtime protocol for structural checks.
Validate conditional message values in `onData`; do not introduce graph-wide
type inference, capability negotiation, or automatic emission checking.

## Execution and recovery

Execution models consume absolute targets, retain only the newest target for
their natural identity, and use Book state plus working orders to derive the
next broker action. Every model has a stable unique configured `name`; persist
and recover that name. Router rules are fixed, ordered, and first-match wins.
Only working orders retain persisted affinity and missing owners fail closed.
Without working orders, current rules own held quantity and recovered direct
targets; startup reassigns an idle target before model recovery.

`SerialTargetExecutionModel` owns one active Contract adjustment at a time and
supports arbitrary same-side resizing. `BracketExecutionModel` owns one
`source_key`, validates initial intent, rejects non-zero same-side resizing,
preserves `position_id` through an episode, and attaches brackets only after a
complete entry fill. Its stop-loss is critical; take-profit is optional and a
missing take-profit is not a sync failure. Regular closes share the active
brackets' OCA group rather than cancelling protection before submitting the
close. `BracketExecutionModel` and its bracket-leg hierarchy belong together in
`bracket_execution.py`; keep the generic execution boundary and serial target
model in `execution_models.py`. Recovery must rebind callbacks to current live
Trade objects and derive work from Book rather than replaying old intent.

Book owns order/fill/state persistence and blotter queries. Controller alone
submits/cancels broker orders, registers OrderInfo immediately, handles status,
Fill and commission events, rebinds Trades, and reconciles aggregate Contract
positions. Do not call the broker from Book or calculate Portfolio policy
inside it.

Physical persistence is limited to `orders`, `state`, and `blotter`. State
identities are `position:{source_key}`,
`target:{execution_model_name}:{conId}`, and
`portfolio:{portfolio_key}`. Orders use actual IB identifiers and preserve
complete serialized Trade plus normalized Fill/Execution evidence. Fill
application must stay idempotent by execution key and ordered before state
projection writes.

## Futures audit and testing

PandasSignalModel audit symbols are
`{source_key}_{ACTIVE.localSymbol}_{run_started_at}`. ACTIVE comes from the
selector regardless of a transaction's Contract role; NEXT-only changes do not
rotate history. A successful ACTIVE change writes a new complete generation,
then only new rows append. Default persistence uses an ordered `DRAIN` sink; no
failed calculation may create a generation. Saving uses the selector's ACTIVE
Contract even when the emitted Signal uses `Atom.which_contract=NEXT`.

Every public export needs a usage-focused Google-style, Sphinx-compatible
docstring and focused pytest coverage. Tests must cover structural connection
validation, immutable messages, STATE/EVENT semantics, absolute target
supersession, execution recovery, Fill deduplication, router affinity, and
one-to-one episode attribution as applicable.

This is a direct-cutover package. Do not add forwarding modules, compatibility
aliases, legacy schema reads, or dual writes. The standalone migration script
may be tested against fakes; never execute it against a real database during
implementation.
