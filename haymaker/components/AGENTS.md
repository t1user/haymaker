# Components package guidance

`haymaker.components` is the discoverable public toolbox for user-composed
trading pipelines. Keep `Atom`, `Pipe`, `Controller`, `Book`, runtime services,
persistence infrastructure, contract selection, and broker infrastructure
outside this package. Add every supported public component to the explicit
`haymaker.components.__all__`; private implementation helpers may remain
unexported.

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

`BinarySignalProcessor` accepts only scalar `-1/0/1` values and exposes
`OpposingSignalPolicy.CLOSE` or `REVERSE`. `BinaryEntryExitSignalProcessor`
accepts only SignalPair, uses entry while flat and exit while positioned, and
never reverses directly. Both may opt into blocked-direction checks. A complete
STOP_LOSS or TAKE_PROFIT fill that flattens the episode sets the block; the
first actual fill of a permitted opposite OPEN clears it. CLOSE and ROLL do not
change it.

## Atom and validation

`Atom` accepts arbitrary messages. Base `onData` raises `NotImplementedError`;
components must emit explicitly. `onStart(data, source)` receives and forwards
arbitrary mutable startup data without reserving keys. `validate_source()` may
reject only structural incompatibility known before values arrive, and
`connect()` validates every target before changing any connection. Fan-out
passes one shared object reference.

Prefer a real class, ABC, or minimal runtime protocol for structural checks.
Validate conditional message values in `onData`; do not introduce graph-wide
type inference, capability negotiation, or automatic emission checking.

## Execution and recovery

Execution models consume absolute targets, retain only the newest target for
their natural identity, and use Book state plus working orders to derive the
next broker action. Every model has a stable unique configured `name`; persist
and recover that name. Router rules are fixed, ordered, and first-match wins.
Active persisted affinity overrides current rules and missing model names fail
closed.

`SerialTargetExecutionModel` owns one active Contract adjustment at a time and
supports arbitrary same-side resizing. `BracketExecutionModel` owns one
`source_key`, validates initial intent, rejects non-zero same-side resizing,
preserves `position_id` through an episode, and attaches brackets only after a
complete entry fill. Its stop-loss is critical; take-profit is optional and a
missing take-profit is not a sync failure. Regular closes share the active
brackets' OCA group rather than cancelling protection before submitting the
close. Recovery must rebind callbacks to current live Trade objects and derive
work from Book rather than replaying old intent.

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
then only new rows append. Audit sinks must use ordered `DRAIN`; no failed
calculation may create a generation.

Every public export needs a usage-focused Google-style, Sphinx-compatible
docstring and focused pytest coverage. Tests must cover structural connection
validation, immutable messages, STATE/EVENT semantics, absolute target
supersession, execution recovery, Fill deduplication, router affinity, and
one-to-one episode attribution as applicable.

This is a direct-cutover package. Do not add forwarding modules, compatibility
aliases, legacy schema reads, or dual writes. The standalone migration script
may be tested against fakes; never execute it against a real database during
implementation.
