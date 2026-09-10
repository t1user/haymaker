# Components: strategy composition and extension

Read the root `AGENTS.md` for workflow, Atom/runtime contracts, accounting and
safety rules. This file owns component-specific guidance. Usage examples and
full APIs are in [the execution guide](../../docs/source/execution.rst);
custom storage configuration is in [the storage guide](../../docs/source/storage.rst).

## Package and public API

- Each public leaf module owns `__all__`. Package initializers aggregate those
  lists; each leaf export is promoted to the public toolbox. Names must be
  unique across modules. Do not maintain duplicate hand-written export lists.
- Keep `aggregators.py` (IB bar transformations) separate from
  `dataframe_aggregators.py` (pandas history, persistence and transformations).
  The latter is public, not an implementation detail.

## Choose the strategy path

```text
One-to-one:
SignalModel -> signal processor -> PositionProposal
  -> PortfolioWrapper(PositionAllocator) -> PositionTarget
  -> BracketExecutionModel -> Controller

Direct:
SignalModel paths -> Portfolio -> PositionTarget(s)
  -> ExecutionRouter (optional) -> SerialTargetExecutionModel -> Controller
```

A process uses one-to-one or direct execution, not both. Do not make the wrapper
a shared Portfolio or make execution models own direct allocation policy.

## Messages and signal production

- `Signal` carries stable `source_key`, Contract, scalar value or
  `SignalPair(entry, exit)`, SignalType, timestamps and metadata.
  STATE replaces desired state; EVENT is a new occurrence, including repetition.
  Raw Signals never carry PositionIntent.
- `PositionProposal` preserves the original Signal, adds direction -1/0/1, and
  requires OPEN/CLOSE/REVERSE intent.
- `PositionTarget` is an absolute signed setpoint, never a captured current
  quantity or delta. It requires a concrete non-zero conId. Direct targets omit
  source_key/intent; one-to-one targets identify a source episode and require
  intent at acceptance. Numeric target remains authoritative afterward.
- Messages are frozen, keyword-only, intentionally unhashable and validate
  finite values/aware timestamps. Metadata is copied/read-only only at the top
  level; Contracts and nested values remain mutable/shared.
- `as_of` labels the observation; `created_at` is local creation time. Keep
  both: IB bars are left-labelled. Do not reinterpret a bar label as its close.
- Custom SignalModels implement `calculate_signal(data)` and return
  `SignalCalculation(value, metadata, as_of)`; the framework builds the envelope.
  Override `validate_signal_value` for additional domain checks.
- `select_signal_contract()` defaults to `self.contract`. It may choose a
  qualified chain member or `contract_blueprint` for a custom direct Portfolio.
  One-to-one allocation needs a concrete opening Contract. SignalModels do not
  own futures-roll policy.
- Pandas models implement `df(data)`; its last row is authoritative. User code
  owns ordering, duplicate rows and calculation correctness. `signal_fields`
  accepts a field name or an (entry, exit) tuple. `metadata_fields=None` selects
  non-signal fields, an empty collection none, an explicit collection selected
  fields. Override `row_to_calculation(row)`, not envelope construction.

## One-to-one processing and allocation

- Processors query `Book.effective_quantity(source_key)`, including working
  orders, and suppress inputs requiring no action. STATE zero requests flat;
  EVENT zero is ignored.
- `BinarySignalProcessor` accepts only -1/0/1. OpposingSignalPolicy.CLOSE closes
  first; REVERSE requests reversal. `BinaryEntryExitSignalProcessor` accepts
  SignalPair, uses entry while flat and exit while positioned, and never
  directly reverses. Preserve the full transition matrices and paired input.
- Lock-aware behavior is opt-in. A STOP_LOSS or TAKE_PROFIT fill that flattens
  the episode blocks re-entry in that direction. The first actual fill of a
  permitted opposite OPEN clears it; CLOSE and ROLL preserve it.
- Allocators implement `target_for(proposal) -> PositionTarget | None`, preserving
  source, Contract and metadata. FixedSizeAllocator accepts a positive size,
  source-keyed mapping or callable; missing mappings raise. Use None, not a zero
  OPEN allocation, to suppress a proposal.
- PortfolioWrapper transfers mandatory proposal intent and emits at most one
  target. It never selects the held Contract or orchestrates closes/reversals.

## Direct Portfolio policy

- Construct/share a Portfolio instance explicitly; it is not a singleton.
  Implement `process(signal)` yielding zero or more concrete targets. One input
  may change several Contracts.
- `sources=None` permits dynamic membership with no completeness policy. An
  explicit collection rejects unknown sources; use a collection, not a bare str.
  Concrete Portfolios own input state, supported SignalTypes, EVENT accumulation,
  synchronization, `as_of`, duplicate/late inputs and recomputation policy.
- Portfolio chooses source allocations and concrete Contracts, including which
  held expiries to reduce. Execution must not substitute a different expiry.
  `positions_for_blueprint` queries filled holdings via registry membership;
  it does not reconstruct desired per-source allocations from broker net fills.
- Optional `PortfolioStateMixin` exposes explicit `load_state/save_state` under
  `portfolio_key`, defaulting to Book. Persist normalized recovery mappings,
  not raw Signals. Custom backends are allowed but own their lifecycle and
  have no transaction spanning their state and execution targets.
- `targetReachedEvent` follows Book accounting and forwards through Atom
  feedback/Router. It contains persisted target fields, not arbitrary metadata,
  and may repeat after recovery. Roll completion events are likewise not a
  durable queue: custom policies reconcile Book and handle callbacks idempotently.

## Execution, routing and brackets

- Stateful models validate before saving/submitting, retain the newest target
  for their identity, and derive work from Book rather than replaying intent.
  Rebind callbacks to current live Trades during recovery.
- Stable model names promise recovery-compatible implementation/configuration.
  Router rules are fixed, ordered, first-match; no default means fail closed.
  The same model instance may serve several rules and the default, but distinct
  instances must have distinct names. Start models once per workload generation.
- Current rules, not persisted affinity, select models. Active direct
  TARGET_ADJUSTMENT orders must still select their persisted owner. A mismatch,
  missing state, ambiguity or unroutable recovery blocks that Router locally;
  it does not cancel orders or disable Controller globally.
- Idle targets/holdings do not pin a model. Resolve every idle reassignment
  before applying any. Recovery predicates must work from Contract, quantity
  and creation time; metadata-based recovery routing is unsupported.
- SerialTargetExecutionModel permits one active adjustment per concrete conId,
  supports same-side resizing and independently converges multiple expiries.
- BracketExecutionModel owns one source episode: OPEN uses the incoming
  Contract; CLOSE uses Book's held/pending-entry Contract; REVERSE closes that
  episode completely before opening the incoming Contract under a new
  position_id. Hold `contract/bracket_inputs` separately from pending
  `target_contract/target_bracket_inputs`; target acceptance cannot overwrite
  a holding. Reject non-zero same-side resizing.
- Attach protection only after complete entry filling. Stop-loss is critical;
  take-profit is optional. Regular closes join the protective orders' OCA group
  so IB cancels the other exits when one fills.
- Live completion and offline initial recovery share `ensure_entry_brackets`.
  Use the exact episode's saved inputs and normalized weighted entry price, not
  new target inputs or broker net cost. Recovery needs Contract ticks ready;
  it must not depend on a new Signal or commission callback.
- Existing stop evidence, including terminal history, prevents automatic
  reinstallation. A surviving take-profit lends its OCA group/type; a missing
  optional take-profit alone is not repaired. Partial entries, active exits and
  rolls keep their own sequencing. Missing evidence or failed required initial
  stop installation fails explicitly.
- Previously active/cancelled/rejected stops are not reconstructed from entry
  price; broker-maintained trailing state may be lost. Remaining missing-stop
  handling follows `controller.missing_brackets` (ignore/warn/remove, default
  ignore). Do not imply all missing protection automatically disables trading.
- Order options follow built-in fallback < global defaults < model constructor.
  StandardOrderRole values are conventions; custom strings remain valid.

## Futures rolling

- Controller owns discovery and recovery coordination. One configured
  FutureRollExecutor family owns durable sequencing; direct and bracket
  families cannot mix. Preserve mode/executor name while a roll is incomplete.
- Default PastToActiveRollPolicy rolls only selector `past_contracts` into ACTIVE,
  retaining NEXT and later eligible expiries. Custom FutureRollPolicy selects
  triggers and same-series destinations from a date-refreshed selector.
  Stable RollDecision occurrence labels prevent repeated fixed-schedule rolls.
  Custom schedulers may call `controller.future_roller.roll()`.
- Bracket models default to `auto_roll_futures=True`; False is the per-source
  opt-out. SignalModels do not declare this policy.
- Direct rolls wait for endpoint adjustments and save repeat-safe target
  transfers: old target zero, destination target plus old target. Newer explicit
  targets take precedence; serial convergence resumes after completion.
- Bracket rolls preserve source/position_id and move broker-net exposure.
  Refresh all unprocessed episode quantities/identities after OPEN/CLOSE work
  and accounting settle. Skip flattened sources without losing completed logical
  offsets. Block replaced episodes or impossible net allocations before trading.
- After movement, replace protection; the new critical stop must be active
  before roll completion, while take-profit replacement is optional.

## Market data, persistence and timeouts

- FuturesPandasAggregator maintains/saves complete futures dataframe history;
  HistoricalDataStreamer reads its saved endpoint to shorten backfills. They
  share runtime-default market stores by bar size, data type and RTH policy.
  `datastore=False` disables only streamer lookup; the aggregator requires storage.
- Initial historical requests intentionally use `timeout=0`: a long backfill
  alone is not grounds for cancellation/restart. Keep subsequent stale-update
  monitoring separate.
- `PandasSignalModel.persistence` accepts False (the default), True (runtime
  storage defaults), or a custom non-blocking SignalFramePersistence. Each
  model owns its persistence state. Queue acceptance precedes Signal emission
  to attach a reference, but enqueue failure never suppresses the Signal.
  Direct `create_signal()` calls do not save.
- Audit symbols are `{source_key}_{ACTIVE.localSymbol}_{run_started_at}`.
  ACTIVE, not transaction Contract/NEXT, determines generation. A successful
  calculation under changed ACTIVE writes the full frame; later saves append
  new rows. Failed calculation/adjustment creates no generation; previous
  generations stay unchanged. Supervised restart continues the run; a new
  process starts a new run.
- Audit metadata records source, run start and ACTIVE Contract. Futures history
  may be back-adjusted without downstream notification; append-only storage is
  not a record of every historical revision.
- EventTimeout is user-owned, rearms on new events and survives workload restart.
  Positive intervals require a running loop; owners cancel at lifetime end.
  MarketDataTimeout.from_atom is created in onStart or later; it adds sessions
  and supervisor policy. Runtime cancels market-data monitors on workload stop.
  Closed markets restart the deadline at next open; restart-triggered monitors
  remain disarmed even if another transition caused the request to be rejected.
  Open but non-liquid sessions still require fresh data; liquidity is not the
  market-open test.

## Extension checks

Use the root validation commands and runtime fixtures; test public exports.
Cover changed transition matrices, message/connection validation, allocation,
target supersession, order ownership, duplicate fills/callbacks, orderId/permId
rebinding, episode attribution, OCA, partial/full fills, offline recovery,
rolling and ACTIVE/NEXT audit boundaries.
Do not claim complete live safety from the fake broker.
