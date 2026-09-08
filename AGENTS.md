# General Coding Rules

- Prefer minimal, surgical changes.
- Do not rewrite unrelated code.
- Preserve existing architecture unless instructed otherwise.
- Explain tradeoffs before major refactors.
- Prefer readability over cleverness.
- Avoid adding dependencies unless justified.
- Run tests after changes when possible.
- If tests fail, explain whether the failure is related to your changes.
- Keep functions focused and short.
- Avoid unnecessary abstraction.
- Prefer explicit code over metaprogramming.

# Workflow

- Before editing, briefly explain understanding of the task.
- For non-trivial tasks, create a short implementation plan.
- After changes, summarize modified files and rationale.
- Update relevant documentation when architecture, setup, commands, conventions,
  workflows, or important assumptions change.
- For research code under `haymaker/research`, read
  `haymaker/research/AGENTS.md` before editing.
- Avoid importing `haymaker.app` in tests unless explicitly needed. It sets up
  logging and imports runtime singletons; prefer lower-level modules for
  focused tests.
- When asked to review logs, check `docs/log-review-guidance.md` first. If the
  requested review is not supervisor-related, use that file only for general
  review discipline and focus on the specific component or behavior the user
  asked about.
- Create docstrings for any new functions/classes/methods
- Use google-style, sphinx compatible docstrings
- If changing user-relevant behaviour, scan existing documentation and update any sections relevant to this newly changed behaviour

# Git

- Never commit unless explicitly instructed.
- When committing, add `[llm]` at the end of the commit message.
- Before committing, make sure any file with secrets is included in `.gitignore`.
- Never push.
- Do not modify `.env` files.
- Do not delete files without explanation.
- Run `mypy` and `pytest`, do not commit if there are any issues.

# Python Preferences

- Prefer type hints.
- Prefer `pathlib` over `os.path`.
- Use `pytest` to create tests.
- Use `black` for formatting.
- Add docstrings for new functions and classes.

# Naming

- Event handlers connected directly to `eventkit.Event` or `ib_insync` events
  should use camelCase callback names.
- For `ib_insync` events, prefer direct correspondence with the event name:
  `orderStatusEvent` -> `onOrderStatusEvent`,
  `updateEvent` -> `onUpdateEvent`,
  `timeoutEvent` -> `onTimeoutEvent`.
- Exception: handlers for `ib.errorEvent` should be named `onErrEvent`, not
  `onErrorEvent`, because IB uses this event for many informational broker
  messages that are not real errors, and callback names can surface in logs.
- Internal lifecycle hooks, injected callbacks, helpers, and ordinary methods
  should use standard Python snake_case.

# Logging

- Normal operational logs must not contain the word `error` unless they report
  a genuine error requiring attention. Expected, successfully recovered
  connection interruptions are lifecycle events, not errors. Avoid leaking
  exception class names such as `TimeoutError` into routine recovery messages.

# Validation

- Run the narrowest meaningful test command for the changed area first.
- For broad Python changes, run:

```bash
python -m pytest
```

- For focused research changes, run:

```bash
python -m pytest tests/test_research
```

- For typing and lint checks when touching research code, run when practical:

```bash
python -m mypy haymaker/research tests/test_research
python -m flake8 haymaker/research tests/test_research --select=F401,F821,F841,E501
```

- Run Black normally across all relevant paths. If sandbox restrictions prevent
  Black from completing, run the same command outside the sandbox:

```bash
.venv/bin/python -m black --check --fast --target-version py312 <paths>
```

# Project Notes

- This is an Interactive Brokers trading framework built around `ib_insync`,
  event-driven runtime components, a historical dataloader, dataframe-first
  research tools.
- Live trading, controller sync, futures rolling, and order reconciliation are
  high-risk areas. Keep changes especially narrow and well verified there.
- Futures roll scheduling is app-lifetime behavior. Schedule the daily roll once
  for the app process, not per supervisor connection/workload cycle; reconnects
  and workload restarts must not create additional roll timers.
- Futures contract roles are intentionally distinct. A selector's `ACTIVE` is
  the current market-data and roll-reference contract, while `NEXT` is an
  early entry candidate used by SignalModels to avoid opening
  positions close to a roll. `Atom.which_contract` selects the role exposed by
  that atom; it does not redefine which contract is ACTIVE. OPEN uses the
  signal-selected contract, CLOSE uses Book's persisted held contract, and
  REVERSE completely closes the old episode before opening the incoming
  Contract. Book keeps held and pending-target Contracts and bracket inputs
  separate; PortfolioWrapper never resolves the held Contract. Recovery derives
  continuation from persisted state rather than a transient filled callback.
  `FutureRoller` defaults to rolling only selector `past_contracts` into ACTIVE,
  retaining NEXT and all later eligible expiries. Custom FutureRollPolicy
  instances select triggers and destinations; stable occurrence labels prevent
  repeated fixed-schedule rolls after recovery. ContractRegistry maps qualified expiry
  `conId` values to their registered blueprint series; rolling never guesses
  identity from symbol fields. A NEXT-only
  change does not require market-data back-adjustment. Selectors are rebuilt on
  each supervised workload start using one timezone-naive UTC timestamp, and
  live operation relies on the IB-driven daily workload restarts to refresh
  these roles. Signal audit generations use ACTIVE `localSymbol` plus the
  process `run_started_at`; a NEXT-only change does not rotate audit history.
  `Atom.contract_selector` is a required runtime-backed property: access raises
  until the Atom has a Contract and the registry has initialized its selector.
  `Atom.contract_blueprint` returns a copy of the registered declaration.
  Registry lookup accepts qualified members and covers the full chain. Distinct
  declarations with overlapping qualified conIds fail rather than creating
  ambiguous identity. Explicit nth selection raises when unavailable; NEXT
  alone retains its intentional last-available fallback.
- IB/TWS connection outages, especially around a broker's daily restart period,
  are expected and should normally be recoverable. Do not treat a connection
  outage alone as an unsafe broker/local state; emergency trading disablement
  should be reserved for failed recovery, unreconciled state, or confirmed
  order/position safety issues.
- `haymaker.supervisor.ConnectionSupervisor` owns IB socket recovery for live
  trading and dataloader runs. It does not manage or restart the gateway
  process. Route new restart triggers through its `request_restart()` method.
  Broker messages are categorized as restart requests, broker-wait signals, or
  recovery hints: broker-wait signals move the supervisor into broker recovery
  wait while connected, `timeoutEvent` and probes remain active health checks,
  and `updateEvent` or `1102` may probe recovery while already waiting.
  The dataloader has no connection modes: `DataloaderRuntime` creates its own
  `IB` object and always runs through the shared application and supervisor;
  see `haymaker/dataloader/AGENTS.md`.
- Controller sync treats a successful `reqPositionsAsync()` result as the
  authoritative broker-position snapshot for that pass. Cached/fresh
  disagreement retries locally; request timeout or failure requests
  supervisor-owned recovery. Position correction is deferred while attributed
  OPEN/CLOSE work remains active. An applied one-to-one correction aligns the
  persisted target to the corrected quantity, clears a flattened episode's
  recovery inputs, and preserves its direction block.
- Graceful shutdown is not currently a broad architecture priority. Terminal
  `Ctrl-C` has historically been acceptable. Before service-manager deployment,
  prefer minimal signal hardening for `SIGINT`/`SIGTERM`: request supervisor stop,
  allow normal runtime cleanup to unwind, and drain critical async save queues
  with a short timeout. Do not introduce a broad shutdown framework unless a
  concrete cleanup need is identified.
- Linux is the supported runtime OS. The shared `App` handles the first
  `SIGTERM` as a graceful supervisor stop and restores default signal handling
  so a second `SIGTERM` can terminate stuck cleanup.
- The supported live execution model is one user strategy and one
  `RuntimeContext` per process. Process-global registries such as
  `Streamer.instances` are not reset for same-process application reuse.
- `LiveRuntime` assembles live services and installs a ready, passive
  `RuntimeContext` before the CLI imports the strategy module. SignalModels
  register their Contract blueprints while they are constructed but do not own
  futures-roll policy. User SignalModels return
  `SignalCalculation(value, metadata, as_of)` while the base supplies source,
  resolved Contract, SignalType, and `created_at`. Observation `as_of` remains
  distinct because IB bars are left-labelled. Strategy composition may use the
  context's narrow `FrameStoreProvider` to build fully configured persistence
  dependencies. `PandasSignalModel(persistence=True)` is the narrow exception:
  it resolves one model-owned default persistence object from a runtime factory.
  The runtime still does not inspect imported module data.
- `EventTimeout` is a general user-owned event inactivity monitor; supervised
  workload restarts never cancel it. Positive intervals must be constructed
  on the running event loop, and owners must call `cancel()` when their own
  lifetime ends. `MarketDataTimeout.from_atom()` adds Contract-session and
  supervisor behavior and must be created from `onStart()` or later, after
  details and the restart callback are available. `LiveRuntime` cancels all
  market-data timeouts when each supervised workload stops. Runtime defaults
  live in `haymaker.config.TimeoutPolicy`, not in the public components
  package.
- `HistoricalDataStreamer` initial `reqHistoricalDataAsync()` requests
  intentionally use `timeout=0`. Legitimate large backfills may take many
  minutes, so elapsed time alone must not cancel the request or trigger a
  restart. Keep post-initialization stale-update monitoring separate. Prefer
  non-cancelling elapsed-time logging for diagnostics, and add a configurable
  hard limit only if there is evidence of requests hanging while the broker
  connection remains healthy.
- CLI entrypoints own logging setup and shutdown. Every configured destination
  handler runs behind its own queue/listener thread; messenger handlers such as
  Telegram are optional YAML configuration, not runtime dependencies.
- Queue shutdown uses one policy: `DRAIN` is critical and propagates processing
  failures or drain timeouts, while `DISCARD` is best effort and logs failures.
  Book mutations and default Signal frame persistence queues drain. Async
  Arctic queued sinks otherwise default to `DISCARD`.
  Awaited `AsyncDataStore` mutations finish an already-started database call
  before propagating cancellation. The dataloader uses only this awaited
  contract and completes each response's persistence and in-memory state
  transition before a supervised restart can resume the job.
- `AsyncDataStore` methods are all awaited; successful mutation return means
  backend completion. Queue-only persistence uses the separate `QueuedDataSink`
  contract and explicit `enqueue_*` methods, whose return means queue acceptance.
- Datastore symbol naming is supplied when a store is constructed and is not
  replaced afterward. Framework-provided naming policies are frozen, consumers
  treat injected stores as fully configured, and each persisted
  `PandasSignalModel` owns its `SignalFramePersistence` state rather than
  sharing generations.
- Strategy module composition may build custom stores through
  `RuntimeContext.frame_store_provider` and inject them into both market-data
  components. Otherwise, `FuturesPandasAggregator()` and
  `HistoricalDataStreamer(datastore=True)` resolve the same runtime-cached
  market-data store by bar size, data type, and RTH policy; custom stores
  bypass that default and `False` disables only the streamer's lookup.
  `PandasSignalModel.persistence` supports
  `False`, runtime-default `True`, or a custom non-blocking
  `SignalFramePersistence`; persistence enqueue failure never suppresses Signal
  emission.
- CLI entrypoints load framework configuration once. Live and dataloader
  configuration stay grouped by owning target where practical until that target
  constructs itself from its mapping. Controller one-run actions belong under
  `controller.startup`. Logging and dataloader `download` remain user-facing
  subsystem groups composed across closely related objects. Live storage contains
  only `base_directory`, `mongodb.client`, and `mongodb.database`; dataloader
  storage contains only `base_directory` and `mongodb.client`. Custom dataframe
  library names belong to strategy composition. Runtime defaults are configured
  under `market_data_store.library` for broker bars and
  `signal_persistence.library` for Signal calculations; save frequency belongs
  to its consumer.
  Bundled base profiles must enumerate supported settings, pin effective
  command defaults, and keep a concise inline comment after every setting.
  Environment variables may select a profile YAML file but must not directly
  override individual settings. Strategy parameters remain user-module Python
  data. Do not change real local `.env` files or credential files.
- `Atom` remains an arbitrary-message composition primitive. It never mutates
  startup/data payloads automatically, base `onData` raises, connection
  validation for required upstream capabilities occurs before wiring, message
  envelopes are validated in `onData`, and fan-out shares one object reference.
  Do not add `input_type` or `output_type` message declarations.
  Dataclass-based Atoms use `@dataclass(eq=False)` at every decorated
  inheritance level so graph nodes retain identity equality. Built-in trading
  components live only under `haymaker.components`; routing, execution models,
  and bracket execution are grouped under `haymaker.components.execution` and
  re-exported by the root components package.
- Built-in messages are frozen `Signal -> PositionProposal -> PositionTarget`
  envelopes. Signal values are finite scalars or `SignalPair(entry, exit)`.
  PositionTarget requires a concrete non-zero `conId` and its quantity is always
  an absolute setpoint. Direct targets address their exact concrete conId and
  omit source_key/intent; one-to-one targets identify their episode by source_key. Signal,
  PositionProposal, and PositionTarget are intentionally unhashable; Contracts
  and nested metadata remain shared mutable
  objects. `PositionIntent` is mandatory only on the one-to-one
  PortfolioWrapper/BracketExecutionModel path, where PortfolioWrapper transfers
  it to the target as an initial assertion.
- Reuse `haymaker.validators` for primitive normalization of aware datetimes,
  finite numbers, copied read-only mappings, non-empty strings, and IB
  Contracts. Keep domain-specific checks with their owning component.
- `Book` owns typed position/target/order/roll recovery, Fill-derived direct
  physical attribution, fill idempotence, the critical ordered persistence
  queue, and blotter access. Explicit reset retains order/Fill evidence and
  persists a per-target cutoff used when rebuilding direct exposure. Controller
  owns broker calls, reconciliation, submission, rebinding, and fill/commission
  event handling. Do not move Portfolio calculations or broker calls into Book.
- Direct Portfolio consumes Signals and allocates among concrete Contracts.
  Each target is the absolute setpoint for its conId; source allocations belong
  to Portfolio, not to broker Fill attribution. Multiple expiries may coexist.
  The one-to-one path uses a signal processor, `PortfolioWrapper`, and
  `PositionAllocator`. Execution models have stable unique configured names;
  preserving a name across deployments promises recovery-compatible behavior.
  PortfolioStateMixin optionally exposes explicit load_state/save_state under
  portfolio_key, defaulting to Book; independent persistence is permitted.
  Execution targetReachedEvent forwards completion via Atom feedback, including
  through Router. It follows fill accounting, carries persisted target fields,
  and may repeat after recovery. Custom Portfolio feedback must be idempotent.
  Current Router rules always select the model. At startup, every active direct
  `TARGET_ADJUSTMENT` must still select its persisted owner; disagreement,
  missing recovery state, or ambiguous ownership blocks that Router locally
  without cancelling orders or disabling Controller trading. One-to-one and
  non-adjustment orders do not participate. Held quantity or an idle latest
  target does not pin an old model: current rules take ownership, and all idle
  direct targets must be routable before any reassignment is applied. Final
  process close warns if active adjustments remain so routing or model changes
  can be deferred until they finish.
- One-to-one scalar processors accept only `-1/0/1` and make opposing CLOSE
  versus REVERSE policy explicit. Paired processors use SignalPair entry while
  flat and exit while positioned. Protective STOP_LOSS and TAKE_PROFIT fills
  set a direction block only when they flatten the episode; the first actual
  fill of a permitted opposite OPEN clears it. CLOSE and ROLL preserve it.
- Bracket-managed positions require critical stop-loss protection; take-profit
  orders are optional and their absence is not a sync failure. Regular closes
  join the active protective orders' OCA group so IB cancels the remaining
  exits only after one exit fills.
- Target execution models register exactly one process-wide mode-specific
  `FutureRollExecutor` family. Controller owns the single daily schedule,
  stale-holding discovery, and startup recovery coordination; direct or bracket
  executors own durable sequencing. Modes cannot be mixed. Direct rolls wait
  for endpoint adjustments and persist an idempotent
  target transfer: old target zero, destination target plus old target; newer
  explicit targets supersede that snapshot. Bracket rolls preserve
  `source_key`/`position_id`, roll broker-net exposure, and require an active
  replacement stop before completion; take-profit replacement is optional.
  `BracketExecutionModel` enables automatic rolling by default and
  `auto_roll_futures=False` is the explicit per-source opt-out. Preserve mode and
  executor name while persisted roll work is incomplete.
  Before bracket roll submission, wait for all unprocessed source OPEN/CLOSE
  orders and refresh episode quantities after accounting settles. Skip sources
  that became flat, retain their completion attribution, and preserve offsets
  from already processed sources. Changed episode identity or an impossible
  remaining physical allocation blocks before further broker work.
- Explicit account reset gives pre-existing order cancellations a bounded grace
  period, then submits liquidation orders even when some cancellations remain
  unconfirmed because flattening is the priority. An incomplete liquidation
  leaves Book recovery state intact and prevents startup from enabling trading.
- Use `tests/runtime_helpers.py` and the `atom_runtime` /
  `atom_runtime_factory` fixtures for tests that need `Atom` runtime services.
  Install custom `ib`, Book, contract registry, controller, restart
  callbacks, frame-store provider, and contract details through those fixtures
  instead of scattering ad hoc runtime monkeypatches.
- See `docs/codebase-map.md` for the current repository map.

Dashboard is experimental and should not be looked at.
