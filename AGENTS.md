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
  early entry candidate used by atoms such as strategy blocks to avoid opening
  positions close to a roll. `Atom.which_contract` selects the role exposed by
  that atom; it does not redefine which contract is ACTIVE. OPEN uses the
  block-selected contract, CLOSE uses the strategy's persisted held contract
  (`Strategy.active_contract`), and `FutureRoller` permits held contracts that
  are either ACTIVE or NEXT and rolls holdings outside that set. A NEXT-only
  change does not require market-data back-adjustment. Selectors are rebuilt on
  each supervised workload start using one timezone-naive UTC timestamp, and
  live operation relies on the IB-driven daily workload restarts to refresh
  these roles. Strategy block collections use the root `contract.symbol`, not
  expiry-specific `localSymbol`, so an early NEXT change does not split the
  persisted strategy series.
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
  `RuntimeContext` before the CLI imports the strategy module. Blocks register
  their own `auto_roll_futures` policy while they are constructed. Strategy
  composition may use the context's narrow `FrameStoreProvider` to build fully
  configured persistence dependencies; the runtime does not inspect imported
  module data.
- Create restart-enabled `Timeout.from_atom()` instances from `onStart()` or
  later, after the supervisor restart callback has been bound. Zero-time and
  debug timeouts remain safe during pipeline construction.
- CLI entrypoints own logging setup and shutdown. Every configured destination
  handler runs behind its own queue/listener thread; messenger handlers such as
  Telegram are optional YAML configuration, not runtime dependencies.
- Queue shutdown uses one policy: `DRAIN` is critical and propagates processing
  failures or drain timeouts, while `DISCARD` is best effort and logs failures.
  State-save queues drain. Async Arctic queued sinks default to `DISCARD`.
  Awaited `AsyncDataStore` mutations finish an already-started database call
  before propagating cancellation. The dataloader uses only this awaited
  contract and completes each response's persistence and in-memory state
  transition before a supervised restart can resume the job.
- `AsyncDataStore` methods are all awaited; successful mutation return means
  backend completion. Queue-only persistence uses the separate `QueuedDataSink`
  contract and explicit `enqueue_*` methods, whose return means queue acceptance.
- Datastore symbol naming is supplied when a store is constructed and is not
  replaced afterward. Framework-provided naming policies are frozen, consumers
  treat injected stores as fully configured, and dataframe blocks hold their
  optional datastore dependency per instance rather than through class state.
- Live dataframe consumers never construct or discover persistence through
  runtime state. Strategy module composition builds stores through
  `RuntimeContext.frame_store_provider` and injects them: `DfAggregator`
  requires `AsyncDataStore`, persisted streamers accept `AsyncDataStore | None`,
  and dataframe blocks accept `QueuedDataSink | None`.
- CLI entrypoints load framework configuration once. Live and dataloader
  configuration stay grouped by owning target where practical until that target
  constructs itself from its mapping. Controller one-run actions belong under
  `controller.startup`. Logging and dataloader `download` remain user-facing
  subsystem groups composed across closely related objects. Live storage contains
  only `base_directory`, `mongodb.client`, and `mongodb.database`; dataloader
  storage contains only `base_directory` and `mongodb.client`. Dataframe library
  names and save frequency belong to strategy composition and consumer
  constructors, not framework storage configuration.
  Bundled base profiles must enumerate supported settings, pin effective
  command defaults, and keep a concise inline comment after every setting.
  Environment variables may select a profile YAML file but must not directly
  override individual settings. Strategy parameters remain user-module Python
  data. Do not change real local `.env` files or credential files.
- Use `tests/runtime_helpers.py` and the `atom_runtime` /
  `atom_runtime_factory` fixtures for tests that need `Atom` runtime services.
  Install custom `ib`, state machine, contract registry, controller, restart
  callbacks, frame-store provider, and contract details through those fixtures
  instead of scattering ad hoc runtime monkeypatches.
- See `docs/codebase-map.md` for the current repository map.

Dashboard is experimental and should not be looked at.
