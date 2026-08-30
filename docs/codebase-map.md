# Haymaker Codebase Map

Last updated: 2026-08-12.

## High-Level Purpose

Haymaker is a Python framework for building Interactive Brokers trading systems on top of `ib_insync`. It has four main operating surfaces:

- live strategy execution as a long-running event-driven process,
- historical data download from Interactive Brokers into Arctic/Mongo-backed stores,
- experimental event-driven simulation of strategy component graphs from saved
  dataloader bars, and
- dataframe-first research and vector backtesting utilities.

The package is still alpha-stage. The event-driven simulator is an experimental
extra and is not fully functional. It is separate from the active research
vector backtester and stop engine.

## Architecture Overview

The live execution side is built around `Atom` pipelines. `Atom` is an
arbitrary-message composition primitive providing validated event wiring,
contract lookup, and shared access to process-owned runtime services. The
pre-built trading toolbox lives under `haymaker.components` and communicates
through frozen `Signal`, `PositionProposal`, and absolute `PositionTarget`
message envelopes.

Live runtime services are assembled in `haymaker/runtime.py` by `LiveRuntime`:

- shared `ib_insync.IB` client,
- contract registry for qualified current/next contracts,
- typed persisted Book for orders, fills, positions, targets, and Portfolio
  recovery state,
- private Mongo client lifecycle and health service,
- controller for broker/state reconciliation and order gateway,
- startup contract-detail initialization and streamer startup jobs.

`RuntimeContext` is the passive container exposed to `Atom` instances. It
holds only ready runtime services, process `run_started_at`, supervised
`workload_generation`, the narrow `FrameStoreProvider` composition capability,
the cached market-data store factory, the per-model default Signal persistence
factory, the supervisor restart callback, and source futures-roll policies; it
does not construct services or inspect the user module.

The `haymaker` console command owns live composition and logging: it configures
Haymaker, starts threaded logging handlers, creates `LiveRuntime`, and imports
the user strategy module so module-level pipelines are built against its
already-installed `RuntimeContext`. SignalModels register Contract blueprints
as they are constructed but do not own futures-roll policy. Strategy module
code may use the provider to build fully configured dataframe stores and inject
them into consumers. The CLI then hands the composed runtime to the shared
`App`. The app-lifetime `Controller` starts its periodic sync, health-check, and
daily UTC futures-roll timers once on the active event loop when
`Controller.run()` first executes. Live and dataloader runtimes use the same
application and supervisor lifecycle.

One-to-one `BracketExecutionModel` instances register their source roll policy;
automatic Controller-owned rolling is the default and explicit opt-out is
available for a source that manages its own roll.

The dataloader is a separate command-line path. It connects to IB, schedules historical-data tasks, observes IB pacing restrictions, and writes pandas frames through the async datastore interface. `DataloaderRuntime` decomposes the merged `download` mapping across `Manager` request policy and `DataloaderSession` worker count, owns Mongo/Arctic composition, and injects datastore construction into `Manager`. `Manager` owns the run-scoped `now` and derives the library from data type and bar size, while contract selectors share a target-owned `FuturesSelectionPolicy`. Arctic is the only supported dataloader backend.

The event-driven backtester is a programmatic experimental extra. It installs a
simulation runtime, invokes a user strategy factory, reads bars and
contract-specific metadata through the read-only `BacktestDataStore` protocol,
and replays the data through
the live-style component graph. Backtester-owned adapters replace Interactive
Brokers, wall-clock sessions, and external framework persistence; live socket
supervision and broker-state reconciliation are outside its scope.

The research package is intentionally separate from live execution. It works directly with pandas dataframes and NumPy/Numba kernels to validate signal timing, stops, synthetic data, and performance without depending on live `Atom` pipelines.

## Module Responsibilities

### Live Execution Core

- `haymaker/base.py`: `Atom`, event connection primitives, contract descriptor, contract-change handling, and `Pipe` composition support.
- `haymaker/cli.py`: `haymaker` and `dataloader` console-script entrypoints.
  Each parses its command once, loads its profile configuration, starts
  threaded logging, builds the appropriate runtime, and owns user strategy
  module loading with failed-import rollback in `sys.modules`.
- `haymaker/config/`: safe YAML loading, deterministic profile merging,
  section-based `LiveConfig` and `DataloaderConfig` aggregates, retained typed
  storage settings, and side-effect-free command-line parsers. Targets own
  their field defaults and necessary mapping conversion.
- `haymaker/app.py`: shared Linux top-level `App.run()` lifecycle, application
  runtime protocol, supervisor composition, graceful `SIGTERM`, and propagation
  of unexpected workload failures after cleanup.
- `haymaker/runtime.py`: `LiveRuntime`, the live composition root that builds
  IB/Book/controller services, installs a passive `RuntimeContext`, and owns
  contract-detail initialization, workload startup, reconnect cleanup, and
  final state flushing. Startup jobs retain the live streamer registry
  populated during strategy import.
- `haymaker/config/settings.py`: typed storage aggregates and the validated
  runtime `TimeoutPolicy`; public timeout components live in the component
  toolbox.
- `haymaker/supervisor/`: IB socket supervisor package for connections it owns.
  It owns workload task lifecycle, broker auto-recovery waits,
  probes, restart coalescing, and reconnect retry pacing. Its run loop evaluates
  each state through a race between state completion, lifecycle requests, and
  workload completion. The supervisor consumes only the narrow `start()` and
  `stop()` workload contract. `App` binds its restart callback and
  connection-unavailable event to the application runtime so live controller
  sync can abort during broker recovery, restart, or shutdown. It does not
  manage the gateway process.
- `haymaker/controller/`: broker submission, order/position reconciliation,
  execution verification, fill and commission event processing, Trade
  rebinding, futures rolling, emergency modes, and broker message handling.
  Sync retries broker-position freshness failures, back-reports known fills
  before comparison, and requests supervisor-owned recovery before correction
  where required.
- `haymaker/trader.py`: thin order placement/cancel/modify wrapper around `ib_insync.IB`.
- `haymaker/book.py`: typed order/fill evidence, one-to-one PositionState,
  direct TargetState, Portfolio recovery mappings, rejection tracking,
  active-order ownership queries, blotter access, and one ordered critical
  persistence queue. Book performs no broker calls or allocation.
- `haymaker/validators.py`: shared primitive normalization for aware datetimes,
  finite numbers, read-only mapping copies, non-empty strings, and IB Contract
  identity, plus IB request/order field validators. Domain-specific validation
  remains with the owning component.
- `haymaker/contract_registry.py`, `contract_selector.py`, `details_processor.py`: broker contract qualification, futures selection, metadata normalization.

### Strategy Pipeline Components

- `haymaker/components/messages.py`: frozen scalar-or-paired `Signal`,
  `SignalPair`, `PositionProposal`, and absolute `PositionTarget` messages plus
  signal, intent, and open-ended order role enums. Message metadata is copied
  only at the top level; contained Contracts and nested values remain shared.
- `haymaker/components/streamers.py`: broker market-data sources. A persisted
  historical streamer reads the stored endpoint to shorten its next IB history
  request.
- `haymaker/components/aggregators.py`: incremental bar-object aggregation and
  count, volume, tick, time, and pass-through eventkit filters.
- `haymaker/components/dataframe_aggregators.py`: the separate public
  DataFrame aggregation family. `FuturesPandasAggregator` restores, stitches,
  saves, and emits complete futures history; count, volume, tick, and time
  groupers recalculate completed bars from cumulative DataFrames.
- `haymaker/components/timeouts.py`: user-owned generic event inactivity
  callbacks and workload-owned, market-session-aware stale-data monitoring.
- `haymaker/components/signal_models.py`: general SignalModel with a
  framework-owned Signal envelope and public SignalCalculation result boundary,
  plus dataframe-based PandasSignalModel with configurable row metadata and
  optional ordered calculation-data persistence.
- `haymaker/components/signal_processors.py`: scalar and paired one-to-one
  binary processors implementing STATE/EVENT, configurable opposing-signal,
  entry/exit, and stopped-direction lock semantics.
- `haymaker/components/portfolio.py`: direct account-wide Portfolio,
  one-to-one PortfolioWrapper, and PositionAllocator boundary.
- `haymaker/components/execution/`: public execution subpackage. Its initializer
  aggregates the leaf modules' exports for both focused imports and promotion
  through `haymaker.components`.
  - `models.py`: the common stateful execution-model boundary and serial
    Contract target convergence.
  - `router.py`: fixed first-match target routing plus startup validation that
    active direct adjustments are still assigned to their persisted owners.
  - `brackets.py`: one-to-one bracket episode execution and its
    user-configurable protective-order legs. Regular closes join the episode's
    OCA group, keeping stop protection active until an exit fills.
- `haymaker/components/__init__.py`: registers public component modules and
  aggregates their module-owned, non-overlapping `__all__` exports into the
  supported package toolbox.

### Persistence and Logging

- `haymaker/datastore/`: synchronous and asynchronous store abstractions,
  the awaited `AsyncDataStore`, queued `QueuedDataSink`, and composition-only
  `FrameStoreProvider` protocol, cached market-data store factory, Signal
  calculation persistence policy and factory, ArcticStore, immutable
  construction-time symbol naming, futures readers, and deprecated store
  helpers.
- `haymaker/databases.py`: focused runtime-owned `MongoService` for lazy client
  construction, initial ping, reuse, and health-check registration, plus the
  private Arctic implementation of the narrow `FrameStoreProvider` contract.
- `haymaker/blotter.py`, `saver.py`: explicitly configured transaction logging
  sinks such as CSV and Mongo-backed savers.
- `haymaker/logging/`: centralized YAML and queue-listener lifecycle setup,
  custom handler implementations, one listener thread per configured
  destination, and optional Telegram delivery. `App` installs the package's
  compact asyncio exception callback on the active loop so otherwise-unhandled
  loop failures use the same configured destinations.

Background queues use one shutdown policy. `DRAIN` queues are critical: item
failures and drain timeouts escape final cleanup. `DISCARD` queues are
best-effort: failures are logged and pending final work is dropped. All Book
mutations share one ordered `DRAIN` queue; default Signal frame persistence also
uses a dedicated `DRAIN` sink. Other async Arctic queued sinks and transient
aggregation default to `DISCARD`. The dataloader uses only awaited datastore
mutations and therefore owns no datastore queue policy.

`AsyncDataStore` methods are all awaited: successful mutation return means the
backend operation finished. `QueuedDataSink` uses explicit `enqueue_*` methods:
return means queue acceptance, while final handling depends on the sink's
`DRAIN` or `DISCARD` shutdown policy.

Datastore symbol naming is fixed when each store wrapper is constructed.
Framework-provided naming policies are frozen, stores expose the configured
policy read-only, and consumers treat injected stores as fully configured.
The runtime `MarketDataStoreFactory` caches awaited stores by bar size,
`whatToShow`, and `useRTH`; its symbols include those dimensions plus the
Contract. `FuturesPandasAggregator()` resolves that default at startup, while
`HistoricalDataStreamer(datastore=True)` resolves the same store during
construction. The streamer uses `False` to disable persisted-endpoint lookup;
the aggregator always requires a default or custom datastore.
`PandasSignalModel.persistence=False` disables calculation history,
`True` creates an independent policy from the runtime default factory, and a
custom `SignalFramePersistence` object overrides that default. Queue acceptance
precedes emission, but failure to enqueue only removes the Signal's audit
reference and never suppresses the Signal.

### Dataloader

- `haymaker/dataloader/dataloader.py`: producer/worker queue, download task
  orchestration, and store writes.
- `haymaker/dataloader/runtime.py`: dataloader runtime construction and adapter
  for supervised IB connection ownership, using client ID `1` by default.
- `haymaker/dataloader/contract_selectors.py`: strict contract-field validation
  and selection from CSV/source inputs, especially futures.
- `haymaker/dataloader/pacer.py`: request throttling and pacing-violation tracking.
- `haymaker/dataloader/scheduling.py`: `TaskPlanner`, `BackfillRangePlan`,
  `UpdateRangePlan`, `GapFillRangePlan`, and pure heuristic or schedule/session gap
  filtering helpers.
- `haymaker/dataloader/store_wrapper.py`: `AsyncStoreView` for read-only
  scheduling boundaries with explicit bar-size policy and `HistorySink` for
  raw historical-data persistence.
- `haymaker/dataloader/time_policy.py`: canonical historical-date policy for
  `formatDate=2`, keeping intraday points as UTC-aware datetimes and
  daily/weekly/monthly points as dates.

### Event-driven Backtester (Experimental)

- `haymaker/backtester/`: programmatic strategy-graph simulation built around
  `Backtester(store, start=None, end=None, initial_cash=100_000.0,
  slippage_ticks=0.0)` and `await backtester.run(strategy_factory)`, with
  backtester-local runtime, broker, clock, session, fill, and persistence
  adapters.
- The input store uses the dataloader's `whatToShow` plus bar-size library naming.
  Bars define available sessions, while contract expiry and other
  contract-specific values come only from datastore metadata.
- Complete MKT, LMT, and STP fills and fixed OCA brackets form the initial
  execution boundary. Missing commission metadata means zero commission;
  tick-based slippage and fixed brackets require a positive `minTick`.

### Research and Backtesting

- `haymaker/research/signal_converters.py`: canonical timing vocabulary and conversions between `signal`, `blip`, `transaction`, and `position`.
- `haymaker/research/upsampling.py`: aligns lower-frequency data to higher-frequency execution bars. Ordinary values propagate after availability; canonical `blip` / `close_blip` events and raw provenance columns remain sparse.
- `haymaker/research/stop/`: public `stop_loss()`, lazy `before_close()` / `BeforeClose`, Python reference stop engine, and Numba stop engine.
- `haymaker/research/backtester/`: transaction-frame pipeline, Numba/Python perf engine, Pyfolio statistics, and legacy utilities.
- `haymaker/research/bootstrap/`: block and regime/state bootstrap generators for synthetic OHLC paths.
- `haymaker/research/optimizer.py`, `tester.py`, `plotting.py`, `grouper.py`, `candlesticks.py`: research workflow helpers around parameter sweeps, plotting, grouping, and indicators.

## Entry Points

- `haymaker strategy.py [options]` builds the live runtime, imports the user
  strategy module, and then starts the framework-owned `App`. `App.run()` runs
  the top-level supervisor
  coroutine through standard `asyncio.run()`; nested-loop patching is not used.
  The CLI flushes threaded logging after application, construction, or strategy-
  import failure. One user strategy per process is the supported lifecycle.
- `dataloader contracts.csv [options]` uses the same command shell, maps the
  positional source file into dataloader configuration, and builds a
  dataloader runtime for the shared `App`.
- `await Backtester(...).run(strategy_factory)` is the programmatic entry point
  for experimental event-driven simulation; there is no backtester CLI.
- Research code usually imports from `haymaker.research`, `haymaker.research.stop`, or `haymaker.research.backtester`.
- Sphinx docs are built from `docs/source` with `make html` from the `docs/` directory.

## Data Flow

### Live Execution Flow

1. The `haymaker` CLI creates `LiveRuntime`. It constructs Book, optionally
   restores persisted state before any broker connection, assembles the other
   live services, creates `StartupJobs` around the live streamer registry, and
   installs the passive `RuntimeContext` on `Atom` before importing the user
   strategy module.
2. User strategy module-level code builds `Atom` pipelines and registers streamers.
3. `ConnectionSupervisor` connects the IB client and waits for a successful
   historical-data probe.
4. `Controller.run()` starts its app-lifetime timers once on the active event
   loop, then `Controller.sync()` races the
   reconciliation pass against the supervisor's connection-unavailable event.
   If the supervisor enters broker recovery, restart, or shutdown, sync aborts
   without disabling trading. Otherwise the internal sync pass runs a bounded
   retry loop around a sync coordinator. Each pass compares cached positions
   with a fresh `reqPositionsAsync()` result. A disagreement retries locally;
   only request timeout or failure asks the supervisor to recover broker state.
   The successful response is the sole broker-position snapshot used by that
   pass. The coordinator then relinks current `ibi.Trade` objects to local
   records, back-reports known completed fills, and compares Book quantity with
   that snapshot. Position correction is deferred for Contracts with active
   one-to-one OPEN/CLOSE work. If unresolved order or actionable position
   mismatches remain on the first pass, the coordinator can ask the controller
   to request one supervised workload restart before local order pruning,
   broker order cancellation, or position correction is allowed on a later
   pass. Applied one-to-one correction also aligns the persisted target to the
   authoritative quantity and clears episode recovery inputs when flat, so a
   supervised recovery cannot reopen the corrected position from a stale
   setpoint.
   Non-retryable unsafe states raise `SyncBrokenStateError`, which disables
   trading immediately. An aborted controller run skips startup jobs for that
   workload generation; a failed run still permits those jobs to provide
   monitoring while outbound trading remains disabled.
5. `StartupJobs` downloads contract details, rebuilds contract selectors from
   one timezone-naive UTC timestamp, logs restart state, and runs all
   registered streamers. Each streamer creates a market-session-aware timeout
   for its subscription. `LiveRuntime` cancels those workload-owned monitors
   before supervised cleanup and again when the workload exits; unrelated
   user-owned `EventTimeout` instances are untouched. Selector `ACTIVE`
   identifies the current market-data and roll-reference contract; `NEXT` is
   an early new-entry candidate. Existing positions retain their persisted held
   contract, and the futures roller acts only after that contract leaves the
   allowed `ACTIVE`/`NEXT` set.
6. Streamers and aggregators emit market data into SignalModels. Custom models
   calculate only a SignalCalculation; the base supplies source, resolved
   Contract, SignalType, and local creation time.
7. SignalModels emit immutable raw Signals. Observation `as_of` remains distinct
   from `created_at`, including for IB's left-labelled bars. PandasSignalModel
   may persist a calculation generation named from source, ACTIVE contract, and
   process start; NEXT-only changes do not rotate it.
8. In the one-to-one flow, a binary processor emits PositionProposal and
   PortfolioWrapper allocates one absolute PositionTarget. In direct mode,
   Portfolio owns input state and may emit targets for several Contracts.
9. ExecutionRouter optionally selects one stable named model using current
   first-match rules. Before model recovery, active TARGET_ADJUSTMENT ownership
   must agree with those rules; mismatch blocks that Router locally. Idle
   recovered targets are reassigned only after every target can be routed.
10. Execution models persist the newest target, derive required work from Book
    quantity and working orders, and call `Controller.trade()` with explicit
    role/model/source/episode attribution.
11. Controller registers complete OrderInfo immediately, applies normalized
    Fill evidence idempotently, attaches late CommissionReports regardless of
    optional blotter configuration, updates Book projections, rebinds Trades,
    and sends source/position-attributed blotter records when enabled. The
    global `controller.missing_brackets` option controls critical stop-loss
    reconciliation; take-profit orders are optional.

### Dataloader Flow

1. `dataloader` loads config, constructs `DataloaderRuntime`, and passes it to
   the shared `App`. The runtime creates its owned `ib_insync.IB` client and
   session before the connection supervisor is assembled.
2. The supervisor connects the socket and waits for a successful historical-data probe before starting dataloader work.
3. Contract source data is expanded into IB contracts.
4. The async store view inspects the Arctic-backed store and normalizes
   scheduling boundaries according to the dataloader date policy.
   `Manager` obtains any async schedule inputs and passes them to
   `TaskPlanner`, which creates update, backfill, and optional gap-fill ranges.
   Continuous futures use IB's empty-`endDateTime` latest-ended request shape
   and do not schedule internal gap-fill ranges.
5. A producer submits work to a source-ordered asyncio queue bounded to
   `max(1, number_of_workers // 4)` jobs. Unknown CSV headers fail before this
   stage and before any broker contract-detail request.
6. Workers call IB historical-data requests under pacer restrictions.
7. Downloaded chunks are buffered by range and passed to `HistorySink` at the
   configured chunk threshold or a correctness boundary such as range completion
   or session cleanup. `HistorySink` concatenates each batch with stored data and
   writes a complete new version through the awaited datastore API; completion
   metadata uses the same awaited mutation contract. An already-started mutation
   and its buffer/range state transition finish before cancellation propagates,
   preventing restart cleanup from submitting an overlapping rewrite. Arctic
   owns final cleaning and metadata updates. The returned first bar timestamp is
   validated before it drives the next request boundary. The dataloader submits
   no queued datastore mutations.
8. Supervisor recovery within the same process resumes in-memory active jobs
   before discovering new work. A full process stop writes no separate
   dataloader checkpoint; the next process rediscovers remaining work from
   persisted datastore boundaries.

### Event-driven Backtester Flow

1. The caller supplies a read-only `BacktestDataStore` (normally an
   `AsyncDataStore`) configured for a dataloader library and calls
   `await Backtester(...).run(strategy_factory)`.
2. The backtester reconstructs exact Contracts from series metadata, installs an
   isolated simulation runtime, and calls the factory to build the ordinary
   strategy component graph.
3. The replay clock walks the union of saved bar timestamps. Bar presence is the
   sole session signal for each Contract; missing data means no session at that
   point.
4. A completed bar is emitted through the event graph and asynchronous callbacks
   settle before the clock advances. Orders created from that observation become
   eligible only on the Contract's next saved bar.
5. The simulated broker produces complete MKT, LMT, and STP fills and fixed OCA
   cancellation events through the normal controller and Book path. Commission
   defaults to zero when absent; configured slippage and bracket prices require
   metadata `minTick`.
6. Trailing and adjustable orders, partial fills, open-position futures rolls,
   and live sync/recovery are not simulated.

### Research Flow

1. Start from an OHLC dataframe.
2. Add generated strategy fields such as `signal`, `blip`, `close_blip`, or executable `position`.
3. If signals are generated on lower-frequency grouped bars, use `upsample()` before deriving executable state.
4. Convert to the transaction-frame schema with `no_stop()` or `stop_loss()`.
5. Run `perf()` to get stats, daily returns, bar-level results, trade records, and warnings.

## External Integrations

- Interactive Brokers TWS/Gateway through `ib_insync`.
- Haymaker-owned `ConnectionSupervisor` instances for live and dataloader IB
  socket recovery. TWS or IB Gateway process management is
  external. Broker message codes are categorized into restart requests,
  broker-connectivity-lost signals, and informational farm/live-update messages.
  Connectivity-lost signals move the supervisor into broker recovery wait while
  connected; informational farm messages are log context only. IB `10182`
  warnings request a stale-subscription restart after a 180-second quiet period.
  `timeoutEvent` and probes remain active health checks, and `1102` can end a
  broker-connectivity wait when IB reports data maintained. Generic
  `updateEvent` traffic does not advance broker recovery.
- MongoDB through `pymongo`.
- Arctic through `arctic` for dataframe time-series storage.
- pandas, NumPy, and Numba for dataframe research and kernels.
- Pyfolio Reloaded for research performance statistics.
- Sphinx/Furo for docs.
- Optional research extras include Jupyter, matplotlib, `arch`, and `hmmlearn`.

## Configuration and Environment

The CLI assembles framework configuration once through
`haymaker/config/loader.py`. Live and dataloader loading return `LiveConfig`
or `DataloaderConfig`; sections remain mappings until the owning target or
subsystem composition boundary constructs them. Broker-facing Controller
one-run actions are nested under `controller.startup`; persisted-state loading
belongs to `book.restore`. Live storage contains only the shared base directory,
Mongo client arguments, and framework database name. Dataloader
storage uses the narrower `DataloaderStorageSettings`, containing only a base
directory and Mongo client arguments. Custom dataframe library names are
selected during strategy composition. Runtime defaults are
`market_data_store.library` for broker-bar history and
`signal_persistence.library` for Signal calculations; save frequency belongs
to the consuming object.
`DataloaderRuntime` passes its client arguments directly to its private
`MongoService`; the base directory remains shared CLI logging infrastructure.
Runtime components receive their specific section or ready service and do not
read a process-global configuration object.
Strategy-specific parameters remain ordinary Python data in the user module.

Configuration precedence, from lowest to highest, is:

1. bundled profile defaults,
2. an environment-selected profile YAML file,
3. a command-line `--file` YAML file,
4. repeatable typed dotted-path `--set-option` overrides,
5. dedicated command-line switches.

YAML is parsed with a safe duplicate-key-rejecting loader. Mappings merge
recursively, while lists and scalars replace lower-priority values. Unknown
top-level sections fail during loading. Dataloader storage structure is also
validated by the loader; its runtime and target constructors validate
`download`, `pacing`, and futures fields and values. The bundled profiles
enumerate supported settings with concise comments and pin each command's
effective defaults; deployment override files remain partial.

Important config files:

- `haymaker/config/live_base_config.yaml`: live execution defaults.
- `haymaker/config/dataloader_base_config.yaml`: dataloader defaults.
- `haymaker/logging/logging_config.yaml`: live logging defaults.
- `haymaker/logging/dataloader_logging_config.yaml`: dataloader logging defaults.

Important environment variables:

- `HAYMAKER_HAYMAKER_CONFIG_OVERRIDES`: live execution YAML override path.
- `HAYMAKER_DATALOADER_CONFIG_OVERRIDES`: dataloader YAML override path.

Environment variables do not directly override individual settings.

Do not commit real `.env` files. `.gitignore` already ignores `.env`, `.venv`, generated builds, local backtests, and credential files.

## Build, Test, and Tooling Commands

Install editable development environment:

```bash
python -m pip install -e ".[dev]"
```

Run all tests:

```bash
python -m pytest
```

Run focused research tests:

```bash
python -m pytest tests/test_research
```

Run research typing and focused lint checks:

```bash
python -m mypy haymaker/research tests/test_research
python -m flake8 haymaker/research tests/test_research --select=F401,F821,F841,E501
```

Format changed Python files:

```bash
python -m black path/to/file.py
```

Build docs:

```bash
cd docs
make html
```

Run dataloader:

```bash
dataloader contracts.csv -f settings.yaml
```

## Technical Debt

- The event-driven backtester remains experimental and incomplete. Its supported
  order boundary excludes trailing and adjustable orders, partial fills,
  open-position futures rolls, and live synchronization/recovery.
- Several modules contain explicit TODO/deprecated comments, especially dataloader futures selection, research numba tools, store deprecations, and old backtester utilities.
- `haymaker/__init__.py` is empty; most public imports are exposed through subpackages, especially `haymaker.research`.

## Risky Areas

- Research timing semantics are the highest-risk area. Do not move signals, blips, positions, execution prices, lower-frequency availability points, or stop events across bars without focused tests.
- `upsample()` must preserve the rule that lower-frequency values become available when the grouped bar completes. `position` must not be upsampled.
- `stop_loss()` treats `blip` as generated events and shifts internally, while `position` is already executable state. `distance` and `scheduled_close` Series must match the dataframe index exactly.
- Python and Numba implementations in the stop engine and backtester engine must stay behaviorally identical.
- Controller sync and reconciliation touches live broker state, typed Book
  records, blotter output, and order cancellation/close logic. Sync correction
  actions should only run after broker position sources agree and known
  completed fills have been applied. If the supervisor marks the connection
  unavailable, the public sync wrapper cancels the pass without treating it as
  unsafe state. Recovered execution models must rebind current Trade callbacks
  before resuming outstanding targets.
- Explicit account reset gives every pre-existing order cancellation a bounded
  grace period, then submits liquidation orders even if some cancellations
  remain unconfirmed. Incomplete liquidation leaves Book state intact and
  prevents startup from enabling trading.
- Final live-runtime close warns about active TARGET_ADJUSTMENT orders. Treat
  the recorded model name as a recovery-compatibility promise and defer routing
  or implementation changes until those direct orders are terminal.
- Futures rolling changes active contracts, next-contract selection, and Book
  PositionState; changes can cause live trading differences.
- Dataloader pacing and gap-fill scheduling can trigger IB pacing violations or silently create incomplete stores if date boundaries are wrong.

## AGENTS.md Notes

The repo-root `AGENTS.md` contains the project-wide development rules. Scoped
guidance lives in `haymaker/components/AGENTS.md` for public trading contracts,
in `haymaker/dataloader/AGENTS.md` for historical request,
persistence, schema, and validation invariants and in
`haymaker/research/AGENTS.md` for timing-sensitive research code.
The root guidance records the standard focused checks, warns against importing
`haymaker.app` in focused tests, and identifies
`haymaker.supervisor.ConnectionSupervisor` as the owner of IB socket recovery.
It also records the timeout/probe-first recovery rule and points runtime work
away from treating every broker message as a direct restart trigger. During
broker-degraded waits, `updateEvent` and `1102` are only hints to probe
recovery; failed probes should not reset the recovery grace timer.

Dashboard is experimental and should not be looked at.
