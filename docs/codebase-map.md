# Haymaker Codebase Map

Last updated: 2026-07-15.

## High-Level Purpose

Haymaker is a Python framework for building Interactive Brokers trading systems on top of `ib_insync`. It has three main operating surfaces:

- live strategy execution as a long-running event-driven process,
- historical data download from Interactive Brokers into Arctic/Mongo-backed stores,
- dataframe-first research and vector backtesting utilities.

The package is still alpha-stage. Some public docs describe the live backtester bridge as in development, while the research package contains an active vector backtester and stop engine.

## Architecture Overview

The live execution side is built around `Atom` pipelines. `Atom` provides event wiring, contract lookup, and shared access to process-owned runtime services. Strategy code composes streamers, signal processors, portfolio sizing, and execution models into event chains.

Live runtime services are assembled in `haymaker/runtime.py` by `LiveRuntime`:

- shared `ib_insync.IB` client,
- contract registry for qualified current/next contracts,
- persisted strategy/order state machine,
- controller for broker/state reconciliation and order gateway,
- startup contract-detail initialization and streamer startup jobs.

`RuntimeContext` is the passive container exposed to `Atom` instances. It
holds only ready runtime services, the supervisor restart callback, and
strategy futures-roll policies; it does not construct services or inspect the
user module.

The `haymaker` console command owns live composition and logging: it configures
Haymaker, starts threaded logging handlers, creates `LiveRuntime`, and imports
the user strategy module so module-level pipelines are built against its
already-installed `RuntimeContext`. Blocks register their futures-roll policy
as they are constructed. The CLI then hands the composed runtime to the shared
`App`. The app-lifetime `Controller` starts
its periodic sync, health-check, and daily UTC futures-roll timers once on the
active event loop when `Controller.run()` first executes. Live and dataloader
runtimes use the same application and supervisor lifecycle.

The dataloader is a separate command-line path. It connects to IB, schedules historical-data tasks, observes IB pacing restrictions, and writes pandas frames through the async datastore interface. `Manager` owns the historical request policy (`bar_size`, `wts`, `max_lookback_days`, `useRTH`, `gap_fill_mode`) and the run-scoped `now` value; worker sessions execute generated jobs rather than carrying independent request-policy defaults. The current supported dataloader backend is Arctic, with the library name derived from `wts` and `barSize`.

The research package is intentionally separate from live execution. It works directly with pandas dataframes and NumPy/Numba kernels to validate signal timing, stops, synthetic data, and performance without depending on live `Atom` pipelines.

## Module Responsibilities

### Live Execution Core

- `haymaker/base.py`: `Atom`, event connection primitives, contract descriptor, contract-change handling, and `Pipe` composition support.
- `haymaker/cli.py`: `haymaker` and `dataloader` console-script entrypoints,
  a shared command shell for explicit profile config and threaded-logging
  lifecycles, and user strategy module loading with failed-import rollback in
  `sys.modules`.
- `haymaker/app.py`: shared Linux top-level `App.run()` lifecycle, application
  runtime protocol, supervisor composition, graceful `SIGTERM`, and propagation
  of unexpected workload failures after cleanup.
- `haymaker/runtime.py`: `LiveRuntime`, the live composition root that builds
  IB/state/controller services, installs a passive `RuntimeContext`, and owns
  contract-detail initialization, workload startup, reconnect cleanup, and
  final state flushing. Startup jobs retain the live streamer registry
  populated during strategy import.
- `haymaker/supervisor/`: IB socket supervisor package for connections it owns.
  It owns workload task lifecycle, broker auto-recovery waits,
  probes, restart coalescing, and reconnect retry pacing. Its run loop evaluates
  each state through a race between state completion, lifecycle requests, and
  workload completion. The supervisor consumes only the narrow `start()` and
  `stop()` workload contract. `App` binds its restart callback and
  connection-unavailable event to the application runtime so live controller
  sync can abort during broker recovery, restart, or shutdown. It does not
  manage the gateway process.
- `haymaker/controller/`: order/position reconciliation, execution verification, futures rolling, emergency modes, and error handling. Controller sync retries broker connection and broker-position freshness failures, back-reports known fills before position comparison, can request one reconnect before corrective mutations on the first sync after hold/startup, then queries broker/local state directly for order and position checks while `sync_brackets.py` owns bracket/protection testing and remedies and `Controller.sync()` owns retry and trading-disable decisions.
- `haymaker/trader.py`: thin order placement/cancel/modify wrapper around `ib_insync.IB`.
- `haymaker/state_machine.py`: persisted strategy and order state, rejection tracking, active positions, and locks.
- `haymaker/contract_registry.py`, `contract_selector.py`, `details_processor.py`: broker contract qualification, futures selection, metadata normalization.

### Strategy Pipeline Components

- `haymaker/streamers.py`: historical, market-data, real-time-bar, and tick streamers that emit `Atom` events.
- `haymaker/block.py`: dataframe-to-signal strategy block base classes; can
  persist generated bar data and register per-strategy futures-roll policy.
- `haymaker/signals.py`: binary signal processors converting strategy output into `OPEN`, `CLOSE`, or `REVERSE` actions.
- `haymaker/portfolio.py`: position sizing layer.
- `haymaker/execution_models.py`: converts portfolio/action data into IB orders, including bracket/stop/take-profit handling.
- `haymaker/bracket_legs.py`: bracket-order leg abstractions.
- `haymaker/dfaggregator.py`, `aggregators.py`: bar aggregation and market-data persistence helpers.

### Persistence and Logging

- `haymaker/datastore/`: synchronous and asynchronous store abstractions, ArcticStore, collection naming, futures readers, and deprecated store helpers.
- `haymaker/databases.py`: cached MongoDB client and health-check registration.
- `haymaker/blotter.py`, `saver.py`: transaction logging sinks such as CSV and Mongo-backed savers.
- `haymaker/logging/`: centralized YAML and queue-listener lifecycle setup,
  custom handler implementations, one listener thread per configured
  destination, and optional Telegram delivery. `App` installs the package's
  compact asyncio exception callback on the active loop so otherwise-unhandled
  loop failures use the same configured destinations.

Background queues use one shutdown policy. `DRAIN` queues are critical: item
failures and drain timeouts escape final cleanup. `DISCARD` queues are
best-effort: failures are logged and pending final work is dropped. State saves
use `DRAIN`; Arctic fire-and-forget writes and transient aggregation use
`DISCARD`.

### Dataloader

- `haymaker/dataloader/dataloader.py`: producer/worker queue, download task
  orchestration, and store writes.
- `haymaker/dataloader/runtime.py`: dataloader runtime construction and adapter
  for supervised IB connection ownership, using client ID `1` by default.
- `haymaker/dataloader/contract_selectors.py`: contract selection from CSV/source inputs, especially futures.
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
- Research code usually imports from `haymaker.research`, `haymaker.research.stop`, or `haymaker.research.backtester`.
- Sphinx docs are built from `docs/source` with `make html` from the `docs/` directory.

## Data Flow

### Live Execution Flow

1. The `haymaker` CLI creates `LiveRuntime`. It assembles live services, creates
   `StartupJobs` around the live streamer registry, and installs the passive
   `RuntimeContext` on `Atom` before importing the user strategy module.
2. User strategy module-level code builds `Atom` pipelines and registers streamers.
   Each block also registers its `auto_roll_futures` policy in the context.
3. `App` starts the IB watchdog and waits for a successful historical-data probe.
4. `Controller.run()` starts its app-lifetime timers once on the active event
   loop, reads or initializes state, then `Controller.sync()` races the
   reconciliation pass against the supervisor's connection-unavailable event.
   If the supervisor enters broker recovery, restart, or shutdown, sync aborts
   without disabling trading. Otherwise the internal sync pass runs a bounded
   retry loop around a sync coordinator. Each coordinator pass first checks
   broker connection and validates broker position freshness, relinks current
   `ibi.Trade` objects to local records, back-reports known completed fills,
   runs order/position reconciliation against direct broker and state-machine
   reads, and returns `False` after broker verification failures or recovery
   actions so sync can retry the checks before disabling trading. If unresolved
   order or position mismatches remain on the first pass, the coordinator can
   ask the controller to reconnect before local order pruning, broker order
   cancellation, or strategy-position correction is allowed on a later pass.
   Non-retryable unsafe states raise `SyncBrokenStateError`, which disables
   trading immediately. A failed controller run still permits startup jobs to
   provide monitoring while outbound trading remains disabled.
5. `StartupJobs` downloads contract details, updates the contract registry, logs restart state, resets timeouts, and runs all registered streamers.
6. Streamers emit market data into strategy blocks.
7. Blocks add strategy fields and emit dictionaries.
8. Signal processors create `action`, `target_position`, and existing-position context.
9. Portfolio sizing adds `amount`.
10. Execution models create IB orders and call `Controller.trade()`.
11. Controller registers orders, reconciles broker events, updates the state machine, and sends blotter records when enabled; sync failures disable further outbound trading. The global `controller.missing_brackets` option controls bracket/protection handling: `ignore` skips bracket checks, `warn` logs local bracket-record mismatches and broker positions without stop-loss protection, and `remove` also cancels obsolete bracket/closing orders and closes local strategy positions whose expected local bracket records are missing.

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
5. A producer submits work to an asyncio queue.
6. Workers call IB historical-data requests under pacer restrictions.
7. Downloaded chunks are buffered by range and passed to `HistorySink` at the
   configured chunk threshold or a correctness boundary such as range completion
   or session cleanup. `HistorySink` concatenates each batch with stored data and
   writes a complete new version through the async datastore; Arctic owns final
   cleaning and metadata updates. The returned first bar timestamp is validated
   before it drives the next request boundary.
8. Supervisor recovery within the same process resumes in-memory active jobs
   before discovering new work. A full process stop writes no separate
   dataloader checkpoint; the next process rediscovers remaining work from
   persisted datastore boundaries.

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

Config is assembled in `haymaker/config/config.py` from:

1. command-line options,
2. YAML override file selected from the command line,
3. YAML override file selected through environment configuration,
4. `HAYMAKER_` environment variables,
5. default YAML files.

Important config files:

- `haymaker/config/base_config.yaml`: live execution defaults.
- `haymaker/config/dataloader_base_config.yaml`: dataloader defaults.
- `haymaker/logging/logging_config.yaml`: live logging defaults.
- `haymaker/logging/dataloader_logging_config.yaml`: dataloader logging defaults.

Important environment variables:

- `HAYMAKER_HAYMAKER_CONFIG_OVERRIDES`: live execution YAML override path after `HAYMAKER_` prefix stripping.
- `HAYMAKER_DATALOADER_CONFIG_OVERRIDES`: dataloader YAML override path after `HAYMAKER_` prefix stripping.
- `HAYMAKER_LOGGING_CONFIG`, `HAYMAKER_LOGGING_PATH`, and other top-level `HAYMAKER_` keys: quick overrides.

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

- Some docs still describe backtesting as non-functional, while the research package has an active refactored backtester. Clarify whether that note refers only to live-strategy simulation.
- `pyproject.toml` has a mypy ignore override for `backtester` with a comment to remove after fixing it.
- Several modules contain explicit TODO/deprecated comments, especially dataloader futures selection, research numba tools, store deprecations, and old backtester utilities.
- `haymaker/__init__.py` is empty; most public imports are exposed through subpackages, especially `haymaker.research`.
- `ConfigMaps.parse_yaml()` uses `yaml.unsafe_load`, which supports Python object construction in config files but makes config files trusted code.

## Risky Areas

- Research timing semantics are the highest-risk area. Do not move signals, blips, positions, execution prices, lower-frequency availability points, or stop events across bars without focused tests.
- `upsample()` must preserve the rule that lower-frequency values become available when the grouped bar completes. `position` must not be upsampled.
- `stop_loss()` treats `blip` as generated events and shifts internally, while `position` is already executable state. `distance` and `scheduled_close` Series must match the dataframe index exactly.
- Python and Numba implementations in the stop engine and backtester engine must stay behaviorally identical.
- Controller sync and reconciliation touches live broker state, state-machine records, blotter output, and order cancellation/close logic. Sync correction actions should only run after broker position sources agree and after known completed fills have been replayed into local position records; if broker connection or broker position validation fails, the coordinator returns a retryable failed pass and sync retries before disabling trading for non-convergence. If the supervisor marks the connection unavailable during broker recovery, restart, or shutdown, the public sync wrapper cancels the in-flight pass without treating it as unsafe state. On unresolved order or position mismatches, the first pass can request a broker reconnect before corrective mutations are attempted. Later sync checks query broker/local state directly; broker stop-loss exposure is reported by bracket sync, while missing-local-bracket emergency closes are based on the affected local strategy position.
- Futures rolling changes active contracts, next-contract selection, and strategy state; changes can cause live trading differences.
- Dataloader pacing and gap-fill scheduling can trigger IB pacing violations or silently create incomplete stores if date boundaries are wrong.
- Config files can instantiate Python objects through YAML tags; operational config must be treated as trusted and reviewed.

## AGENTS.md Notes

The repo-root `AGENTS.md` contains the project-wide development rules. There is
also a scoped `haymaker/research/AGENTS.md` for timing-sensitive research code.
The root guidance records the standard focused checks, warns against importing
`haymaker.app` in focused tests, and identifies
`haymaker.supervisor.ConnectionSupervisor` as the owner of IB socket recovery.
It also records the timeout/probe-first recovery rule and points runtime work
away from treating every broker message as a direct restart trigger. During
broker-degraded waits, `updateEvent` and `1102` are only hints to probe
recovery; failed probes should not reset the recovery grace timer.

Dashboard is experimental and should not be looked at.
