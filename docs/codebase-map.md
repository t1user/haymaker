# Haymaker Codebase Map

Last updated: 2026-07-01.

## High-Level Purpose

Haymaker is a Python framework for building Interactive Brokers trading systems on top of `ib_insync`. It has three main operating surfaces:

- live strategy execution as a long-running event-driven process,
- historical data download from Interactive Brokers into Arctic/Mongo-backed stores,
- dataframe-first research and vector backtesting utilities.

The package is still alpha-stage. Some public docs describe the live backtester bridge as in development, while the research package contains an active vector backtester and stop engine.

## Architecture Overview

The live execution side is built around `Atom` pipelines. `Atom` provides event wiring, contract lookup, shared access to the global `ib_insync.IB` instance, and shared access to the `StateMachine`. Strategy code composes streamers, signal processors, portfolio sizing, and execution models into event chains.

Runtime singletons are assembled in `haymaker/manager.py`:

- `IB`: shared `ib_insync.IB` client,
- `CONTRACT_REGISTRY`: maps contract blueprints to qualified current/next contracts,
- `STATE_MACHINE`: persisted strategy/order state,
- `CONTROLLER`: broker/state reconciliation and order gateway,
- `JOBS`: startup data acquisition plus streamer execution.

`haymaker/app.py` wraps those singletons in an `ib_insync.Watchdog`, probes broker connectivity after gateway connection, schedules futures rolls, and starts the controller and jobs.

The dataloader is a separate command-line path. It connects to IB, schedules historical-data tasks, observes IB pacing restrictions, and writes pandas frames to a configured datastore.

The research package is intentionally separate from live execution. It works directly with pandas dataframes and NumPy/Numba kernels to validate signal timing, stops, synthetic data, and performance without depending on live `Atom` pipelines.

## Module Responsibilities

### Live Execution Core

- `haymaker/base.py`: `Atom`, event connection primitives, contract descriptor, contract-change handling, and `Pipe` composition support.
- `haymaker/app.py`: application watchdog, connection probe, restart lifecycle, futures-roll schedule, and top-level `App.run()`.
- `haymaker/manager.py`: constructs runtime singletons and injects shared IB/state/contract data into `Atom`.
- `haymaker/controller/`: order/position reconciliation, execution verification, futures rolling, emergency modes, and error handling. Controller sync retries broker connection and broker-position freshness failures, then queries broker/local state directly for order and position checks while `sync_brackets.py` owns bracket/protection testing and remedies and `Controller.sync()` owns retry and trading-disable decisions.
- `haymaker/trader.py`: thin order placement/cancel/modify wrapper around `ib_insync.IB`.
- `haymaker/state_machine.py`: persisted strategy and order state, rejection tracking, active positions, and locks.
- `haymaker/contract_registry.py`, `contract_selector.py`, `details_processor.py`: broker contract qualification, futures selection, metadata normalization.

### Strategy Pipeline Components

- `haymaker/streamers.py`: historical, market-data, real-time-bar, and tick streamers that emit `Atom` events.
- `haymaker/block.py`: dataframe-to-signal strategy block base classes; can persist generated bar data.
- `haymaker/signals.py`: binary signal processors converting strategy output into `OPEN`, `CLOSE`, or `REVERSE` actions.
- `haymaker/portfolio.py`: position sizing layer.
- `haymaker/execution_models.py`: converts portfolio/action data into IB orders, including bracket/stop/take-profit handling.
- `haymaker/bracket_legs.py`: bracket-order leg abstractions.
- `haymaker/dfaggregator.py`, `aggregators.py`: bar aggregation and market-data persistence helpers.

### Persistence and Logging

- `haymaker/datastore/`: synchronous and asynchronous store abstractions, ArcticStore, collection naming, futures readers, and deprecated store helpers.
- `haymaker/databases.py`: cached MongoDB client and health-check registration.
- `haymaker/blotter.py`, `saver.py`: transaction logging sinks such as CSV and Mongo-backed savers.
- `haymaker/logging/`: logging setup, queue logging, asyncio exception handling, and YAML logging configs.

### Dataloader

- `haymaker/dataloader/dataloader.py`: `dataloader` console-script entrypoint, producer/worker queue, download task orchestration, and store writes.
- `haymaker/dataloader/connect.py`: IB connection modes: `watchdog`, `reconnect`, and `wait`.
- `haymaker/dataloader/contract_selectors.py`: contract selection from CSV/source inputs, especially futures.
- `haymaker/dataloader/pacer.py`: request throttling and pacing-violation tracking.
- `haymaker/dataloader/scheduling.py`: task generation for backfill, updates, and optional gap filling.
- `haymaker/dataloader/store_wrapper.py`: datastore access wrapper used by download writers.

### Research and Backtesting

- `haymaker/research/signal_converters.py`: canonical timing vocabulary and conversions between `signal`, `blip`, `transaction`, and `position`.
- `haymaker/research/upsampling.py`: aligns lower-frequency data to higher-frequency execution bars. Ordinary values propagate after availability; canonical `blip` / `close_blip` events and raw provenance columns remain sparse.
- `haymaker/research/utils.py`, `result_analysis.py`: light research utilities
  for data validation/sampling, tick rounding, and backtest result
  decomposition.
- `haymaker/research/indicators/`: indicator package with mean/weighted-mean
  helpers, indicator-adjacent metrics, resampling helpers, technical
  indicators, breakout/channel indicators, VuManChu-style range-filter helpers,
  and generic transformers for converting indicators or prices into discrete
  events, filters, or desired exposure. `haymaker/indicators.py` remains a
  compatibility facade for older notebooks and live strategy imports.
- `haymaker/research/stop/`: public `stop_loss()`, lazy `before_close()` / `BeforeClose`, Python reference stop engine, and Numba stop engine.
- `haymaker/research/backtester/`: transaction-frame pipeline, Numba/Python perf engine, Pyfolio statistics, and legacy utilities.
- `haymaker/research/bootstrap/`: block and regime/state bootstrap generators for synthetic OHLC paths.
- `haymaker/research/grid_search.py`, `tester.py`, `plotting.py`, `grouper.py`, `candlesticks.py`: research workflow helpers around `GridSearch` parameter/dataframe sweeps, plotting, grouping, and indicators.

## Entry Points

- Live strategy scripts import and instantiate `haymaker.app.App`, then call `App().run()`.
- `dataloader` console script maps to `haymaker.dataloader.dataloader:start`.
- Research code usually imports from `haymaker.research`, `haymaker.research.stop`, or `haymaker.research.backtester`.
- Sphinx docs are built from `docs/source` with `make html` from the `docs/` directory.

## Data Flow

### Live Execution Flow

1. User strategy code builds `Atom` pipelines and starts `App.run()`.
2. `App` starts the IB watchdog and waits for a successful historical-data probe.
3. `Controller.run()` reads or initializes state, then `Controller.sync()` runs a bounded retry loop around a sync coordinator. Each coordinator pass first checks broker connection and validates broker position freshness, relinks current `ibi.Trade` objects to local records, runs order/position reconciliation against direct broker and state-machine reads, and returns `False` after broker verification failures or recovery actions so `Controller.sync()` can retry the checks before disabling trading. Non-retryable unsafe states raise `SyncBrokenStateError`, which `Controller.sync()` catches to disable trading immediately.
4. `Jobs` downloads contract details, updates the contract registry, logs restart state, resets timeouts, and runs all registered streamers.
5. Streamers emit market data into strategy blocks.
6. Blocks add strategy fields and emit dictionaries.
7. Signal processors create `action`, `target_position`, and existing-position context.
8. Portfolio sizing adds `amount`.
9. Execution models create IB orders and call `Controller.trade()`.
10. Controller registers orders, reconciles broker events, updates the state machine, and sends blotter records when enabled; sync failures disable further outbound trading. The global `controller.missing_brackets` option controls bracket/protection handling: `ignore` skips bracket checks, `warn` logs local bracket-record mismatches and broker positions without stop-loss protection, and `remove` also cancels obsolete bracket/closing orders and closes local strategy positions whose expected local bracket records are missing.

### Dataloader Flow

1. `dataloader` loads config, creates an `ib_insync.IB` client, and connects by configured run mode.
2. Contract source data is expanded into IB contracts.
3. Writers inspect the target store and schedule backfill, update, and optional gap-fill download containers.
4. A producer submits work to an asyncio queue.
5. Workers call IB historical-data requests under pacer restrictions.
6. Downloaded chunks are normalized, concatenated with stored data, cleaned, and written through the configured datastore.

### Research Flow

1. Start from an OHLC dataframe.
2. Add generated strategy fields such as `signal`, `blip`, `close_blip`, or executable `position`.
3. If signals are generated on lower-frequency grouped bars, use `upsample()` before deriving executable state.
4. Convert to the transaction-frame schema with `no_stop()` or `stop_loss()`.
5. Run `perf()` to get stats, daily returns, bar-level results, trade records, and warnings.

## External Integrations

- Interactive Brokers TWS/Gateway through `ib_insync`.
- `ib_insync.Watchdog` / `IBC` for live app restarts and optional dataloader watchdog mode.
- MongoDB through `pymongo`.
- Arctic through `arctic` for dataframe time-series storage.
- pandas, NumPy, and Numba for dataframe research and kernels.
- Pyfolio Reloaded for research performance statistics.
- Sphinx/Furo for docs.
- Optional research extras include Jupyter, matplotlib, `arch`, and `hmmlearn`.

## Configuration and Environment

Config is assembled in `haymaker/config/config.py` from:

1. command-line options,
2. YAML override file,
3. `HAYMAKER_` environment variables,
4. default YAML files.

The code's actual `ChainMap` order is command line, config file, environment, defaults. Existing `docs/source/configuration.rst` says environment variables take precedence over user YAML; that should be reconciled before relying on docs for operational precedence.

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
dataloader -f settings.yaml
```

## Technical Debt

- `pyproject.toml` declares `readme = "README.md"`, but the repo has `README.rst` and no top-level `README.md`.
- `docs/source/configuration.rst` documents environment-over-YAML precedence, while the current `ConfigMaps.maps` order gives YAML override files precedence over environment values.
- Some docs still describe backtesting as non-functional, while the research package has an active refactored backtester. Clarify whether that note refers only to live-strategy simulation.
- `pyproject.toml` has a mypy ignore override for `backtester` with a comment to remove after fixing it.
- Several modules contain explicit TODO/deprecated comments, especially dataloader futures selection, research numba tools, store deprecations, and old backtester utilities.
- `haymaker/__init__.py` is empty; most public imports are exposed through subpackages, especially `haymaker.research`.
- `ConfigMaps.parse_yaml()` uses `yaml.unsafe_load`, which supports Python object construction in config files but makes config files trusted code.
- Live runtime singletons are constructed at import time in `haymaker/manager.py`, so imports can have side effects and tests need to avoid importing `haymaker.app`.

## Risky Areas

- Research timing semantics are the highest-risk area. Do not move signals, blips, positions, execution prices, lower-frequency availability points, or stop events across bars without focused tests.
- `upsample()` must preserve the rule that lower-frequency values become available when the grouped bar completes. `position` must not be upsampled.
- `stop_loss()` treats `blip` as generated events and shifts internally, while `position` is already executable state. `distance` and `scheduled_close` Series must match the dataframe index exactly.
- Python and Numba implementations in the stop engine and backtester engine must stay behaviorally identical.
- Controller sync and reconciliation touches live broker state, state-machine records, blotter output, and order cancellation/close logic. Sync correction actions should only run after broker position sources agree; if broker connection or broker position validation fails, the coordinator returns a retryable failed pass and `Controller.sync()` retries before disabling trading for non-convergence. Later sync checks query broker/local state directly; broker stop-loss exposure is reported by bracket sync, while missing-local-bracket emergency closes are based on the affected local strategy position.
- Futures rolling changes active contracts, next-contract selection, and strategy state; changes can cause live trading differences.
- Dataloader pacing and gap-fill scheduling can trigger IB pacing violations or silently create incomplete stores if date boundaries are wrong.
- Config files can instantiate Python objects through YAML tags; operational config must be treated as trusted and reviewed.

## AGENTS.md Notes

There is no repo-root `AGENTS.md` file in this checkout. The active guidance for this run came from the user-provided repo instructions, and there is a scoped `haymaker/research/AGENTS.md` for timing-sensitive research code.

Useful repo-root additions, if a root `AGENTS.md` is created later:

- Keep the current minimal-change and no-commit rules.
- Add a pointer to `haymaker/research/AGENTS.md` before editing research code.
- Document the standard focused checks: `python -m pytest tests/test_research`, mypy/flake8 for research changes, and full `python -m pytest` for broader runtime changes.
- Warn that importing `haymaker.app` sets up logging and runtime singletons, so tests should import lower-level modules where possible.

Dashboard is experimental and should not be looked at.
