# Dataloader Package Guidance

This package is the standalone Interactive Brokers historical-data downloader.
It never places orders. Changes can consume IB pacing allowance, rewrite large
Arctic series, or alter historical boundaries, so keep edits narrow and cover
request, restart, and date-policy behavior with focused tests.

## Architecture

- The shared `App` is the process composition root. `DataloaderRuntime` creates
  its owned `IB` client and `DataloaderSession`, then adapts resumable work to
  the common runtime `start()` / `stop()` / `close()` contract.
- The dataloader has no connection-mode abstraction or setting. Its CLI loads
  `DataloaderConfig`, creates `DataloaderRuntime(config)`, and always runs it
  through the shared `App` and `ConnectionSupervisor`. The runtime decomposes
  the `download` mapping across `Manager` and `DataloaderSession`; contract
  selectors retain the target-owned `FuturesSelectionPolicy`. Keep `download`
  as a user-facing run group rather than splitting out worker count solely to
  mirror these internal constructors. The bundled dataloader profile must list
  every supported setting with a concise inline comment. Its `storage` group
  is deliberately narrow: only `base_directory` and `mongodb.client` belong to
  this runtime. The library name is derived from `what_to_show` and `bar_size`.
- The dataloader defaults to client ID `1`, distinct from the live runtime's
  expected client ID `0`. A duplicate client ID is a configuration failure; do
  not retry automatically with another ID.
- `DataloaderSession` owns producer/worker execution, failure collection,
  cancellation, and final buffered-data flushing.
- `Manager` owns source expansion, contract discovery, request policy, the
  run-scoped `now`, datastore construction, active jobs, and broker schedule
  requests needed for planning.
- `TaskPlanner` and the range-plan objects in `scheduling.py` are pure planning
  code. Keep IB calls out of them; obtain broker inputs in `Manager` and pass
  ordinary values into the planner.
- Planned work executes in `update`, `backfill`, then `gap` order.
- Contract jobs preserve source order through a bounded FIFO queue sized as
  `max(1, number_of_workers // 4)`. Keep the lower bound: asyncio treats a
  queue size of zero as unbounded.
- `DownloadJob` owns request progression for one contract. `DownloadContainer`
  owns one range's buffered chunks and next request boundary.
- `AsyncStoreView` is the read-only scheduling boundary. `HistorySink` is the
  persistence boundary.
- `AsyncStoreView` loads existing data and metadata once for planning. It
  requires explicit bar-size policy and exposes normalized boundaries without
  hiding broker or datastore refreshes behind properties.
- `DataloaderSession` reads request policy from generated jobs and `Manager`; do
  not restore independent compatibility defaults such as session-level
  `bar_size` or `whatToShow`.

See `docs/source/dataloader.rst` for user-facing behavior and
`docs/codebase-map.md` for repository-level flow.

## Historical Time Policy

- Keep `formatDate=2` hardcoded for historical bars and head timestamps.
- Intraday scheduling points must be timezone-aware `datetime` values and are
  normalized to UTC.
- Daily, weekly, and monthly scheduling points are `date` values. Do not compare
  these directly with intraday datetimes.
- `Manager.now` is one run-scoped snapshot. Do not recompute freshness cutoffs
  during worker execution.
- IB sometimes returns bars even when `reqHeadTimeStamp` returns no value. Keep
  the bounded fallback probe: at most five calendar years, further clamped by
  `max_lookback_days` and known small-bar or expired-future limits.

## IB Request Policy

- Route historical data, head timestamps, schedules, and contract-details calls
  through the session-scoped `RequestPacing` object.
- Keep documented IB capacities in code, not user YAML. The global historical
  capacity applies to bars of `30 secs` or less and schedules, and reserves one
  slot for supervisor probes. Bars of `1 min` or longer retain only the open
  request bound because IB has lifted their hard historical pacing rules.
- Head timestamps and contract details use the separate discovery bucket.
  Preload the store and apply known availability rules first, then request a
  head timestamp only when an older backfill range can still exist.
- `pacing.allowance_fraction < 1.0` reserves capacity for other clients. Values
  above `1.0` are intentionally allowed for experimentation but exceed
  published IB limits and may trigger throttling.
- `BID_ASK` historical requests consume double pacing weight.
- Connection-class failures escape the session for supervisor recovery.
  Historical-data job request failures are recorded and workers continue.
  Discovery, local processing, or datastore failures abort the session.
- Do not add arbitrary elapsed-time restart triggers. IB may legitimately
  throttle for long periods.
- Reject unknown or ambiguous contract specifications during discovery instead
  of sending predictable failed historical requests.
- Reject unknown CSV headers before constructing selectors or making broker
  requests. Programmatic selector construction must follow the same rule; do
  not silently discard unsupported contract fields.
- Keep `ConnectionSupervisor` workload-agnostic. Dataloader-specific behavior
  belongs in this package, and live runtime remains the owner of live-trading
  recovery.
- Dataloader logging is separate from live Telegram alerts. Reserve high
  severity for failures requiring attention, not ordinary recovery waits.
- Preserve request-stage wording in diagnostics: `prepared` is pre-pacer,
  `Local pacer delaying` is a client-side wait, and `Submitted ... to IB` means
  broker response time has begun. Locally inferred availability skips must say
  `local policy`; do not phrase them as broker responses.
- Periodic status is an operational snapshot, not a dump of empty buckets. Keep
  worker utilization, named IB waits, named queued contracts, and actual producer
  backpressure visible; omit inactive request families.

## Availability And Contract Rules

- Bars of `30 secs` or smaller are clamped to IB's six-month availability
  window.
- Expired futures backfill is clamped to two years before exact expiry.
- Expired options, futures options, and warrants are skipped when exact expiry
  proves historical data unavailable.
- Other unavailable-data cases rely on IB responses and datastore metadata.
- Missing metadata is never an error. `backfill_exhausted: true` suppresses only
  older backfill. `update_exhausted: true` suppresses only terminal updates for
  contracts with exact expiry strictly before the run boundary; it never
  suppresses live updates, backfill, or gap filling.
- An intraday terminal update no more than one bar interval beyond the stored
  endpoint is marked `update_exhausted` without an IB request. Longer terminal
  ranges are marked only after their IB-backed range completes.
- Continuous futures require an empty `endDateTime`, produce at most one
  latest-ended request range, and do not schedule internal gap fills.

## Persistence

- Arctic versioned writes are deliberate. `HistorySink` reads the existing
  series, concatenates downloaded data, and creates a new complete version;
  do not replace this with append semantics without addressing gap safety.
- `save_every_chunks` is the only routine batching policy. Range completion,
  terminal empty responses, and session cleanup flush incomplete batches as
  correctness boundaries.
- Stop workers before the final flush so persistence cannot race an active
  download.
- Dataloader Arctic stores use a dedicated `DRAIN` queue. All queued datastore
  writes, including metadata, are critical and processing failures or drain
  timeouts must propagate from orderly shutdown. The shared/default async
  Arctic policy remains `DISCARD`; select the dataloader policy at store
  construction without adding another async-store method.
- Standalone Ctrl-C cancellation must finish the session flush before shared
  application shutdown drains datastore background queues. Do not restore
  `ib_insync.util.patchAsyncio()` in the CLI; nested-loop patching is for
  notebooks, not this standalone command.
- A supervisor restart preserves active jobs in memory. A new process derives
  remaining work from persisted datastore boundaries; there is no separate
  checkpoint file.
- Arctic remains responsible for final sorting, duplicate removal, metadata,
  and collection naming. Do not normalize raw downloaded frames in
  `HistorySink`.
- Arctic is the only supported backend and there is no datastore-backend YAML
  selector. Do not accept an arbitrary persistence object unless ownership of
  sorting, duplicate removal, metadata, and naming is explicitly defined.

## Gap Filling

- Supported modes are `off`, `heuristic`, `schedule`, and `auto`; default is
  `off`.
- `schedule` must fail if no usable IB schedule is available. `auto` falls back
  to the fixed two-pass heuristic.
- Schedule requests use the same `download.use_rth` setting as historical-data requests.
- Weekend gaps are ignored. Repeated short no-data patterns are learned only
  for the current run and must not be persisted in datastore metadata.
- Keep heuristic and schedule comparison functions pure and testable without an
  IB connection.

## Extension And Validation

- Keep configuration validation at the nearest existing owner. The loader
  rejects unknown root/storage keys and constructs `DataloaderStorageSettings`;
  `DataloaderRuntime` rejects unknown `download` and `pacing` keys and composes
  the targets; `FuturesSelectionPolicy`, `Manager`, `DataloaderSession`,
  `DownloadJob`, and `RequestPacing` validate their own values.
- Keep the dataloader schema limited to settings it consumes. Supported
  `storage` leaves are `base_directory` and `mongodb.client`; supported
  `download` leaves are `source`, `bar_size`, `what_to_show`,
  `max_lookback_days`, `gap_fill_mode`, `use_rth`, `save_every_chunks`, and
  `number_of_workers`; supported `pacing` leaves are `no_restriction` and
  `allowance_fraction`; supported `futures` leaves are `selector`,
  `full_chain_spec`, and `current_index`.
- Treat booleans distinctly from integers during validation. Worker and save
  counts are positive integers; lookback is a positive integer or `None`;
  `use_rth` and pacing bypass are booleans; pacing allowance is finite and
  positive. Preserve the existing bar-size, data-type, gap-mode, and futures
  validators.

- Do not change `haymaker/durationStr_converters.py` as part of dataloader
  helper cleanup; it is trading-specific and has separate risk.
- Do not add gateway process management or Watchdog/IBC behavior to
  `ConnectionSupervisor` without an explicit architecture decision.
- Never place orders during IB integration experiments. When the user requests
  live dataloader testing, use the explicitly supplied paper-account settings;
  the current standing port preference is `4002`.
- Start with focused checks:
  `.venv/bin/python -m pytest tests/test_dataloader*.py tests/test_dataloder_helpers.py --tb=short`.
- Run typing after package changes:
  `.venv/bin/python -m mypy haymaker/dataloader tests/test_dataloader*.py tests/test_dataloder_helpers.py`.
- For lifecycle, persistence, or shared-boundary changes, also run the full
  suite with `.venv/bin/python -m pytest --tb=short`.
- Run the package-wide Black command from the root `AGENTS.md`; use an
  outside-sandbox execution if sandbox restrictions prevent completion.
