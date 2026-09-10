# Dataloader Package Guidance

Read the [root guidance](../../AGENTS.md) for workflow and shared contracts.
This package is the standalone Interactive Brokers historical-data downloader;
it never places orders. Changes can consume IB pacing allowance, rewrite large
Arctic series, or alter historical boundaries.

## Architecture

- `DataloaderRuntime` owns its `IB` client and `DataloaderSession` and always
  runs through the shared `App` and `ConnectionSupervisor`. The runtime
  decomposes the `download` mapping across `Manager` and `DataloaderSession`;
  contract selectors retain the target-owned `FuturesSelectionPolicy`. Keep `download`
  as a user-facing run group rather than splitting out worker count solely to
  mirror these internal constructors. The library name is derived from
  `what_to_show` and `bar_size`.
- The dataloader defaults to client ID `1`, distinct from the live runtime's
  expected client ID `0`. A duplicate client ID is a configuration failure; do
  not retry automatically with another ID.
- `DataloaderSession` owns producer/worker execution, failure collection,
  cancellation, and final buffered-data flushing.
- `DataloaderRuntime` owns process Mongo lifecycle and Arctic construction.
  `Manager` owns source expansion, contract discovery, request policy, the
  run-scoped `now`, datastore library derivation and lazy invocation of its
  injected datastore factory, active jobs, and broker schedule requests needed
  for planning. Focused callers may inject a ready datastore instead. Each
  session start obtains a fresh discovery generator. A contract becomes planned
  only after it is conclusively skipped or its constructed job is registered in
  `active_jobs`, so cancellation during an awaited planning step leaves that
  contract eligible for discovery after a reconnect.
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
- `DataloaderSession` reads request policy from generated jobs and `Manager`.

See [the dataloader guide](../../docs/source/dataloader.rst) for usage and
[supervisor guidance](../supervisor/AGENTS.md) for lifecycle changes.

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
- Keep dataloader-specific request and resume behavior in this package.
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
- `HistorySink` uses awaited dataframe and metadata mutations, with the root
  datastore cancellation contract. Apply one downloaded response as a
  single cancellation-safe transition so successful persistence is followed by
  buffer clearing, completion metadata, and range progression before restart.
- The dataloader consumes only the awaited `AsyncDataStore` contract and does
  not submit datastore mutations through a background queue.
- Standalone Ctrl-C cancellation must finish the session flush before shared
  application shutdown closes other background queues. The standalone CLI owns
  its event loop; nested-loop patching belongs to notebook callers.
- A supervisor restart preserves active jobs in memory and queues those jobs
  before starting a fresh discovery pass. The pass skips contracts already
  planned in this process and retries the contract whose planning was
  interrupted. A new process derives remaining work from persisted datastore
  boundaries; there is no separate checkpoint file.
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

- For user-requested IB integration checks, use the supplied paper-account
  connection settings; a port number alone does not establish account mode.
- Start with focused checks:
  `.venv/bin/python -m pytest tests/test_dataloader*.py tests/test_dataloder_helpers.py --tb=short`.
- Run typing after package changes:
  `.venv/bin/python -m mypy haymaker/dataloader tests/test_dataloader*.py tests/test_dataloder_helpers.py`.
