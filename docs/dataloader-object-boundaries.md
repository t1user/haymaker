# Dataloader Object Boundaries

This note records the target object boundaries for the dataloader store,
scheduling, download, and persistence path. It is intentionally narrower than a
full architecture proposal; gap-fill scheduling and custom datastore injection
remain separate work.

## Current Target

### AsyncStoreView

- Owns an async datastore, one contract, and the run's `now` value.
- Requires the caller to provide the run's bar size explicitly; it does not
  carry a compatibility default.
- Loads existing stored data once before scheduling.
- Exposes `backfill_boundary`, `to_date`, and `expiry_or_now()` for scheduling.
- Normalizes read-side scheduling boundaries according to the configured bar
  size: intraday values are UTC-aware `datetime` objects, while `1 day`,
  `1 week`, and `1 month` values are `date` objects.
- Does not write downloaded data.

### Time Policy

- Keeps `ib.reqHistoricalData` and `ib.reqHeadTimeStamp` on `formatDate=2`.
- Rejects naive intraday datetimes before scheduling.
- Uses date-only scheduling points for daily, weekly, and monthly bars.
- Does not expose alternative IB date formats in the current datastore path.

### TaskPlanner

- Is the pure scheduling boundary.
- Consumes an `AsyncStoreView`, a head timestamp, optional lookback policy, gap-fill
  mode, optional cached timezone, optional historical sessions, and run-local
  learned gap patterns.
- Owns the run lookback clamp and produces backfill, update, and gap-fill
  ranges.
- Returns ranges in execution order: update, backfill, then gap-fill.
- Skips gap-fill ranges for continuous futures because IB historical requests
  for `CONTFUT` cannot target an explicit `endDateTime`.
- Keeps broker schedule calls out of pure scheduling code; `Manager` owns those
  async request-layer inputs.

### DownloadJob

- Owns one contract's planned ranges and request progression.
- Produces request parameters for workers.
- Uses FIFO range execution. Continuous-future jobs pass an empty
  `endDateTime`; planning produces at most one latest-ended range for them, and
  the job treats that range as one terminal IB request.
- Prunes pending same-job gap ranges once the run-local learner marks their
  pattern as repeatedly empty.
- Carries the bar size used for request duration calculation and worker
  `barSizeSetting`.
- Buffers downloaded bars only until they are handed to persistence.
- Does not read existing store state.

### HistorySink

- Owns persistence for downloaded bars.
- For now, preserves current behavior: read existing data, concatenate the new
  chunk, and call `async_write()`.
- Does not normalize or clean the downloaded dataframe; it persists the
  IB-side dataframe shape it receives.
- Later can switch to smarter append, prepend, or gap-fill persistence without
  changing downloader request logic.

## Store Policy

The dataloader currently depends on `AsyncAbstractBaseStore` and the Arctic
implementation behind it. Arctic remains responsible for final cleaning,
metadata, duplicate removal, and monotonic sorting. The dataloader should not
accept an arbitrary persistence object unless it also defines where those data
cleanliness guarantees live.

There is no dataloader datastore backend config while Arctic is the only
supported backend. Arctic library naming is derived from `wts` and `barSize`.
Collection naming uses the datastore default `simple_collection_namer`.

## Runtime Scope

The current split is:

- `Manager` owns source expansion, contract discovery, headstamp lookup, store
  and sink construction, request policy (`bar_size`, `wts`, `max_lookback_days`), the
  run-scoped `now` value, and active job tracking for restart/resume.
- `TaskPlanner` owns pure scheduling, lookback clamping, and conversion from
  stored gaps or supplied sessions into actionable gap-fill ranges.
- `DownloadJob` owns request progression for one contract.
- `DownloadContainer` owns per-range buffering and missing-range progress.
- `AsyncStoreView` owns read-only datastore boundaries for scheduling.
- Time policy owns date/datetime normalization before scheduling comparisons.
- `HistorySink` owns persistence.
- `DataloaderSession` owns producer/worker orchestration and IB request
  execution. It does not own an independent `bar_size` or `whatToShow` policy.

Deferred scheduling work should keep IB historical schedule requests outside
`TaskPlanner`; those requests belong in the async request layer under
session-scoped pacing before their result is supplied to pure scheduling code.
