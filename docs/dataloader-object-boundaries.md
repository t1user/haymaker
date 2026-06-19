# Dataloader Object Boundaries

This note records the target object boundaries for the dataloader store,
scheduling, download, and persistence path. It is intentionally narrower than a
full architecture proposal; gap-fill scheduling and custom datastore injection
remain separate work.

## Current Target

### AsyncStoreView

- Owns an async datastore, one contract, and the run's `now` value.
- Loads existing stored data once before scheduling.
- Exposes `from_date`, `to_date`, and `expiry_or_now()` for scheduling.
- Does not write downloaded data.

### TaskPlanner

- Is the pure scheduling boundary.
- Consumes an `AsyncStoreView`, a head timestamp, max-period policy, and gap
  policy.
- Owns the run lookback clamp and produces download ranges.
- Later can consume IB historical schedules without putting broker calls inside
  pure scheduling code.

The older `task_factory()` and `task_factory_with_gaps()` functions remain as
compatibility wrappers around the same lower-level range logic.

### DownloadJob

- Owns one contract's planned ranges and request progression.
- Produces request parameters for workers.
- Buffers downloaded bars only until they are handed to persistence.
- Does not read existing store state.

`DataWriter` remains a compatibility alias for older imports.

### HistorySink

- Owns persistence for downloaded bars.
- For now, preserves current behavior: read existing data, concatenate the new
  chunk, and call `async_write()`.
- Later can switch to smarter append, prepend, or gap-fill persistence without
  changing downloader request logic.

## Store Policy

The dataloader currently depends on `AsyncAbstractBaseStore` and the Arctic
implementation behind it. Arctic remains responsible for final cleaning,
metadata, duplicate removal, and monotonic sorting. The dataloader should not
accept an arbitrary writer-like object unless it also defines where those data
cleanliness guarantees live.

There is no dataloader datastore backend config while Arctic is the only
supported backend. Arctic library naming is derived from `wts` and `barSize`.
Collection naming uses the datastore default `simple_collection_namer`.

## Runtime Scope

The current split is:

- `Manager` owns source expansion, contract discovery, headstamp lookup, store
  and sink construction, and active job tracking for restart/resume.
- `TaskPlanner` owns pure scheduling and max-period clamping.
- `DownloadJob` owns request progression for one contract.
- `DownloadContainer` owns per-range buffering and missing-range progress.
- `AsyncStoreView` owns read-only datastore boundaries for scheduling.
- `HistorySink` owns persistence.

Deferred scheduling work should keep IB historical schedule requests outside
`TaskPlanner`; those requests belong in the async request layer under
session-scoped pacing before their result is supplied to pure scheduling code.
