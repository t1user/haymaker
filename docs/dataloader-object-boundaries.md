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

- Should become the pure scheduling boundary.
- Consumes an `AsyncStoreView`, a head timestamp, max-period policy, and gap
  policy.
- Produces download ranges.
- Later can consume IB historical schedules without putting broker calls inside
  pure scheduling code.

The current `haymaker.dataloader.scheduling` factories fill this role only
partially.

### DataWriter Or DownloadJob

- Owns one contract's planned ranges and request progression.
- Produces request parameters for workers.
- Buffers downloaded bars only until they are handed to persistence.
- Does not read existing store state.

The current `DataWriter` still owns request progression and buffering.

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

## Deferred Review

The next review should inspect whether the legacy objects still have the right
scope:

- `Manager`
- `DataWriter`
- `DownloadContainer`
- `haymaker.dataloader.scheduling` factories
- `AsyncStoreView`
- `HistorySink`

The review should decide whether to introduce a first-class `TaskPlanner`,
rename `DataWriter` to `DownloadJob`, and move persistence details fully out of
the writer path.
