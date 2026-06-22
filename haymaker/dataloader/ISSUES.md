# Dataloader Refactor Issues

This file tracks dataloader refactor issues across sessions. Keep issue IDs
stable and mark items off as they are addressed.

## Issues

- [x] `DL-001`: Restart cleanup can leave old workers/tasks alive.
- [x] `DL-002`: `run_mode = reconnect|wait` mixes workload policy with
  connection ownership.
- [x] `DL-003`: Request retry/readiness is inconsistent across producer,
  contract selector, headstamp, and workers.
- [x] `DL-004`: Workload exceptions are swallowed and may look like clean
  completion.
- [x] `DL-005`: Global mutable runtime state blocks safe restart and
  concurrent use.
- [x] `DL-006`: `Manager.store` is ignored.
- [x] `DL-007`: Store/date helpers have brittle one-row and stale-cache
  assumptions.
- [x] `DL-008`: `BID_ASK` pacing adjustment appears backwards.
- [x] `DL-009`: Docs are stale around dataloader connection implementation.
  Public dataloader, supervisor, configuration, and codebase-map docs now
  describe the supervised-only connection model: dataloader owns a separate
  `IB` socket under `ConnectionSupervisor`, uses a distinct default client ID,
  and leaves TWS/Gateway process management outside the supervisor.
- [x] `DL-010`: Workload has two phases, but restart currently re-enters
  discovery instead of resuming known remaining work.
- [x] `DL-011`: Reconcile dataloader pacing limits with current Interactive
  Brokers documentation and verify compliance.
- [x] `DL-012`: Remove object construction from YAML config.
- [x] `DL-013`: Integrate dataloader collection naming with
  `haymaker/datastore/collection_namer.py`.
- [x] `DL-014`: Review dataloader/datastore coupling and define the narrow store
  interface dataloader should depend on.
- [x] `DL-015`: Define live-trading impact of dataloader pacing and any
  reduced allowance for future optional non-supervised modes.
- [x] `DL-016`: Make managed restart resume in-memory discovered work instead of
  rerunning discovery after every supervisor restart.
- [x] `DL-017`: Define process-stop semantics explicitly: no separate checkpoint
  state for now; restart from persisted datastore boundaries on next process
  run. The user-facing dataloader docs now state that supervisor recovery
  preserves in-memory active jobs within the same process, while a new process
  derives remaining work from the persisted Arctic-backed datastore.
- [ ] `DL-018`: Review `haymaker.dataloader.helpers` period converters and make
  sure duration/bar-size conversions match intended IB request behavior.
- [ ] `DL-019`: Make gap-fill scheduling session-aware using IB
  historical schedules.  Current `GapFillFactory.gap_factory()` infers
  expected gaps only from stored data regularity, so scheduled market
  closures, holidays, early closes, and weekend breaks have to be
  guessed from timestamps. Later refactor work should combine
  datastore gap detection with
  `ib_insync.IB.reqHistoricalScheduleAsync` output and schedule only
  missing ranges that fall inside IB sessions where bars should
  exist. The IB schedule request should not live inside
  `TaskFactory`/`GapFillFactory`: it is an async broker call and must
  run through the dataloader request layer under session-scoped
  `RequestPacing`, so schedule requests are counted with the rest of broker
  usage. Preserve a
  narrow pure scheduling helper that can be tested from stored data
  plus a supplied historical schedule. The observed
  `ib.reqHistoricalScheduleAsync` return shape is like this: [
  HistoricalSession(startDateTime='20250330- 17:00:00',
  endDateTime='20250331-16:00:00', refDate='20250331'),] is an
  `ib_insync.objects.HistoricalSchedule` with `startDateTime`,
  `endDateTime`, `timeZone`, and `sessions`; each session is an
  `ib_insync.objects.HistoricalSession` with `startDateTime`,
  `endDateTime`, and `refDate`. In the observed output, datetimes are
  IB strings such as `20250330-17:00:00`, timezone is a string such as
  `US/Central`, and sessions include normal trading days, weekend
  gaps, holidays, and early closes.
- [x] `DL-020`: Define run-wide `now` snapshot semantics explicitly. `Manager`
  owns the run-scoped `now` value, normalizes it for the configured bar size,
  and passes it into scheduling and request-age validation. `DataloaderSession`
  does not recompute a separate freshness cutoff during worker execution.
- [x] `DL-021`: Review legacy dataloader runtime objects for overly broad scope
  or execution responsibilities. Pay special attention to whether manager,
  job, store wrapper, selector, and scheduler boundaries still match the
  simplified queue/worker architecture. Start from
  `docs/dataloader-object-boundaries.md`. Manager now owns request policy
  (`bar_size`, `wts`, `max_bars`, run `now`), generated jobs carry the bar size
  used by workers, and `DataloaderSession` no longer has independent
  compatibility `bar_size`/`wts` fields.
- [ ] `DL-022`: Classify recorded dataloader job failures as terminal or
  retryable. Current behavior records failed download jobs and lets workers keep
  draining the queue; later policy should decide which failures deserve retries,
  which should only be summarized, and which indicate process/session failure.
- [x] `DL-023`: Define dataloader historical date policy around
  `formatDate=2`. Intraday scheduling now uses UTC-aware `datetime` values,
  daily/weekly/monthly scheduling uses `date` values, naive intraday datetimes
  are rejected before comparison, and alternative IB date formats remain out of
  scope for the current datastore path.

## Refactor Plan

1. **Connect Dataloader To Supervisor Correctly**
   - Dataloader owns its own `IB` socket through `ConnectionSupervisor`.
   - Supervisor restarts socket/workload, but dataloader does not rediscover
     everything on every supervisor restart.
   - Discovery runs once per process/session unless explicitly refreshed.
   - Restart resumes in-memory unfinished work.
   - Full program stop does not save a special checkpoint; future process runs
     rediscover from datastore boundaries.
   - Cleanup cancels producer/workers reliably and leaves no stale tasks behind.
   - Covers `DL-001`, `DL-002`, `DL-010`, `DL-016`, and `DL-017`.

2. **Create First-Class Dataloader Session**
   - Own manager, queue, workers, active jobs, pacer, pacing registry, and
     cancellation state in one runtime object.
   - Remove or shrink `OBJECTS`, global `PACER`, global `PCR`, and class-level
     `ContractSelector.ib`.
   - Make restart/resume testable.
   - Covers `DL-005`; supports `DL-001` and `DL-010`.

3. **Normalize Supervised-Only Connection Model**
   - Dataloader always owns its own `IB` socket and runs it under
     `ConnectionSupervisor`.
   - Remove legacy `run_mode` configuration and code paths.
   - Use dataloader client ID `1` by default so it is distinct from the live
     runtime's expected `clientId=0`.
   - Do not retry alternate client IDs automatically; duplicate client ID is a
     connection configuration failure.
   - Covers the remaining active part of `DL-002`.

4. **Unify Request Retry And Failure Semantics**
   - Define retryable broker/request errors.
   - Define supervised connection-loss behavior.
   - Define when to pause, retry, abort, or propagate failure.
   - Remove silent broad exception loops.
   - Covers `DL-003` and `DL-004`.

5. **Review And Fix Pacing Compliance**
   - Reconcile max concurrent historical requests, request-rate limits,
     identical request cooldown, and `BID_ASK` counting with current IB docs.
   - Apply configured `pacer_allowance_fraction` when building pacing limits so
     supervised dataloader runs can voluntarily reserve part of the IB allowance
     for live trading or other sessions.
   - Covers `DL-008`, `DL-011`, and `DL-015`.

6. **Clean Store And Collection Naming Boundary**
   - Remove `!!python/object/apply` from dataloader config.
   - Integrate naming with `haymaker/datastore/collection_namer.py`.
   - Fix `Manager.store` being ignored.
   - Decide the narrow store interface dataloader should depend on.
   - Fix brittle store boundary assumptions.
   - Covers `DL-006`, `DL-007`, `DL-012`, `DL-013`, and `DL-014`.

7. **Review Runtime Object Boundaries**
   - Review whether manager, download job, download container,
     scheduling factories, `AsyncStoreView`, and `HistorySink` have the right
     responsibilities.
   - First-class `TaskPlanner` now owns pure range planning and max-period
     clamping.
   - `DownloadJob` now names the request-progression object.
   - `Manager` owns historical request policy and run-scoped `now`; worker
     execution reads request bar size from each generated job.
   - `AsyncStoreView` requires explicit bar-size policy, and `HistorySink`
     preserves raw downloaded data without scheduling normalization.
   - Start from `docs/dataloader-object-boundaries.md`.
   - Covers `DL-021`.

8. **Review Period Conversion Helpers**
   - Review `haymaker.dataloader.helpers` duration and bar-size converters.
   - Confirm conversions match intended IB request behavior and project
     assumptions around sessions, days, weeks, months, and max bars.
   - Monthly bar-size duration support was added while defining the historical
     date policy.
   - Covers `DL-018`; partially addressed by `DL-023`.

9. **Make Gap-Fill Scheduling Session-Aware**
   - Fetch IB historical schedules in the async dataloader request path, not in
     `TaskFactory`/`GapFillFactory`.
   - Run `reqHistoricalScheduleAsync` under session-scoped `RequestPacing` so
     schedule lookups count against broker usage and can share retry/failure
     policy with other requests.
   - Filter detected datastore gaps to ranges that overlap scheduled sessions
     where data should exist.
   - Keep the core gap/schedule comparison as a pure helper that accepts stored
     data and a supplied schedule, so it can be regression-tested without broker
     access.
   - Review usability of `haymaker/research/utils.py gap_tracer()`,
     consider combining with new dataloader implementation
   - Covers `DL-019`; depends on the request/session refactor shape.

10. **Update Tests And Docs Around The New Model**
   - Add focused regression coverage as each item is handled.
   - Update dataloader docs, codebase map if architecture changes, and stale
     async audit notes.
   - Covers `DL-009` plus test gaps across the plan.

11. **Optional Future Connection Modes**
   - After the core dataloader refactor is complete, reconsider whether a
     gateway-managing Watchdog/IBC runner is useful for paper-account,
     multi-day downloads.
   - Keep attached/borrowed-IB mode out of active scope unless a real
     same-interpreter caller appears.
   - Do not implement before pacing and retry semantics are settled.
