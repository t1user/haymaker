# Dataloader Refactor Issues

This file tracks dataloader refactor issues across sessions. Keep issue IDs
stable and mark items off as they are addressed.

## Issues

- [ ] `DL-001`: Restart cleanup can leave old workers/tasks alive.
- [ ] `DL-002`: `run_mode = reconnect|wait` mixes workload policy with
  connection ownership.
- [ ] `DL-003`: Request retry/readiness is inconsistent across producer,
  contract selector, headstamp, and workers.
- [ ] `DL-004`: Workload exceptions are swallowed and may look like clean
  completion.
- [ ] `DL-005`: Global mutable runtime state blocks safe restart, attached, and
  concurrent use.
- [ ] `DL-006`: `Manager.store` is ignored.
- [ ] `DL-007`: Store/date helpers have brittle one-row and stale-cache
  assumptions.
- [ ] `DL-008`: `BID_ASK` pacing adjustment appears backwards.
- [ ] `DL-009`: Docs are stale around dataloader connection implementation.
- [ ] `DL-010`: Workload has two phases, but restart currently re-enters
  discovery instead of resuming known remaining work.
- [ ] `DL-011`: Reconcile dataloader pacing limits with current Interactive
  Brokers documentation and verify compliance.
- [ ] `DL-012`: Remove object construction from YAML config.
- [ ] `DL-013`: Integrate dataloader collection naming with
  `haymaker/datastore/collection_namer.py`.
- [ ] `DL-014`: Review dataloader/datastore coupling and define the narrow store
  interface dataloader should depend on.
- [ ] `DL-015`: Define attached-mode pacing policy, likely a configurable
  fraction of normal historical-data allowance.
- [ ] `DL-016`: Make managed restart resume in-memory discovered work instead of
  rerunning discovery after every supervisor restart.
- [ ] `DL-017`: Define process-stop semantics explicitly: no separate checkpoint
  state for now; restart from persisted datastore boundaries on next process
  run.
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
  run through the dataloader request layer under `PACER`, so schedule
  requests are counted with the rest of broker usage. Preserve a
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

## Refactor Plan

1. **Connect Managed Dataloader To Supervisor Correctly**
   - Managed mode owns its own `IB` socket through `ConnectionSupervisor`.
   - Supervisor restarts socket/workload, but dataloader does not rediscover
     everything on every supervisor restart.
   - Discovery runs once per process/session unless explicitly refreshed.
   - Restart resumes in-memory unfinished work.
   - Full program stop does not save a special checkpoint; future process runs
     rediscover from datastore boundaries.
   - Cleanup cancels producer/workers reliably and leaves no stale tasks behind.
   - Covers `DL-001`, `DL-002`, `DL-010`, `DL-016`, and `DL-017`.

2. **Create First-Class Dataloader Session**
   - Own manager, queue, workers, active writers, pacer, pacing registry, and
     cancellation state in one runtime object.
   - Remove or shrink `OBJECTS`, global `PACER`, global `PCR`, and class-level
     `ContractSelector.ib`.
   - Make restart/resume testable.
   - Covers `DL-005`; supports `DL-001` and `DL-010`.

3. **Define Attached Mode As Opportunistic**
   - Borrow an already connected externally managed `IB`.
   - Never call `connectAsync()`, `disconnect()`, or `request_restart()`.
   - Do not attempt supervisor-style readiness detection.
   - Abort or fail cleanly on connection loss or unrecoverable request failure.
   - Intended for short/manual downloads, not long backfills during live trading.
   - Covers the attached side of `DL-002`.

4. **Unify Request Retry And Failure Semantics**
   - Define retryable broker/request errors.
   - Define connection-loss behavior in managed and attached modes.
   - Define when to pause, retry, abort, or propagate failure.
   - Remove silent broad exception loops.
   - Covers `DL-003` and `DL-004`.

5. **Review And Fix Pacing Compliance**
   - Reconcile max concurrent historical requests, request-rate limits,
     identical request cooldown, and `BID_ASK` counting with current IB docs.
   - Define live-trading impact and attached-mode pacing fraction.
   - Covers `DL-008`, `DL-011`, and `DL-015`.

6. **Clean Store And Collection Naming Boundary**
   - Remove `!!python/object/apply` from dataloader config.
   - Integrate naming with `haymaker/datastore/collection_namer.py`.
   - Fix `Manager.store` being ignored.
   - Decide the narrow store interface dataloader should depend on.
   - Fix brittle store boundary assumptions.
   - Covers `DL-006`, `DL-007`, `DL-012`, `DL-013`, and `DL-014`.

7. **Review Period Conversion Helpers**
   - Review `haymaker.dataloader.helpers` duration and bar-size converters.
   - Confirm conversions match intended IB request behavior and project
     assumptions around sessions, days, weeks, months, and max bars.
   - Covers `DL-018`.

8. **Make Gap-Fill Scheduling Session-Aware**
   - Fetch IB historical schedules in the async dataloader request path, not in
     `TaskFactory`/`GapFillFactory`.
   - Run `reqHistoricalScheduleAsync` under `PACER` so schedule lookups count
     against broker usage and can share retry/failure policy with other
     requests.
   - Filter detected datastore gaps to ranges that overlap scheduled sessions
     where data should exist.
   - Keep the core gap/schedule comparison as a pure helper that accepts stored
     data and a supplied schedule, so it can be regression-tested without broker
     access.
   - Review usability of `haymaker/research/utils.py gap_tracer()`,
     consider combining with new dataloader implementation
   - Covers `DL-019`; depends on the request/session refactor shape.

9. **Update Tests And Docs Around The New Model**
   - Add focused regression coverage as each item is handled.
   - Update dataloader docs, codebase map if architecture changes, and stale
     async audit notes.
   - Covers `DL-009` plus test gaps across the plan.
