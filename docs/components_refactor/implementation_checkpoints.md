# Concrete contracts and configurable rolling: implementation checkpoints

Approved scope (2026-09-07): preserve one-to-one episode execution, remove
direct `target_key`, expose blueprint/chain access, and separate futures-roll
policy from durable execution. No production migration or compatibility layer.

Each checkpoint includes focused tests, documentation, pytest and mypy before
its commit. Existing integration tests and processor matrices are retained.

1. Independent broker/event end-to-end test foundation.
2. Separate held and pending-target Contracts and bracket inputs; recover
   CLOSE/REVERSE from Book without depending on transient callbacks.
3. Public blueprint and qualified-member lookup; strict explicit nth selection
   with intentional ACTIVE/NEXT fallback; SignalModel Contract-selection hook.
4. Concrete direct targets/accounting/router ownership; remove `target_key`
   throughout the schema and public API.
5. Portfolio query helpers, completion feedback and optional state mixin.
6. User-defined roll triggers/destinations, default past-to-ACTIVE policy,
   repeat-safe scheduled checks and public custom-check entry point.
7. Integrate both roll modes, preserve brackets/episodes, transfer settled
   direct targets to the destination exactly once, honor newer targets.
8. Conversion tooling, comprehensive recovery tests and final public/agent docs.
9. Full validation and independent review of components/Portfolio. Report
   remaining findings instead of silently expanding scope.

Confirmed choices: direct rolls transfer the settled old target additively to
the destination and notify Portfolio; explicit out-of-range `nth_contract`
raises rather than silently clamping. PortfolioWrapper does not resolve CLOSE
Contracts. One-to-one OPEN uses the incoming Contract, CLOSE the Book episode
Contract, and REVERSE closes the old episode fully before opening the incoming
Contract. Direct targets always refer to their exact concrete Contract.

## Progress

- Planning and baseline inspection: complete; worktree initially clean.
- Checkpoint 1: independent IB-boundary harness and three pipeline tests added.
  Full pytest: 1,596 passed. Mypy: 110 source files clean. Black and diff checks
  passed. Blotter-disabled commission persistence is exercised explicitly.
- Checkpoint 2: held/pending Contract and bracket-input separation, source-based
  CLOSE verification, durable reversal recovery, and first-fill-only block
  clearing. Independent suite covers 17 scenarios, including restarts before,
  during and after close fills. Full pytest: 1,610 passed; mypy: 110 files clean.
- Checkpoint 3: public blueprint snapshots, full qualified-member lookup,
  atomic overlap rejection, strict explicit nth selection, and the narrow
  SignalModel selection hook. Aggregators use public blueprint access. Full
  pytest: 1,615 passed; additional aggregator checks: 129 passed; mypy clean.
- Checkpoint 4: removed user-defined direct execution keys across messages,
  orders, targets, Router and roll participants. Direct execution is concrete
  Contract convergence; old implicit held-expiry tests now assert explicit
  Portfolio allocation. Added two independent direct pipeline scenarios.
  Existing direct roll plumbing now saves additive target-transfer snapshots
  to keep the schema cutover functional. Full pytest: 1,615 passed; mypy clean.
- Checkpoint 5: Portfolio blueprint position queries, optional explicit state
  mixin, and post-accounting completion feedback through Router. Independent
  tests demonstrate user-owned flat-before-open sequencing and repeat-safe
  recovery notifications. Full pytest: 1,620 passed; mypy: 110 files clean.
- Checkpoint 6: public roll policies select triggers and same-series endpoints;
  default policy retains every eligible later expiry. Check-time selector
  snapshots and durable schedule markers prevent fixed-schedule cascades across
  restart. Public daily/custom check behavior documented. Full pytest: 1,626
  passed; mypy: 111 files clean; Black and diff checks passed.
- Checkpoint 7: independent IB-boundary roll tests now cover physical combo
  legs, additive transfers, newer targets, missed callbacks, critical bracket
  replacement, preserved episode identity and a reversal requested mid-roll.
  Recovery between the two transfer writes is explicitly tested. Target
  verification now waits for pending roll state, not standing protection.
  Full pytest: 1,631 passed; mypy: 111 files clean; Black/diff checks passed.
- Checkpoint 8: converter supports explicit component evidence and refuses
  ambiguous keyed allocations, missing episode evidence, mixed source schemas
  and pending rolls. Versioned provenance makes reruns idempotent. Added
  Fill-level commission/P&L reports independent of optional blotter. Reset
  cutoffs now also cover residual evidence without a Portfolio target. Final
  semantics documented; no database conversion executed. Full pytest: 1,637
  passed; mypy: 112 files clean; Black/diff checks passed. Strict documentation
  and additional diagnostic checks continue in checkpoint 9.
- Checkpoint 9: validation and the separate components/Portfolio review are
  complete; see `final_review.md`. Full pytest: 1,637 passed; mypy: 112 files
  clean; all 38 changed Python files pass Black. Remaining findings include a
  reproduced stale-quantity bracket-roll wait, repeated fill-history query
  work, two Portfolio configuration surprises, five existing Pyright override
  diagnostics, and 81 strict Sphinx reference warnings. These are reported,
  not silently fixed or represented as passing acceptance checks.

## Approved review follow-up (2026-09-08)

- `9a175ea`: refresh waiting bracket episodes and net allocation, skip flat
  sources, reject replaced episodes, defer resumption until accounting settles.
- `b329d7f`: replay direct evidence once per compound query, reject invalid
  Portfolio configuration, and align override parameter names.
- Documentation references corrected without global suppression; strict Sphinx
  passes. Final pytest: 1,651 passed; mypy: 113 files clean; Black/diff clean.
  Pyright: zero errors, two known aggregated-export warnings. All actionable
  findings in final_review.md are resolved. No database migration or push.
