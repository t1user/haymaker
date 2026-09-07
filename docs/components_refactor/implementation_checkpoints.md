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
