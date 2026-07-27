## Issue: Sync correction can preserve a stale absolute target

### Summary

Controller synchronization can correct a logical `PositionState.quantity` to
the broker quantity without changing the state's previously accepted absolute
target. If the broker is flat but the persisted target is still non-zero,
`BracketExecutionModel` recovery can interpret the corrected state as work
still required and reopen the position.

This is deferred for consideration together with
[`issue_1.md`](./issue_1.md), because both issues concern fills and position
state changing across non-atomic synchronization and recovery boundaries.

### Example

1. Book contains `quantity=1` and `target_quantity=1`.
2. The broker reports the Contract flat.
3. Sync corrects Book to `quantity=0` but leaves `target_quantity=1`.
4. The supervised graph starts and the execution model recovers the target.
5. Convergence submits a new OPEN order for one unit.

### Relevant boundary

`SyncCoordinator.handle_error_positions()` updates the logical quantity and
sometimes the position episode, while `BracketExecutionModel.recover()`
subsequently converges the persisted target. Neither subsystem currently
records whether the broker correction should supersede the old strategy
setpoint.

### Questions to resolve

- Does an authoritative broker correction cancel the previous target, or should
  the strategy still be expected to reach it?
- Should reconciliation clear the target, align it to the corrected quantity,
  or persist a separate “awaiting strategy decision” state?
- How should a fill arriving during sync affect that decision?
- Should startup recovery wait for a fresh strategy/Portfolio target after a
  corrective sync?

### Acceptance criteria

- Correcting a broker-flat position cannot unintentionally reopen it.
- A deliberate outstanding target is not silently discarded.
- Fill, sync, persistence, and supervised-start ordering is deterministic.
- Regression tests cover broker-flat correction, fills during sync, and process
  recovery before source re-emission.
