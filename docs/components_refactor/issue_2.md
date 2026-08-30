## Issue: Sync correction can preserve a stale absolute target

Status: resolved on 2026-08-30.

### Summary

Before resolution, Controller synchronization could correct a logical
`PositionState.quantity` to the broker quantity without changing the state's
previously accepted absolute target. If the broker was flat while the target
remained non-zero, `BracketExecutionModel` recovery could interpret it as work
still required and reopen the position.

### Example

1. Book contains `quantity=1` and `target_quantity=1`.
2. The broker reports the Contract flat.
3. Sync corrects Book to `quantity=0` but leaves `target_quantity=1`.
4. The supervised graph starts and the execution model recovers the target.
5. Convergence submits a new OPEN order for one unit.

### Relevant boundary

`SyncCoordinator.handle_error_positions()` updates the logical quantity and
position episode, while `BracketExecutionModel.recover()` subsequently
converges the persisted target. The missing contract between those subsystems
was whether an authoritative broker correction supersedes the old strategy
setpoint.

### Resolution

An authoritative broker correction supersedes the previously accepted
one-to-one target. Sync aligns `target_quantity` to the corrected logical
quantity and advances `target_created_at` to the correction time. If the
corrected quantity is flat, it also closes the episode by clearing
`position_id` and `bracket_inputs`; `blocked_direction` is preserved.

This deliberately discards the old setpoint: after reconciliation establishes
that broker authority differs from recovered strategy state, replaying that
setpoint is unsafe. A subsequent strategy Signal may create a newer target if
the position should be opened again.

The companion synchronization fix uses one fresh broker snapshot per pass and
defers Contracts with active OPEN/CLOSE work, so correction is not performed
against a known in-flight one-to-one transaction. Regression coverage verifies
broker-flat correction, the transient fill race, and
`BracketExecutionModel.recover()` before any source re-emission; recovery
submits no replacement OPEN order.
