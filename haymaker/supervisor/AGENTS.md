# Supervisor Package Guidance

This package owns recovery around Interactive Brokers socket loss, broker
connectivity interruptions, stale market-data subscriptions, and supervised
workload restarts. Treat it as live-trading lifecycle code: changes should be
narrow, evidence-driven, and tested against focused supervisor/controller
cases.

## Core Invariants

- `ConnectionSupervisor` owns IB socket recovery for live trading and managed
  dataloader runs. It does not manage or restart the gateway/TWS process.
- Route new live-trading restart triggers through `request_restart()`.
- Workload cleanup must set controller hold before stopping runtime work, even
  if the workload task has already completed after socket loss.
- A real socket outage should move the workload into a held/restarting path,
  then require successful connection recovery, probe, and controller sync before
  normal operation resumes.
- Do not treat broker issue/recovery messages alone as proof that local state is
  safe. Judge recovery by socket readiness, probe success, stream health, and
  controller sync.
- Futures roll scheduling is app-lifetime behavior. Do not recreate roll timers
  on each connection or workload cycle.

## Supervisor Implementations

- The state-machine implementation is the primary structure for current
  supervisor work.
- Keep `supervisor_one.py` usable until the user explicitly decides to delete
  it, but do not copy new behavior there unless asked or required for parity.
- Avoid broad redesign while both implementations exist. Prefer local fixes and
  clear notes about behavior that should be ported or deliberately left behind.

## State-Machine Rules

- State objects should make their interrupt behavior visible with
  `accepts_stop`, `accepts_restart`, and `observes_workload`.
- Prefer state-local ownership for temporary behavior. Avoid adding supervisor
  context flags that future states must remember to maintain.
- If a state needs a timer or event subscription, create it inside that state and
  cancel/disconnect it when leaving the state.
- `wait_for_wakeup_or(...)` is the preferred pattern for state-local waits that
  must also react to broker events.
- `ConnectionLostState` is for broker/socket connectivity loss, not ordinary
  data-farm noise.
- Backoff belongs in an explicit recovery path, not as hidden global supervisor
  state. It should reduce churn after repeated failures without delaying the
  first recovery attempts too much.

## Broker Message Classification

- Shared broker-code constants live in `haymaker.supervisor.codes`. Do not
  redefine supervisor-owned codes in controller code.
- Controller logging should ignore supervisor-owned broker messages through its
  configured/effective ignore list. Supervisor should remain the component that
  interprets those messages.
- `1100` and `2110` are hard broker connectivity-loss signals.
- `1101` means data was lost and should be treated as requiring recovery.
- `1102` means data was maintained. It may be a recovery hint while already
  waiting, but should not force a broad restart by itself.
- `1300` is an API socket reset and should be treated as a hard restart signal.
- `2103`, `2105`, `2157`, `2104`, `2106`, `2158`, and `10182` are data-farm or
  live-update signals. They are useful context, but usually not sufficient by
  themselves to prove complete recovery or local correctness.
- `2106` only says an HMDS farm is OK again. Do not use it as proof that all
  stale subscriptions are healthy.

## Stale Subscription Handling

- Nightly `10182` clusters often indicate that existing
  `reqHistoricalData(..., keepUpToDate=True)` subscriptions have stopped.
- `stale_subscription_restart_delay` is the single shortcut for this case. A
  value greater than zero starts/resets a quiet-period timer after `10182`; if
  the timer expires while the workload is otherwise running, request a normal
  workload restart to rebuild subscriptions before streamer timeouts fire.
- A value of `0` disables the shortcut and leaves streamer timeout/backfill as
  the fallback recovery path.
- Keep logging around this path concise. Log enough to reconstruct the sequence,
  but avoid per-message noise during broker-code clusters.

## Streamer Timeouts

- `timeoutEvent` and lightweight historical probes remain useful live health
  checks.
- A stale streamer timeout may indicate a broken subscription even during
  `OPEN NOT LIQUID` periods. European hours for US futures can be marked
  non-liquid while still being important.
- Do not add complex session classification just to avoid occasional quiet-hour
  restarts. Prefer cheaper recovery, such as future per-streamer restart, over
  smarter-but-fragile timeout policy.
- Fully `CLOSED` sessions can avoid action where the existing stream/session
  logic supports that safely.

## Controller Interaction

- Supervisor recovery should not cause controller trading disablement merely
  because IB is unreachable during a restart.
- If restart/stop begins while controller sync is running, sync should abort
  cleanly and not treat cancellation as an unsafe controller error.
- Controller must only resume normal operation after recovery evidence is
  current, not from stale broker messages replayed around reconnect.
- Duplicate trade/commission events may still be emitted by IB after reconnect.
  Controller-side duplicate filtering is expected, but supervisor cleanup should
  keep hold active during restart windows so replayed events do not trigger
  live corrections prematurely.

## Efficiency Improvements To Revisit

- Per-streamer restart remains the preferred long-term way to reduce churn from
  one stale subscription. Restarting only the affected streamer is acceptable
  even if occasional quiet-period restarts remain unnecessary.
- Add bounded reconnect/rebuild backoff if logs show repeated failed recovery
  loops during real broker outages.
- Keep broad workload restart for cleanup/rebuild paths where local component
  ownership or subscription state is uncertain.

## Log Review

- Read `docs/log-review-guidance.md` before supervisor log reviews.
- Separate real broker outages from overreaction to noisy broker messages.
- For each restart, identify the direct trigger, the state at the time, whether
  controller hold was active, whether probe/sync succeeded, and whether backfill
  later showed missed live data.
- Daily IB reset behavior around the broker maintenance window can produce
  clustered farm/live-update messages. Treat repeated logs around that window as
  a pattern to classify, not as independent incidents.

## Validation Defaults

- Focused supervisor checks:
  `python -m pytest tests/test_supervisor.py tests/test_supervisor_one.py`.
- Run controller tests as well when changing controller/supervisor interaction:
  `python -m pytest tests/test_controller.py tests/test_supervisor.py`.
- Run mypy on touched modules when practical, especially after changing shared
  contracts or broker-code constants.
