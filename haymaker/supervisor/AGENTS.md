# Supervisor: lifecycle and recovery

Read the [root guidance](../../AGENTS.md) for shared runtime ownership,
trading safety and queue shutdown. For incident investigation, use
[log-review guidance](../../docs/log-review-guidance.md); this file covers
implementation and extension contracts.

## Ownership and code layout

- `ConnectionSupervisor` owns one IB socket and one supervised workload.
  Use it only when Haymaker owns that connection.
- `supervisor.py` owns lifecycle request events, broker-event routing,
  workload tracking, socket cleanup and `SupervisorRace`.
- `states.py` owns connection, probing, recovery, restart, backoff and shutdown.
  Temporary timers and event-specific flags belong to their active state.
- `settings.py` owns connection/recovery settings; `codes.py` owns shared
  broker-code groups.
- The workload protocol exposes `start()` and `stop()` for reconnectable
  cycles. The application runtime additionally exposes `ib`,
  `bind_supervisor()` and `close()`. `App` calls `close()` after the
  supervisor finishes, then completes detached-task and queue cleanup.

## Lifecycle and races

Normal startup is `Connecting -> Probing -> StartingWorkload -> Connected`.

- `Connecting` retries `connectAsync()` after `retry_delay` until connected
  or stopped. Restart requests during connection attempts are discarded.
- `Probing` requests 30 seconds of five-second `MIDPOINT` historical data
  with `probe_timeout`; a non-empty response marks the connection usable.
  Success reuses an existing workload or starts one tracked workload task.
- `Connected` arms the IB idle timeout and observes broker events, lifecycle
  requests, stale subscriptions and workload completion.
- Idle timeout returns to `Probing`. Broker loss enters `ConnectionLost`,
  waits for recovery or `auto_recovery_grace_period`, then probes.
- Failed probes follow
  `BackoffRestartCleanup -> BackoffRestarting -> Connecting`.
  Cleanup finishes before the interruptible `connection_lost_retry_delay`.
- Accepted restarts follow `Restarting -> Connecting` without adding a
  failed-probe delay. Normal workload completion stops the supervisor.
- Exceptions escaping `run()` are terminal: final cleanup runs and the
  exception is re-raised.

`SupervisorRace` observes only the active state's declared signals:
`accepts_stop`, `accepts_restart`, and `observes_workload`. Priority is
stop, restart, state completion, then workload completion. Pending lifecycle
requests are checked before state work and again after the race wakes.

- Restart requests coalesce through one event. A request rejected by the state
  is discarded; deferral would create a stale restart after recovery.
- `Restarting` accepts neither stop nor restart during cleanup. A pending stop
  is consumed immediately afterward, before reconnect.
- `BackoffRestarting` accepts stop and ignores restart.
- Restart, backoff, stopping and stopped states do not observe workload
  completion because cleanup already owns that task.
- Define all three flags deliberately for new states and cover unusual
  combinations with race tests.

## Cleanup and controller integration

- Submit restart triggers through `request_restart(reason)`, never by setting
  the internal event. `App` injects this callback and the
  `connection_unavailable` event through `bind_supervisor()`.
- `connection_unavailable` is set at startup, broker loss, restart, disconnect
  and shutdown; a successful probe clears it. Controller sync races against
  this event so recovery or stop can abort an in-progress sync.
- `cleanup_workload()` calls `stop(reason)` only for a still-active workload.
  For a completed task, collect its result without a redundant stop callback.
  After `stop()`, cancel and await any remaining workload task. Unexpected
  workload/stop failures escape after terminal cleanup.
- Live cleanup marks connection unavailable, sets Controller hold through
  `LiveRuntime.stop()`, cancels workload work, then disconnects. Reconnection
  must probe successfully before starting work.
- Startup sync validates broker position freshness and scans orders while
  held, then releases hold before back-reporting completed fills and performing
  position/bracket checks. Preserve that boundary.
- Hold suppresses order-status persistence/logging and commission saves, but
  not `onExecDetailsEvent`; duplicate executions must remain idempotent.
- Guard intentional socket closure with `_intentional_disconnect` so its
  `disconnectedEvent` does not request another restart.
- Attach supervisor event handlers once in `__post_init__`, not per reconnect.

## Broker-event policy

Keep shared code groups in `codes.py`. Supervisor-owned codes belong in the
Controller's effective ignore list to avoid duplicate broker-event logging.
The [IB message guide](../../docs/source/ib_message_codes.rst) documents code
meanings; preserve these control decisions:

- `1100` and `2110` enter broker-connectivity recovery. Repeated loss
  messages must not create overlapping waits or rebuilds.
- `1101` and `1300` request a rebuild.
- `1102` wakes `ConnectionLost` for a probe;
  `restart_on_recovered_connection=True` requests a rebuild instead.
  It does not establish subscription freshness.
- Farm-status messages are context only. Neither they nor generic
  `updateEvent` traffic can establish connection recovery.
  `log_datafarm_status` changes logging only.
- In `Connected`, `10182` starts the state-owned
  `STALE_SUBSCRIPTION_RESTART_DELAY` timer. Later occurrences reset it.
  Quiet-period expiry rebuilds the workload; leaving the state cancels it.
  Broker loss takes priority over this timer.
- Streamer timeout policy and monitor lifetimes belong to the
  [components guide](../components/AGENTS.md). Keep the state-owned stale
  subscription timer independent from those monitors.

## Extending behavior

Classify a new signal as an immediate rebuild, broker-loss wait followed by
probe, state-local health evidence, or informational context before wiring it.

- Keep recovery flow here; Controller exchanges the injected restart callback
  and connection-unavailable event.
- Cancel state-owned tasks and event subscriptions on every state exit.
- Use `wait_for_wakeup_or()` for waits that broker events must interrupt.
- Keep transitions observable without logging the same event in multiple owners.
- Test normal flow, repeated signals, stop/restart priority, cancellation,
  cleanup ordering and stale events after a transition.

## Validation

Use the root validation policy, with these focused suites:

- Lifecycle and exports:
  `.venv/bin/python -m pytest tests/test_supervisor.py tests/test_supervisor_package.py`
- Controller interaction: add `tests/test_controller.py`.
- Dataloader ownership: add `tests/test_dataloader_runtime.py`.
- Package typing:
  `.venv/bin/python -m mypy haymaker/supervisor tests/test_supervisor.py`
