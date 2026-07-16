# Supervisor Package Guidance

This package owns recovery for Haymaker-managed Interactive Brokers API
connections. Changes here affect live trading and managed dataloader runs, so
base decisions on current code, focused tests, and complete incident timelines.

## Scope And Ownership

- `ConnectionSupervisor` owns one `ib_insync.IB` socket and one supervised
  runtime. It connects, probes, starts work, handles recovery, and performs
  connection/workload cleanup.
- It does not start, stop, or restart TWS or IB Gateway.
- Use it only when Haymaker owns the socket. Attached dataloader work borrows an
  external connection and must not connect, disconnect, or restart it.
- `App` is the single process runner for live and dataloader runtimes. It owns
  final runtime, detached-task, and queue cleanup. The command-line entrypoint
  owns threaded logging setup and final flushing so strategy-import and runtime-
  construction failures are also delivered. In particular,
  futures-roll scheduling is created once for the process and must not be
  recreated during workload restarts.
- The workload owns component cleanup. For live trading, `LiveRuntime.stop()`
  sets the controller hold before the supervisor cancels the workload task.

## Code Layout

- `supervisor.py`: supervisor object, lifecycle request events, broker-event
  routing, workload task ownership, socket cleanup, `SupervisorRace`, and the
  narrow supervised-workload protocol.
- `states.py`: connection, probe, connected, recovery, restart, backoff, and
  shutdown behavior. Temporary timers and event-specific flags belong here.
- `settings.py`: connection and recovery settings.
- `codes.py`: broker-code groups shared with controller logging policy.
- `tests/test_supervisor.py`: focused lifecycle, race, cleanup, and broker-code
  regression coverage.
- `tests/test_supervisor_package.py`: public package export coverage.

## Lifecycle

The normal startup path is:

`Connecting -> Probing -> StartingWorkload -> Connected`

- `Connecting` retries `connectAsync()` after `retry_delay` until connected
  or stopped. Redundant restart requests are ignored while connection attempts
  are already in progress.
- `Probing` requests 30 seconds of five-second `MIDPOINT` historical data and
  waits up to `probe_timeout`. A non-empty response marks the connection
  available.
- `StartingWorkload` creates one tracked workload task.
- `Connected` arms the IB idle timeout and waits for timeout, broker, restart,
  stop, stale-subscription, or workload-completion signals.

Recovery paths are:

- Idle `timeoutEvent`: `Connected -> Probing`.
- Broker connectivity loss (`1100` or `2110`):
  `Connected -> ConnectionLost -> Probing`.
- Successful probe with an existing workload: return to `Connected` without
  rebuilding it.
- Failed probe:
  `Probing -> BackoffRestartCleanup -> BackoffRestarting -> Connecting`.
- Accepted restart request:
  `Connected -> Restarting -> Connecting`.
- Normal workload completion: transition to `Stopping`; do not silently start
  it again.
- Stop request: finish required cleanup and end in `Stopped`.

Unexpected exceptions escaping the supervisor are terminal. `run()` logs the
exception, performs final cleanup, and re-raises it. Do not document or assume an
automatic retry budget around arbitrary programming errors.

## Race And Priority Rules

`SupervisorRace` waits on state work and only the external signals declared by
the active state:

- `accepts_stop`: stop may interrupt this state.
- `accepts_restart`: restart may interrupt this state.
- `observes_workload`: workload completion participates in the race.

The priority is stop, then restart, then state completion, then workload
completion. Pending lifecycle requests are checked before state work starts and
again after the race wakes.

- Restart requests coalesce through one `asyncio.Event`.
- A restart request rejected by the active state is discarded, not deferred.
  Deferring it would cause stale, unnecessary restarts after recovery.
- `Restarting` does not accept stop or restart while cleanup is running. A stop
  requested during cleanup is consumed immediately afterward, before reconnect.
- `BackoffRestarting` accepts stop but ignores restart, so a long outage cannot
  create overlapping rebuild loops.
- Restart, backoff, stopping, and stopped states do not observe workload
  completion because cleanup already owns that task.

When adding or changing a state, define these three flags deliberately and add a
race regression test for any non-default combination.

## Workload And Cleanup Contract

- The supervisor workload contract contains `start()` and `stop()` for
  reconnectable cycles. The application runtime contract additionally contains
  `ib`, `bind_supervisor()`, and `close()` for process composition and final
  cleanup. `App` calls `close()` only after the supervisor has finished.
- Queue runners declare `DRAIN` for critical persistence or `DISCARD` for
  best-effort processing. `DRAIN` makes processing failures and drain timeouts
  terminal; `DISCARD` logs processing failures and drops pending final work.
  Final queue shutdown is bounded and must not run during an ordinary
  supervisor reconnect. Arctic's queued fire-and-forget writes are `DISCARD`;
  state-saver queues are `DRAIN`.

- Route restart triggers through `request_restart(reason)`; do not set the
  internal event directly.
- `connection_unavailable` is set from startup through successful probe, and
  during broker loss, restart, disconnect, and shutdown. A successful probe
  clears it.
- `App` explicitly supplies every runtime with `request_restart` and
  `connection_unavailable` through `bind_supervisor()` after constructing the
  supervisor.
- `Controller.sync()` races its sync task against `connection_unavailable`.
  Restart or stop must abort an in-progress sync without disabling trading.
- `cleanup_workload()` calls `workload.stop(reason)` only while the tracked task
  is still active. If it already completed, the supervisor collects its result
  without a redundant stop callback.
- If the workload task is still running after `stop()`, the supervisor cancels
  and awaits it. Unexpected workload or stop failures escape after terminal
  cleanup so the process exits unsuccessfully.
- Intentional socket closure is guarded by `_intentional_disconnect` so the
  resulting `disconnectedEvent` does not request another restart.
- Event handlers are attached once in `ConnectionSupervisor.__post_init__`.
  Avoid reconnect-time handler registration that would duplicate callbacks.

For live trading, the supervisor-side order is: mark connection unavailable,
set controller hold through workload cleanup, cancel workload work, disconnect,
connect, probe, and start the workload. During startup sync, broker position
freshness and the initial order scan happen while held. The coordinator then
releases hold before back-reporting known completed fills and running position
and bracket checks. Do not describe hold as lasting through the entire sync, and
do not move this boundary without reviewing controller reconciliation semantics.

## Broker Messages

Keep shared code definitions in `codes.py`. Supervisor-owned codes must be
included in the controller's effective ignore list so one broker event is not
logged by both components.

- `1100`: connectivity between IB and TWS/Gateway is lost.
- `2110`: TWS/Gateway connectivity to IB servers is broken.
- `1101`: connectivity restored with data lost; request a rebuild.
- `1102`: connectivity restored with data maintained. While waiting for broker
  recovery it wakes `ConnectionLost` for a probe. It does not prove that old
  subscriptions remain healthy. `restart_on_recovered_connection=True` makes
  it request a rebuild instead.
- `1300`: API socket port reset; request a rebuild.
- `2103`, `2105`, and `2157`: market-data, HMDS, or security-definition farm
  unavailable.
- `2104`, `2106`, and `2158`: corresponding farm available.
- Farm status messages provide context only. They do not mark the entire
  connection unavailable and do not move recovery forward.
- `2106` proves only that one HMDS farm is available. Never use it as proof
  that every farm or existing subscription recovered.
- Generic `updateEvent` traffic is ignored while broker connectivity is lost.
  Arbitrary traffic is not sufficient recovery evidence.

`log_datafarm_status` controls farm-status logging only. It must not change
broker-message policy.

## Stale Subscriptions And Timeouts

- `10182` means IB failed to provide live updates and is treated as evidence of
  a broken historical live-update subscription.
- In `Connected`, the first `10182` starts the state-owned
  `STALE_SUBSCRIPTION_RESTART_DELAY` timer, currently 180 seconds. Later
  `10182` messages reset it.
- When that quiet period expires, rebuild the workload rather than waiting for
  every streamer timeout. Leaving `Connected` cancels the timer.
- Broker connectivity loss takes priority over a pending stale-subscription
  timer because rebuilding subscriptions is pointless while connectivity is
  known to be unavailable.
- A `10182` soon after a rebuild indicates the farm disturbance was still in
  progress or a new subscription failed. Expect another delayed rebuild; use
  several nightly samples before changing the constant.
- Streamer timeouts remain an independent fallback and call
  `request_restart()`.
- `OPEN NOT LIQUID` does not mean no trades should occur. European hours for US
  futures can be marked non-liquid while stream freshness still matters.
- Fully `CLOSED` sessions may suppress timeout action where the streamer
  session logic already does so.
- Avoid complicated session heuristics for marginal restart savings. A future
  per-streamer rebuild is the preferred way to reduce the cost of one stale
  subscription, but broad cleanup remains appropriate while component state is
  shared or uncertain.

## Backoff And Efficiency

- Failed connection attempts use `retry_delay`; this loop should remain
  interruptible by stop.
- A failed probe performs cleanup first, then waits
  `connection_lost_retry_delay` before reconnecting. Cleanup must finish even if stop
  arrives; the subsequent wait is interruptible.
- Broker connectivity loss waits up to `auto_recovery_grace_period`, unless
  `1102` wakes it earlier, then probes before deciding whether to rebuild.
- Repeated `1100` or `2110` messages while already waiting should not create
  repeated transitions or restart cycles.
- Repeated restart requests before cleanup should coalesce into one rebuild.
- Do not add broad reconnect delay to the first ordinary restart. Add further
  bounded backoff only if logs show rapid repeated failures during a sustained
  outage.

## Controller Safety

- A routine broker outage is not by itself proof of unsafe order or position
  state and must not disable trading.
- Controller sync should skip or abort when `connection_unavailable` is set.
  It may disable trading only after current broker data is available and
  reconciliation genuinely fails.
- Duplicate order, execution, and commission events may be replayed after
  reconnect. Hold suppresses order-status persistence/logging and commission
  saves, but it does not suppress `onExecDetailsEvent`. Execution duplicate
  detection must prevent an already-accounted fill from changing position
  records twice.
- Trade relinking during the first sync is expected. Hold is released after
  broker position-source validation and the initial order scan, before known
  completed fills are back-reported and before position/bracket checks.
- A clean post-recovery sync should preserve broker positions and local strategy
  state. Backfill should cover the live-data gap without duplicating strategy
  actions.

## Extending Supervisor Behavior

Before adding a broker signal or restart source, classify it as one of:

1. Immediate rebuild because the socket or subscriptions must be recreated.
2. Broker-connectivity wait followed by a probe.
3. State-local health evidence such as timeout or stale subscription.
4. Informational context with no control action.

Then:

- Add shared broker codes to `codes.py`, not controller code.
- Keep temporary flags, timers, and callbacks inside the state that owns them.
- Cancel state-owned tasks and event subscriptions whenever that state exits.
- Use `wait_for_wakeup_or()` for waits that broker events must interrupt.
- Preserve stop/restart/workload race priority.
- Do not move supervisor flow into `Controller`; exchange only the restart
  callback and connection-unavailable event unless a new contract is required.
- Keep logs concise: one line for a meaningful trigger or transition is useful;
  logging the same fact from multiple components is not.
- Add focused tests for normal flow, repeated signals, signal precedence,
  cleanup ordering, cancellation, and stale events after a transition.

## Reviewing Logs

Read `docs/log-review-guidance.md` first. Live investigation files are normally
under `/home/tomek/ib_data/test_logs`:

- `haymakerLog`: supervisor, controller, workload, timeout, backfill, and trade
  lifecycle.
- `broker.log`: raw IB events. Its `ERROR` label often includes informational
  IB status codes; severity alone does not prove a fault.
- `strategies.log`: evidence that subscriptions resumed and strategy bars
  continued.
- Rotated files such as `haymakerLog.YYYY-MM-DD` are needed when an incident
  crosses midnight.

Establish the exact time window and timezone, then reconstruct every incident:

1. Last healthy subscription update, successful sync, or heartbeat.
2. Direct trigger: restart reason, socket disconnect, broker code,
   `timeoutEvent`, failed probe, stale-streamer timeout, workload completion, or
   stop.
3. State transition and whether repeated signals caused extra transitions.
4. Whether `connection_unavailable` and controller hold became active before
   workload cleanup, and whether controller sync skipped or aborted as expected.
5. Workload stop/cancellation and intentional socket disconnect.
6. Connection attempts, retry spacing, and connection success.
7. Probe result and elapsed time.
8. Workload start and controller sync result.
9. First resumed update for each relevant streamer.
10. Backfill range and whether it confirms missed live transactions.
11. Later timeout, `10182`, sync failure, duplicate event, or trading-disable
    symptom.

For each restart, record its direct trigger. Do not attribute a restart merely
to nearby farm chatter.

### Healthy Signatures

- Socket loss causes one restart request, hold precedes workload stop, retries
  occur at the configured interval, probe succeeds, sync completes, and data
  resumes.
- `1100` or `2110` causes one transition to `ConnectionLost`; repeated
  messages do not rebuild. `1102` or grace expiry causes a probe, and a
  successful probe returns to `Connected`.
- `10182` starts one 180-second quiet timer. A rebuild happens after the last
  message, before ordinary streamer timeouts, and backfill closes the gap.
- Farm broken/OK pairs are logged without state transitions.
- Controller sync during restart logs skip/abort rather than trading
  disablement.
- Existing trades may be relinked during startup sync while hold is active.
  Hold release after broker freshness and order scanning, followed by clean
  position/bracket checks, matches the current controller flow.

### Warning Signs And Inefficiencies

- More than one rebuild for one accepted restart request.
- Restart requests accepted during restart cleanup or reconnect backoff.
- Stop followed by a reconnect.
- Workload starts before a successful probe or starts more than once in a
  cycle.
- Hold is absent before workload cleanup, or released before broker position
  freshness and the initial order scan complete.
- Controller sync continues into correction or disables trading after
  connection loss/restart has begun.
- A successful probe is treated as proof that old subscriptions are fresh, but
  streamers later stop without `10182`, timeout, or rebuild.
- Repeated `10182` shortly after rebuild causes clustered broad restarts. This
  suggests the quiet period is shorter than the farm disturbance.
- A `10182` timer survives leaving `Connected` and triggers a stale restart
  in a later state.
- Connection attempts or failed-probe rebuilds loop rapidly without configured
  retry/backoff spacing.
- Repeated broker-loss messages create overlapping waits, probes, or restarts.
- Data-farm broken/OK messages alone cause connection-unavailable state or
  rebuild.
- Workload completion unexpectedly stops the supervisor; inspect the workload
  for a swallowed exception or premature return.
- Replayed order-status or commission events persist while hold is active, or a
  duplicate execution changes position records instead of being abandoned.
- Backfill repeatedly covers the same window, leaves gaps, or produces strategy
  actions inconsistent with preserved controller state.
- Logs contain the same broker event from controller and supervisor.
- Unexpected supervisor exception performs cleanup and exits; if the service
  repeatedly relaunches it, distinguish external process restart from internal
  recovery.

Quantify efficiency rather than calling a restart unnecessary: count rebuilds,
measure trigger-to-hold, cleanup, retry, probe, sync, and first-data times, and
compare downtime with configured delays. Separate confirmed facts, reasonable
inferences, and missing evidence.

## Validation

- Supervisor behavior:
  `.venv/bin/python -m pytest tests/test_supervisor.py tests/test_supervisor_package.py`
- Controller interaction:
  `.venv/bin/python -m pytest tests/test_controller.py tests/test_supervisor.py`
- Dataloader ownership:
  `.venv/bin/python -m pytest tests/test_dataloader_runtime.py tests/test_supervisor.py`
- Type checking:
  `.venv/bin/python -m mypy haymaker/supervisor tests/test_supervisor.py`
- For package exports, configuration, or broadly shared lifecycle changes, run
  the full `.venv/bin/python -m pytest` and `.venv/bin/python -m mypy .`.
