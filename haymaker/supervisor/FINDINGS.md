# Supervisor Findings

Date: 2026-06-29

This note records findings from recent live-log reviews while the
state-machine supervisor was running, before real-world testing of
`supervisor_one.py`.

## Current Assessment

The state-machine supervisor behaved correctly during the reviewed IB outages
and farm degradation periods. It waited through real broker-side disruption,
avoided unsafe corrective action, restarted the workload when needed, and
preserved order and position state across recovery.

No reviewed logs showed evidence of:

- controller trading disablement caused by supervisor recovery,
- emergency close,
- unreconciled broker/local state after recovery,
- false position verification,
- order mismatch,
- traceback or supervisor crash.

## Useful Behaviors To Preserve

- Broker data-maintained recovery messages should not force an immediate broad
  restart by themselves.
- `timeoutEvent` and lightweight historical probes remain the useful live
  health checks.
- Real socket outages should put the workload into a held/restarting path, then
  require successful probe and controller sync before normal operation resumes.
- Recovered state should be judged by confirmed socket readiness and
  controller sync, not by broker informational messages alone.
- Futures roll scheduling must remain app-lifetime behavior, not something
  recreated by each connection or workload cycle.

## Main Inefficiencies Observed

### Broad Restart For One Stale Streamer

A stale streamer timeout can currently request a full workload restart. In the
reviewed logs, the clearest example was `RTY` during an `OPEN NOT LIQUID`
period. Historical backfill later showed data existed for the timed-out window,
so the most likely explanation is a stale subscription rather than a real
no-trade period.

This does not look unsafe, but it is broader than necessary. The preferred
future direction is per-streamer recovery: resubscribe or restart only the
affected streamer and its downstream path where possible. Occasional unnecessary
per-streamer restarts during quiet periods are acceptable if they avoid broader
workload churn.

### Repeated Reconnect Attempts During Real Outage

During a real IB outage, the supervisor retried connection attempts repeatedly
at a fixed cadence. This recovered correctly, but after repeated failed
connection attempts a modest backoff would reduce noise and pointless churn.

The backoff should not delay the first recovery attempts too much. A simple
bounded strategy after several consecutive failures is preferable to a complex
policy.

### Broker Messages Are Mostly Weak Signals

IB broker issue/recovery messages were noisy and not precise enough to be the
primary recovery signal. The useful split remains:

- data-lost or socket-reset style messages can request restart,
- hard broker-connectivity-lost messages can move the supervisor into a
  waiting/probing state,
- data-maintained and farm messages should generally be hints only,
  with probes and stream health deciding whether work is actually healthy.

### Deferred 10182 Handling

Recent nightly logs suggest that `10182` live-update failure clusters probably
invalidate existing `reqHistoricalData(..., keepUpToDate=True)` subscriptions.
When the state-machine supervisor waited for normal stale-streamer detection,
the later restart was followed by backfill for the affected window, which
supports the interpretation that the stream had really stopped.

A possible future optimization is to treat a `10182` cluster as "streamer
rebuild needed", wait for a short quiet period after weak data-farm messages
such as `2105`/`10182`, then restart the workload without waiting for streamer
timeouts. `2106` only says an HMDS farm is OK again, so it should not by itself
be treated as a complete recovery signal across all farms/contracts.

This optimization is intentionally deferred. The current behavior misses a few
minutes of live updates during the nightly IB reset, but backfills immediately
after restart and avoids extra policy based on weak broker messages. Revisit
only if `10182` starts appearing during active sessions, backfill becomes
unreliable, or another streamer type proves it needs a different recovery path.

## Non-Liquid Session Timeouts

Changing timeout policy purely because a contract is `OPEN NOT LIQUID` is
probably not worth much complexity. European hours for US futures are often
marked non-liquid, but stale data during those hours can still be a real stream
problem.

Prefer cheaper recovery over smarter classification:

- keep strict behavior for `OPEN LIQUID`,
- avoid action while fully `CLOSED`,
- for `OPEN NOT LIQUID`, allow the same stale detection but make the response
  cheap by restarting only the affected streamer when feasible.

## Candidate Direction After Testing `supervisor_one.py`

After real-world testing, compare the two implementations using:

- correctness during real IB outages,
- avoidance of unsafe or hasty controller actions,
- restart scope and churn,
- clarity of lifecycle ownership,
- ease of adding per-streamer restart,
- ease of adding reconnect backoff,
- readability and testability,
- readability of logs.

Then pick one implementation, fold in the best ideas from the other, add only
the improvements that still look useful after live testing, and delete the
losing implementation.
