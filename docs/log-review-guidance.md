# Log Review Guidance

Use this note when asked to review Haymaker logs. It is a checklist for agents,
not a replacement for the user's specific question.

## First Clarify Scope

- If the user asks about a specific incident, component, symbol, order, strategy,
  or time window, answer that question first.
- Do not force every log review into a supervisor investigation. Supervisor
  recovery is one important case, but log reviews can also be about controller
  sync, order reconciliation, strategy behavior, dataloader work, persistence,
  futures rolling, or research jobs.
- Keep timestamps explicit. When possible, build a chronological sequence of
  events before explaining cause or proposing code changes.
- Separate facts from inference. Quote or summarize the log line that supports
  each important conclusion.

## Supervisor Recovery Focus

When the review is about IB connectivity, broker recovery, or the supervisor,
track whether each broker event preserves:

- API socket connectivity: did `IB.isConnected()` remain true, did
  `disconnectedEvent` fire, or did the supervisor reconnect?
- Broker request usability: did probes, controller sync, historical-data
  requests, or market-data requests succeed after the event?
- Subscriptions: did existing live market-data, realtime-bar, historical-update,
  or streamer subscriptions keep delivering data without being rebuilt?
- Workload continuity: did the live runtime or dataloader workload continue,
  stop, restart, or get cancelled?

The key observation is not just "which code arrived." The key observation is
what was still working afterward.

## Broker Codes To Track

- `1100`: IB reports connectivity between IB and TWS/Gateway is lost. Check
  whether Haymaker moves into broker wait, whether the API socket stays open,
  and whether subscriptions resume or later become stale.
- `1101`: Connectivity restored with data lost. Expect a restart/rebuild path;
  verify that subscriptions are resubmitted and workload restarts cleanly.
- `1102`: Connectivity restored with data maintained. Verify whether existing
  subscriptions actually keep producing data. A successful probe only proves
  current request connectivity, not subscription freshness.
- `1300`: API socket port reset. Expect reconnect/rebuild behavior; verify no
  stale workload keeps running on the old socket.
- `2110`, `2103`, `2105`, `2157`, `10182`: Broker farm or data connectivity
  degradation. Check whether Haymaker waits for broker recovery, whether
  `updateEvent` or later broker messages trigger a probe, and whether waiting
  avoids unnecessary reconnects without leaving stale subscriptions undetected.

Also note repeated request errors, pacing violations, `ConnectionError`,
streamer timeout messages, controller sync skips, and order/position mismatch
logs around the same time window.

## Event Order To Reconstruct

For supervisor-related logs, reconstruct this sequence when possible:

1. Last known healthy data point or heartbeat before the broker event.
2. Broker code, `timeoutEvent`, `updateEvent`, or `disconnectedEvent`.
3. Supervisor state transition, if logged.
4. Probe attempt and result.
5. Workload stop/start/restart, if any.
6. Reconnect attempt count and whether connection parameters changed.
7. First successful broker request after recovery.
8. First resumed subscription update after recovery.
9. Any later streamer timeout or stale-data symptom.

If the logs do not show enough detail for one of these points, say that rather
than filling the gap.

## Signs Of Healthy Recovery

- Broker-degraded messages enter a broker wait without immediate unnecessary
  reconnect when the socket remains connected.
- `updateEvent` or `1102` is followed by a successful probe while waiting.
- A `1101`, `1300`, failed probe, or unexpected socket disconnect causes one
  coherent reconnect/rebuild cycle, not repeated overlapping restarts.
- Workload cleanup happens once per restart/shutdown and a new workload starts
  only after connection probing succeeds.
- Existing subscriptions either continue producing updates after maintained
  recovery or stale subscriptions are detected promptly by streamer timeouts and
  handled by restart.

## Warning Signs

- Connection appears restored, but streamers stop receiving updates and no
  timeout/restart follows.
- Repeated broker wait cycles hide a condition that clearly requires reconnect.
- Multiple restart requests overlap, or a stop request is followed by reconnect.
- Controller sync treats a routine outage as an unsafe unreconciled trading
  state instead of a recoverable missing-connection condition.
- Dataloader logs suggest attached work is restarting or disconnecting a socket
  owned by live trading.
- Log severity is misleading: routine broker recovery should not look like an
  unrecoverable trading safety failure unless reconciliation or order/position
  state is actually unsafe.

## Output Expectations

Prefer a compact incident report:

- Time window reviewed.
- Timeline of important log events.
- What was observed about connectivity, request usability, subscriptions, and
  workload continuity.
- What is confirmed, what is inferred, and what is unknown.
- Suggested next observation or instrumentation if the logs are insufficient.

Do not propose code changes until the log evidence supports a specific failure
mode or the user explicitly asks for implementation.
