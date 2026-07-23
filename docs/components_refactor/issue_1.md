## Issue: Controller forces IB reconnect on transient position snapshot race

### Summary

A normal fill occurring during controller sync can make two non-atomic broker position reads temporarily disagree. The disagreement is incorrectly classified as a connection failure, causing the controller to disconnect the IB socket and trigger a full supervised workload restart.

This was not an external connectivity loss.

### Concrete incident: 2026-07-23 13:01 UTC

1. `rcross_NQ` opened a short NQU6 position at `13:01:05.452` ([strategies.log:19499](</home/tomek/ib_data/test_logs/strategies.log:19499>)).

2. Scheduled sync started at `13:01:07.680` ([haymakerLog:531](</home/tomek/ib_data/test_logs/haymakerLog:531>)).

3. The NQU6 order filled at `13:01:07.681–07.682` ([broker.log:5273](</home/tomek/ib_data/test_logs/broker.log:5273>), [haymakerLog:532](</home/tomek/ib_data/test_logs/haymakerLog:532>)).

4. Sync compared two position sources:

   - Cached `ib.positions()`: did not contain NQU6.
   - Fresh `reqPositionsAsync()`: contained `NQU6: -1`.

   The only material difference was the position that had just filled ([haymakerLog:541](</home/tomek/ib_data/test_logs/haymakerLog:541>)).

5. After the configured one-second retry delay, the controller disconnected the socket. The supervisor consequently logged:

   `Restart requested: IB socket disconnected`

   at `13:01:08.723` ([haymakerLog:545](</home/tomek/ib_data/test_logs/haymakerLog:545>), [broker.log:5308](</home/tomek/ib_data/test_logs/broker.log:5308>)).

6. Attempt 2 ran against the socket that the controller had just disconnected, producing the warning:

   `broker position request failed: ConnectionError('Not connected')`

   ([haymakerLog:546](</home/tomek/ib_data/test_logs/haymakerLog:546>)).

7. The supervisor reconnected normally. A fresh startup sync saw `NQU6: -1` in both state sources and completed at `13:01:09.706` ([haymakerLog:568](</home/tomek/ib_data/test_logs/haymakerLog:568>)).

The NQ protective stop had reached `PreSubmitted` before disconnection and was recovered afterward. No lost position, order, or market data was observed.

### Root cause

[`verify_broker_position_source()`](</home/tomek/haymaker/haymaker/controller/sync_coordinator.py:253>) performs inherently non-atomic reads:

```python
positions = tuple(ib.positions())
requested_positions = tuple(await ib.reqPositionsAsync())
```

A fill can arrive between those reads. That is exactly what happened: the cached snapshot was taken immediately before the NQU6 fill, while the requested snapshot included it.

The actual bug is that all verification failures are reduced to the same Boolean result:

```python
if not await verify_broker_position_source(...):
    self.request_restart = True
    return False
```

([sync_coordinator.py:104](</home/tomek/haymaker/haymaker/controller/sync_coordinator.py:104>))

This conflates:

- A transient snapshot disagreement.
- A broker request timeout.
- A disconnected or unusable broker connection.
- A persistent position-source inconsistency.

The controller then calls `self.ib.disconnect()` directly ([controller.py:409](</home/tomek/haymaker/haymaker/controller/controller.py:409>)). Because this bypasses the supervisor’s restart API, the supervisor correctly interprets it as an unexpected disconnection ([supervisor.py:354](</home/tomek/haymaker/haymaker/supervisor/supervisor.py:354>)).

The confusing `Sync attempt 2/3` without `1/3` is secondary: attempt 1 is logged only as `--- Sync ---`; numbered logging starts when `attempt > 1` ([controller.py:392](</home/tomek/haymaker/haymaker/controller/controller.py:392>)).

### Proposed solution

After the current refactor:

1. Represent verification outcomes explicitly, for example:

   - `MATCH`
   - `SNAPSHOT_DISAGREEMENT`
   - `REQUEST_UNAVAILABLE`

2. Treat an initial snapshot disagreement as retry-only. Take fresh readings after the configured delay without disconnecting.

3. Escalate only a repeated disagreement or an actual request/connectivity failure.

4. Route any required reconnect through the runtime/supervisor `request_restart()` callback. The controller must not call `ib.disconnect()` directly.

5. Longer-term, use one authoritative broker-position snapshot per sync pass rather than comparing or consuming independently timed snapshots.

6. Log attempt 1 explicitly as `Sync attempt 1/3` for an unambiguous incident timeline.

### Acceptance criteria

- A fill arriving between the cached and requested reads causes a retry, not a disconnect.
- If the next read converges, sync completes without restarting or disabling trading.
- A genuine request timeout/failure requests one supervisor-owned restart.
- No sync attempt runs against a socket the controller deliberately disconnected.
- Persistent non-convergence still follows the defined safety policy.
- Regression tests simulate a fill appearing between the two reads.
- Existing test [`test_broker_position_source_disagreement_disables_trading`](</home/tomek/haymaker/tests/test_controller.py:496>) is revised because it currently codifies the problematic outcome.

