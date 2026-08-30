## Issue: Controller forces IB reconnect on transient position snapshot race

Status: resolved for the synchronization safety path on 2026-08-30.

### Summary

A normal fill occurring during controller sync can make two non-atomic broker
position reads temporarily disagree. Before resolution, that disagreement was
incorrectly classified as a connection failure and triggered a full supervised
workload restart.

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

The affected `verify_broker_position_source()` performed inherently non-atomic reads:

```python
positions = tuple(ib.positions())
requested_positions = tuple(await ib.reqPositionsAsync())
```

A fill can arrive between those reads. That is exactly what happened: the cached snapshot was taken immediately before the NQU6 fill, while the requested snapshot included it.

The actual bug was that all verification failures were reduced to one Boolean result:

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

The affected implementation then called `self.ib.disconnect()` directly. That
bypassed the supervisor's restart API, so the supervisor correctly interpreted
it as an unexpected disconnection.

The confusing historical `Sync attempt 2/3` without `1/3` was secondary:
attempt 1 was logged only as `--- Sync ---`, while numbered logging started
with later attempts.

### Resolution

Broker verification now returns one of three explicit outcomes:

- `MATCH`
- `SNAPSHOT_DISAGREEMENT`
- `REQUEST_UNAVAILABLE`

A cached/fresh disagreement consumes a local retry and never requests a
reconnect. If the next request converges, synchronization continues normally;
persistent disagreement exhausts the bounded retries and follows the existing
fail-closed policy. Timeout or request failure alone asks the owning supervisor
to recover broker state. Controller does not disconnect the IB socket.

The successful `reqPositionsAsync()` response is the sole broker-position
snapshot consumed by `PositionSync` during that pass. Position correction is
also deferred for Contracts with active attributed OPEN/CLOSE orders, avoiding
correction while a one-to-one transaction can still fill.

Regression coverage reproduces a fill between the cached and requested reads,
verifies convergence without restart, verifies unavailable-request recovery,
and verifies persistent disagreement fails closed without a spurious restart.

Every reconciliation pass is now logged consistently as `Sync attempt N/M`,
including the first attempt, so incident timelines no longer mix a generic
banner with numbered retries.

