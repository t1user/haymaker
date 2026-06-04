# Dataloader Connection Modes Plan

This is a future implementation plan. Do not implement it until the live
application and connection supervisor refactor are stable enough to use as the
primary runtime.

## Context

The dataloader currently has legacy `run_mode` values such as `reconnect` and
`wait`. Those names describe older managed-connection recovery behavior. They do
not describe who owns the IB socket and should not become separate future
connection modes.

Future dataloader work should center on connection ownership:

- connection ownership
- dataloader request retry/readiness behavior

This matters because the dataloader should support both standalone operation and
operation attached to an existing externally managed IB connection.

## Goals

- Keep standalone dataloader operation available.
- Allow dataloader work to attach to an existing `ib_insync.IB` connection.
- Prevent attached dataloader work from restarting or disconnecting a live
  trading connection.
- Keep `ConnectionSupervisor` focused on owned socket recovery.
- Provide one helper that selects the correct dataloader mode from config.

## Non-Goals

- Do not change live trading ownership rules for this plan.
- Do not make `ConnectionSupervisor` understand live-vs-dataloader workload
  semantics.
- Do not give attached dataloader mode access to restart, connect, or disconnect
  operations.
- Do not implement this before the live runtime is usable.

## Connection Modes

The future dataloader should expose two connection modes only:

- `managed`
- `attached`

### Managed

Managed mode is the standalone dataloader path.

- The dataloader creates or receives its own `IB` object.
- The dataloader uses `ConnectionSupervisor`.
- The supervisor may connect, disconnect, probe, and restart that dataloader
  socket.
- Recovery behavior that used to be called `reconnect` or `wait` should be
  represented by supervisor decisions, especially whether broker recovery
  maintained or lost data.
- The dataloader can stop its supervisor when the workload finishes.

### Attached

Attached mode borrows an existing `IB` connection owned by something else,
usually a live process.

- The dataloader receives an already managed `IB` object.
- The dataloader does not create a `ConnectionSupervisor`.
- The dataloader never calls `connectAsync()`, `disconnect()`, or
  `request_restart()`.
- The dataloader passively waits for `ib.isConnected()` and, if appropriate, a
  successful probe before starting or resuming work.
- Connection recovery is owned by the external runtime. The dataloader should
  wait, retry its own broker requests, or cancel its own work, but it must not
  alter socket lifecycle.

## Workload Recovery

Do not preserve `reconnect` and `wait` as independent future modes.

In managed mode, the supervisor owns recovery decisions. If broker messages or
probe behavior indicate that data was maintained, the dataloader should be able
to continue or wait for requests to resume. If data was lost or recovery does not
complete, the supervisor can restart the managed dataloader workload.

In attached mode, there is no dataloader-owned reconnection. The dataloader must
only wait for the externally managed connection to become usable again and retry
or cancel its own broker requests. It must not request connection restart.

## Helper Shape

Provide a helper that selects the correct path from config, for example:

```python
connection_mode = CONFIG.get("connection_mode", "managed")
```

The helper should dispatch roughly as follows:

- `connection_mode == "managed"`: build `ManagedDataloaderConnection` and run it
  under `ConnectionSupervisor`.
- `connection_mode == "attached"`: build `AttachedDataloaderConnection` around
  the provided `IB` object and run the dataloader workload without supervisor.

Attached mode should likely expose an async-first API because a live process will
already own the event loop. A synchronous CLI wrapper can continue to exist for
managed standalone operation.

## Request Retry and Readiness

The dataloader request layer will need additional cleanup before attached mode is
safe:

- Broker request paths should share a small readiness/retry helper.
- Workers should wait for externally restored connectivity after
  `ConnectionError`.
- Producer, contract selection, and headstamp requests should avoid tight retry
  loops while disconnected.
- Attached mode should be able to retry dataloader work without escalating to
  connection restart.

## Logging

Dataloader runs are not intended to be connected to the live Telegram alerting
path. Dataloader logging can therefore be more liberal than live trading logs.

Still, attached mode should not produce misleading live-critical messages when it
shares a process with live trading. Reserve high-severity logs for dataloader
failures that require user attention, not ordinary waits for an externally
managed connection to recover.

## Supervisor and App Constraints

Future changes to `haymaker.app` and `haymaker.supervisor` should keep this plan
in mind:

- `ConnectionSupervisor` should remain workload-agnostic.
- The supervisor should own socket recovery only when it is explicitly used.
- Live runtime must remain the primary owner of live trading connection recovery.
- Dataloader-specific behavior should live in dataloader adapters or request
  helpers, not in the live app.
