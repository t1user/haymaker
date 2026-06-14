# Dataloader Connection Ownership Notes

This note records the current dataloader connection decision and optional future
work. The active implementation target is supervised-only.

## Context

The dataloader creates its own `ib_insync.IB` client and runs it under
`ConnectionSupervisor`. It cannot borrow the live runtime's `IB` object when the
live process runs in another interpreter. Running dataloader next to live trading
therefore means using a separate IB API client ID, not an attached object.

## Goals

- Keep dataloader operation supervised by `ConnectionSupervisor`.
- Keep the dataloader client ID distinct from the live runtime's expected
  `clientId=0`.
- Keep `ConnectionSupervisor` focused on owned socket recovery, not
  TWS/Gateway process management.
- Defer any Watchdog/IBC process-management runner until the core dataloader
  retry and pacing model is stable.

## Non-Goals

- Do not change live trading ownership rules.
- Do not make `ConnectionSupervisor` understand live-vs-dataloader workload
  semantics.
- Do not add attached/borrowed-IB mode without a concrete same-interpreter
  caller.
- Do not add Watchdog/IBC mode before pacing and retry semantics are settled.

## Active Connection Model

The dataloader exposes one active connection model:

- supervised

Supervised mode is the standalone dataloader path.

- The dataloader creates its own `IB` object.
- The dataloader uses `ConnectionSupervisor`.
- The supervisor may connect, disconnect, probe, and restart that dataloader
  socket.
- The dataloader default API client ID is `1`; live runtime is expected to use
  `clientId=0`.
- Duplicate client ID is a connection configuration failure.
- Normal dataloader workload completion should let the supervisor stop itself.

## Workload Recovery

Do not preserve `reconnect` and `wait` as independent modes.

In supervised mode, the supervisor owns recovery decisions. If broker messages or
probe behavior indicate that data was maintained, the dataloader should be able
to continue or wait for requests to resume. If data was lost or recovery does not
complete, the supervisor can restart the managed dataloader workload.

## Request Retry and Readiness

The dataloader request layer still needs additional cleanup:

- Broker request paths should share a small readiness/retry helper.
- Producer, contract selection, and headstamp requests should avoid tight retry
  loops while disconnected.
- Workers should follow the same supervised connection-loss policy as the rest
  of the request layer.

## Logging

Dataloader runs are not intended to be connected to the live Telegram alerting
path. Dataloader logging can therefore be more liberal than live trading logs.

Reserve high-severity logs for dataloader failures that require user attention,
not ordinary waits for supervisor-managed recovery.

## Supervisor and App Constraints

Future changes to `haymaker.app` and `haymaker.supervisor` should keep this plan
in mind:

- `ConnectionSupervisor` should remain workload-agnostic.
- The supervisor should own socket recovery only when it is explicitly used.
- Live runtime must remain the primary owner of live trading connection recovery.
- Dataloader-specific behavior should live in dataloader adapters or request
  helpers, not in the live app.

## Optional Future Watchdog Runner

After the core dataloader refactor is complete, it may be worth reconsidering a
gateway-managing Watchdog/IBC runner for paper-account, multi-day data loading.
That would be a sibling runner, not a replacement for supervised mode and not a
feature inside `ConnectionSupervisor`. It should wait until pacing compliance and
request retry semantics are settled.
