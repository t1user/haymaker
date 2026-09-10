# Reviewing Haymaker logs

Use this guide for log discovery and operational evidence. Keep the review
within the requested incident, component, strategy or time window; connectivity
is only one possible focus.

## Locate the run's destinations

- Start with any supplied log path and the strategy project's own `AGENTS.md`,
  launcher and selected configuration. Local operational notes are leads;
  confirm they describe the run being reviewed.
- Haymaker configuration precedence is bundled profile, environment-selected
  YAML, `-f/--file` YAML, then `-s/--set-option PATH VALUE`. The environment
  selectors are `HAYMAKER_HAYMAKER_CONFIG_OVERRIDES` for live trading and
  `HAYMAKER_DATALOADER_CONFIG_OVERRIDES` for the dataloader. They select files,
  not individual setting values. See
  [configuration](source/configuration.rst) and
  [the loader](../haymaker/config/loader.py).
- Trace effective `logging.config_file`, `logging.directory` and
  `storage.base_directory`. The CLI passes these to
  [setup_logging](../haymaker/logging/setup.py) before building the runtime
  or importing a strategy. A logging config filename is tried relative to the
  process working directory, then under `haymaker/logging/`.
- Built-in handler factories resolve their output directory as
  `Path.home() / storage.base_directory / logging.directory`. Absolute
  base or log directories override preceding path segments. Bundled defaults
  are `ib_data` and `logs`, giving `~/ib_data/logs`; this is a fallback,
  not evidence of the actual destination.
- Inspect the selected logging YAML's handlers and logger assignments.
  Built-in factories join bare filenames to the resolved directory, defaulting
  to `haymakerLog` when none is supplied. Filenames with a directory component
  are kept as configured; relative ones are relative to the process working
  directory, not the YAML file. Custom handlers may have their own path rules
  or send output elsewhere. See
  [handler factories](../haymaker/logging/handlers.py) and
  [logging configuration](source/logging.rst).
- If configuration is unavailable or may have changed since launch, the
  running process's open log destinations and service output can establish
  where it actually writes. Do not launch a strategy or call logging setup
  merely to discover paths: those entrypoints have runtime/output side effects.
  If unresolved, request the launch command, selected profile or log location.

Bundled destinations are:

| Run/logger | File | Evidence |
| --- | --- | --- |
| Live `haymaker` | `haymakerLog` | Framework lifecycle, reconciliation, backfill and execution |
| Live `strategy` | `strategies.log` | Strategy/component output, including data updates when logged |
| Live `broker` | `broker.log` | Raw IB events; requires `logging.log_broker: true` |
| Dataloader `haymaker` | `dataloaderLog_YYYYMMDD_HHMM` | Historical requests, pacing and persistence |

Live defaults rotate at midnight, producing files such as
`haymakerLog.YYYY-MM-DD`. Dataloader filenames receive a UTC creation-time
suffix. Bundled record formatters use UTC; rotation follows the selected
handler's timezone settings. Include rotations spanning the requested window.
Custom logging configuration can change names, formats, levels and destinations.

## Establish the evidence

- Identify the actual strategy module, runtime/process start and log cutoff.
  A supervised workload restart does not reload Python code. When attributing
  behavior to a framework change, check the running version; the current
  checkout alone is insufficient.
- State the time window and timezone. File presence alone does not prove the
  current process writes it, and a logger may be silent because of its level.
- Match records across relevant logs by timestamps and Contract/order/execution
  identifiers. Raw broker `ERROR` labels include informational status messages;
  interpret the code and resulting behavior. Use the
  [IB message guide](source/ib_message_codes.rst).
- For accounting or roll questions, logs may need corroboration from Book's
  persisted order/fill and state evidence. Distinguish repeated callbacks from
  repeated broker executions or persisted transactions.
- Quote or summarize supporting records. Separate confirmed facts, inference
  and missing evidence; absence of logged strategy data is not proof of a stall.

## Connectivity and recovery investigations

Use the [supervisor guide](../haymaker/supervisor/AGENTS.md) for state transitions,
code policy and hold ordering. Verify four separate outcomes: API connectivity,
broker request usability, resumed subscriptions and order/position reconciliation.
A successful probe alone does not prove the other outcomes.

Reconstruct the relevant sequence, allowing for probe-only recovery:

1. Last healthy subscription update, request or sync.
2. Direct trigger: broker code, disconnect, idle/stale timeout, explicit
   restart/stop, or workload completion.
3. State transition, connection-unavailable state and Controller hold where
   applicable; note whether sync skipped or aborted.
4. Cleanup, reconnect attempts and probe results, with configured retry delays.
5. Workload start, order rebinding, hold release and reconciliation.
6. First resumed update for each relevant streamer and the backfill range.
7. Later stale-data, duplicate-execution, sync-failure or trading-disable symptoms.

Attribute each restart to its direct trigger, not adjacent farm-status chatter.
Look for repeated rebuilds, stop followed by reconnect, work starting before
probe success, stale state timers firing later, or corrections continuing after
connection loss. For replayed executions, verify fill idempotence rather than
assuming repeated records changed accounting.

When assessing efficiency, count rebuilds and measure cleanup, retry, probe, sync
and first-data times against the effective settings. Distinguish external service
relaunches from internal workload restarts.

## Report

Give the paths and time window reviewed, important events, confirmed outcome,
uncertainties and any useful next observation. For recovery, report the four
outcomes separately. Recommend a code change only when evidence supports a
specific failure mode or the user asks for implementation.
