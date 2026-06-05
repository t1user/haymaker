# General Coding Rules

- Prefer minimal, surgical changes.
- Do not rewrite unrelated code.
- Preserve existing architecture unless instructed otherwise.
- Explain tradeoffs before major refactors.
- Prefer readability over cleverness.
- Avoid adding dependencies unless justified.
- Run tests after changes when possible.
- If tests fail, explain whether the failure is related to your changes.
- Keep functions focused and short.
- Avoid unnecessary abstraction.
- Prefer explicit code over metaprogramming.

# Workflow

- Before editing, briefly explain understanding of the task.
- For non-trivial tasks, create a short implementation plan.
- After changes, summarize modified files and rationale.
- Update relevant documentation when architecture, setup, commands, conventions,
  workflows, or important assumptions change.
- For research code under `haymaker/research`, read
  `haymaker/research/AGENTS.md` before editing.
- Avoid importing `haymaker.app` in tests unless explicitly needed. It sets up
  logging and imports runtime singletons; prefer lower-level modules for
  focused tests.
- Create docstrings for any new functions/classes/methods
- Use google-style, sphinx compatible docstrings
- If changing user-relevant behaviour, scan existing documentation and update any sections relevant to this newly changed behaviour

# Git

- Never commit unless explicitly instructed.
- When committing, add `[llm]` at the end of the commit message.
- Before committing, make sure any file with secrets is included in `.gitignore`.
- Never push.
- Do not modify `.env` files.
- Do not delete files without explanation.
- Run `mypy` and `pytest`, do not commit if there are any issues.

# Python Preferences

- Prefer type hints.
- Prefer `pathlib` over `os.path`.
- Use `pytest` to create tests.
- Use `black` for formatting.
- Add docstrings for new functions and classes.

# Naming

- Event handlers connected directly to `eventkit.Event` or `ib_insync` events
  should use camelCase callback names.
- For `ib_insync` events, prefer direct correspondence with the event name:
  `orderStatusEvent` -> `onOrderStatusEvent`,
  `updateEvent` -> `onUpdateEvent`,
  `timeoutEvent` -> `onTimeoutEvent`.
- Exception: handlers for `ib.errorEvent` should be named `onErrEvent`, not
  `onErrorEvent`, because IB uses this event for many informational broker
  messages that are not real errors, and callback names can surface in logs.
- Internal lifecycle hooks, injected callbacks, helpers, and ordinary methods
  should use standard Python snake_case.

# Validation

- Run the narrowest meaningful test command for the changed area first.
- For broad Python changes, run:

```bash
python -m pytest
```

- For focused research changes, run:

```bash
python -m pytest tests/test_research
```

- For typing and lint checks when touching research code, run when practical:

```bash
python -m mypy haymaker/research tests/test_research
python -m flake8 haymaker/research tests/test_research --select=F401,F821,F841,E501
```

- In Codex sandbox runs, prefer a `/tmp` Black cache to avoid hangs caused by
  restricted access to the normal user cache:

```bash
BLACK_CACHE_DIR=/tmp/haymaker-black-cache .venv/bin/python -m black --check --fast --target-version py312 --workers 1 <paths>
```

# Project Notes

- This is an Interactive Brokers trading framework built around `ib_insync`,
  event-driven runtime components, a historical dataloader, dataframe-first
  research tools.
- Live trading, controller sync, futures rolling, and order reconciliation are
  high-risk areas. Keep changes especially narrow and well verified there.
- IB/TWS connection outages, especially around a broker's daily restart period,
  are expected and should normally be recoverable. Do not treat a connection
  outage alone as an unsafe broker/local state; emergency trading disablement
  should be reserved for failed recovery, unreconciled state, or confirmed
  order/position safety issues.
- `haymaker.supervisor.ConnectionSupervisor` owns IB socket recovery for live
  trading and managed dataloader runs. It does not manage or restart the gateway
  process. Route new restart triggers through its `request_restart()` method.
  Broker messages are recovery context, not automatic restart triggers:
  `timeoutEvent` and connection probes are the first active health checks,
  `updateEvent` may probe recovery while already broker-degraded, and recent
  broker-degraded messages decide whether to wait for automatic broker recovery
  or request a reconnect/rebuild.
  Future app/supervisor architecture decisions should leave room for attached
  dataloader mode, where dataloader work borrows an externally managed `IB`
  connection and has no right to restart or disconnect it; see
  `docs/dataloader-connection-modes-plan.md`.
  Future dataloader design should treat `managed` and `attached` as the only
  connection modes. Legacy `reconnect`/`wait` behavior is a managed-mode recovery
  concern handled by supervisor decisions such as broker data-maintained vs
  data-lost messages, not a separate attached-mode policy.
- After the current app/supervisor architecture work settles, reconsider
  whether `app.recovery_warning_after` and `app.recovery_warning_interval`
  should remain user-facing config. They look like internal notification
  throttling details rather than trading/deployment policy.
- Graceful shutdown is not currently a broad architecture priority. Terminal
  `Ctrl-C` has historically been acceptable. Before service-manager deployment,
  prefer minimal signal hardening for `SIGINT`/`SIGTERM`: request supervisor stop,
  allow normal runtime cleanup to unwind, and drain critical async save queues
  with a short timeout. Do not introduce a broad shutdown framework unless a
  concrete cleanup need is identified.
- Configuration is primarily YAML-driven. Do not change real local `.env` files
  or credential files.
- See `docs/codebase-map.md` for the current repository map.

Dashboard is experimental and should not be looked at.
