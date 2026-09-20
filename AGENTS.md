# Working in Haymaker

Haymaker is an event-driven Interactive Brokers framework built on `ib_insync`,
with a historical dataloader and separate dataframe research tools. Live order
execution, reconciliation and futures rolling are high-risk code.

## Workflow and scope

- Explain your understanding before editing; make a short plan for non-trivial
  work. Reviews and discussions are read-only unless changes are requested.
- Prefer minimal, explicit changes and short functions. Preserve unrelated
  work. Ask before architecture changes or adding dependencies; explain the
  tradeoffs.
- Update relevant user and agent documentation when behavior, architecture,
  setup or conventions change. Document current contracts, not implementation
  history or historical test counts.
- Keep shared rules here and subsystem rules in the owning scoped guide; link
  instead of repeating them. Machine paths, launch profiles and account/port
  preferences belong in the user's strategy project's `AGENTS.md`.
- Use type hints, `pathlib`, pytest and Black. New public APIs need usage-focused,
  Google-style, Sphinx-compatible docstrings; internal documentation should
  explain ownership, ordering and non-obvious invariants.
- Direct event callbacks use camelCase matching the event name
  (`orderStatusEvent -> onOrderStatusEvent`). The IB `errorEvent` callback is
  `onErrEvent`, since many messages are informational. Helpers, internal hooks
  and injected callbacks use snake_case.
- Operational logs use the word `error` only for genuine failures needing
  attention. Routine recovered outages must not leak exception names such as
  `TimeoutError` into lifecycle messages.
- Never commit without explicit instruction; append `[llm]` to commit messages.
  Never push. Before committing, run pytest and mypy successfully and ensure
  secrets are ignored. Do not modify `.env` or credential files.
- Explain deletions. Do not execute database migrations or change production
  configuration as part of implementation.

## Where to look

- For strategy composition or component changes, read
  [components guidance](haymaker/components/AGENTS.md), including when helping
  with a user strategy outside that package. Also read that strategy project's
  own guidance for its configuration, data locations and operational choices.
- Read the scoped guidance before work in
  [supervisor](haymaker/supervisor/AGENTS.md),
  [dataloader](haymaker/dataloader/AGENTS.md), or
  [research](haymaker/research/AGENTS.md).
- [Codebase map](docs/codebase-map.md): module ownership and runtime flow.
- [Execution guide](docs/source/execution.rst): public composition and recovery
  examples; [storage guide](docs/source/storage.rst): defaults and advanced
  persistence configuration; [configuration guide](docs/source/configuration.rst):
  profiles and CLI overrides.
- Read [log-review guidance](docs/log-review-guidance.md) before reviewing logs;
  apply its general evidence discipline without expanding the requested scope.
- Dashboard is experimental and should not be inspected. The event-driven
  backtester is also experimental; do not imply it certifies live recovery.

## Validation

Use the project interpreter, `.venv/bin/python`. Run the narrowest meaningful
tests first; for broad Python changes run the full suite. Report failing checks
and distinguish pre-existing failures from regressions.

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy haymaker
.venv/bin/python -m black --check --fast --target-version py312 <changed-python-paths>
.venv/bin/python -m sphinx -n -W --keep-going -b html docs/source /tmp/haymaker-docs
```

- Sphinx reference checks need external inventories; do not suppress missing
  references to get a clean result.
- Use `tests/runtime_helpers.py` and `atom_runtime` / `atom_runtime_factory`
  for IB, Book, registry, Controller, restart callbacks and storage dependencies.
  Do not scatter ad hoc Atom runtime monkeypatches.
- Avoid importing `haymaker.app` in focused tests unless testing that entrypoint.
  Prefer lower-level components to avoid logging/runtime side effects.
- Independent broker-boundary coverage uses `tests/episode_harness.py`,
  `test_episode_end_to_end.py`, `test_direct_end_to_end.py`,
  `test_roll_end_to_end.py`, and `test_bracket_recovery.py`.
  Test durable recovery without replaying transient callbacks.

## Architecture and Atom contracts

- Built-in trading components live only in `haymaker.components`; execution,
  routing, bracket legs and roll executors are under `components.execution`.
  Atom/Pipe, Runtime, Book, Controller, Trader, storage and contract management
  remain outside. Do not add compatibility modules, aliases or dual schemas.
- `Atom` is an arbitrary-message composition primitive. Base `onData` raises;
  subclasses emit explicitly. Synchronous `onStart(data, source)` propagates
  arbitrary startup data without automatic mutation or reserved fields.
  Preserve `onFeedback` and reverse feedback connections.
- `connect` validates every target before wiring any. `validate_source` must
  raise for missing upstream capabilities; message-value validation belongs in
  `onData`. Use real classes/protocols, not class-name checks. Do not add
  `input_type`/`output_type`, graph-wide type negotiation or emission detection.
- Fan-out shares one object reference. Mutating branches must copy explicitly;
  do not add blanket deep copies of dataframes or broker objects.
- Dataclass Atoms require `@dataclass(eq=False)` at every decorated inheritance
  level to retain node identity equality.
- Reuse `haymaker.validators` for primitive normalization; keep domain-specific
  checks with their owning component.

## Runtime and contract ownership

- Linux, one user strategy module and one `RuntimeContext` per process are the
  supported live lifecycle. Process registries such as `Streamer.instances`
  are not reset for same-process application reuse.
- `LiveRuntime` builds services and installs a ready, passive RuntimeContext
  before importing the strategy. Runtime does not inspect module data.
  `run_started_at` stays fixed; `workload_generation` increments before each
  supervised workload. These do not replace generic startup data.
- Contract blueprints are registered at component construction. Runtime
  initializes Contract details/selectors before Controller reconciliation;
  streamers start afterward. Rebuild selectors using one naive UTC timestamp
  per workload; supervised daily restarts refresh ACTIVE/NEXT roles.
- `Atom.contract` resolves `which_contract`; `contract_blueprint` returns a
  copy of the declaration. Accessing `contract_selector` raises until both
  Contract registration and selector initialization are available.
- Registry identity is derived from the snapshotted user Contract declaration;
  qualified members map explicitly to that blueprint. Overlapping qualification
  results from different declarations must fail atomically, never be guessed
  from symbol fields.
- ACTIVE is the market-data/roll reference, NEXT an early-entry candidate.
  `which_contract` does not redefine ACTIVE. Full-chain membership is public;
  explicit `nth_contract(n)` raises out of range, while NEXT retains its
  intentional last-available fallback.
- `ConnectionSupervisor` owns IB socket recovery for live and dataloader runs,
  not gateway process management. Route restart requests through it. An outage
  alone is not unsafe trading state; disable trading for failed recovery or
  confirmed unreconciled order/position safety problems.
- Timers for Controller sync and daily rolling are process-owned. Reconnects
  must not duplicate them. Component timeout lifetimes are described in the
  components guide.
- CLI owns logging setup/shutdown; each destination has its own queue/listener.
  Messaging handlers are optional configuration, not runtime dependencies.
- First SIGTERM requests graceful supervisor stop; a second may terminate
  stuck cleanup. Keep shutdown changes narrow and drain critical queues with
  bounded waits rather than introducing a broader lifecycle framework.

## Book, Controller and recovery safety

- Haymaker owns every order/position in the connected account or subaccount.
  `Book` owns typed accounting, recovery, fill idempotence, persistence and
  optional blotter access. It performs neither broker calls nor Portfolio
  calculations. Controller owns broker submission/cancellation, immediate
  OrderInfo registration, rebinding, fills/commissions and reconciliation.
- `haymaker/book/` keeps records, codecs and collection semantics together:
  `orders.py`, `positions.py`, `targets.py`, `rolls.py`, and `portfolio.py`.
  Query through `book.orders/positions/targets/rolls/portfolios`. Book coordinates
  accounting mutations across owners; their private mutation hooks must not be
  called by components or Controller. All owners share `PersistenceWriter`;
  never give them separate queues. Restoration reads each physical collection
  once, delegates decoding, then repairs dependent projections before trading.
  SyncCoordinator chooses safe broker corrections; PositionState defines their
  field changes and Book persists their accounting effects.
- Persist complete Trade diagnostics and authoritative normalized
  Fill/Execution evidence with actual IB orderId/clientId/permId. Rebind by
  orderId with permId fallback; deduplicate normally by execId. Commission
  updates must persist even with blotter output disabled; blotter is never
  required to reconstruct accounting.
  Broker callbacks must use Book's fill/commission/rebinding methods, not replace
  OrderInfo.trade directly. Book Mongo savers decode UTC dates as aware datetimes;
  injected persistence backends must preserve that contract and raise on failure.
- Physical collections are `orders`, `state`, `blotter`. State identities:
  `position:{source_key}`, `balance:{conId}`, `target:{conId}`, `roll:{series_key}`,
  `portfolio:{portfolio_key}`. Do not add snapshot/decision/lock collections
  without an explicit design change.
- Queue Book mutations through one ordered critical DRAIN queue: order evidence
  precedes derived projections, and the first failed write stops dependent work.
  Source position documents checkpoint applied fill keys atomically with quantity;
  startup finishes only uncheckpointed fills, preserving corrections and resets.
  Rebound order records retain previous broker IDs so startup cannot count their
  obsolete documents twice. Recover completed as well as active order
  evidence. An explicit reset retains fills and persists concrete-target
  cutoffs so old fills do not resurrect cleared direct exposure.
- Both execution modes use maintained `ContractPosition` balances through
  `book.positions.quantity(contract)` and `book.positions.by_contract()`;
  `for_source`/`source_states` expose the separate one-to-one episode view.
  Normal queries never scan fills.
  Update balances with the underlying episode/order/roll mutation, not on read.
  Startup verifies the shared balances against durable accounting records and
  repairs interrupted balance writes. Preserve one-to-one broker corrections;
  do not reconstruct corrected episodes blindly from historical fills.
  Both roll modes share BAG/explicit-leg contribution rules: explicit leg
  evidence replaces the BAG fallback instead of counting it again.
- Reconcile aggregate logical quantity against broker net quantity per concrete
  Contract, including opposing logical one-to-one positions. Use one successful
  `reqPositionsAsync()` snapshot per pass. Cached/fresh disagreement retries
  locally; timeout/unavailable requests use supervisor recovery.
- `controller.position_mismatch_policy` is `fail` by default, with `correct`
  opting into inferred one-to-one corrections. Enforce failure before order
  shortcuts, roll advancement and protection recovery. Failed reconciliation
  skips strategy startup; the trading-disabled latch ends running strategy work
  and persists across reconnects. Keep genuine fill/commission accounting active.
  See the execution guide's position-mismatch and offline-repair contract.
- Defer mismatch decisions while attributed OPEN/CLOSE/TARGET_ADJUSTMENT or
  roll orders work, not merely because a roll plan exists. Applied opt-in
  corrections align quantity and target; flattening clears episode recovery
  inputs but preserves the direction block. Serialize full sync cycles.
- After order/fill and position reconciliation, Controller invokes model-owned
  initial-protection recovery before missing-bracket remediation. See the
  components guide for model inputs and installation limits.
- Target verification waits for OPEN/CLOSE/TARGET_ADJUSTMENT and pending roll
  work, not standing STOP_LOSS/TAKE_PROFIT orders. Superseded checks are abandoned.
- A reset closes all open positions and cancels pending orders. Use `reset`
  consistently for this action. Keep sequencing in `haymaker.controller.reset`:
  `Controller.execute_reset` and `execute_emergency_reset` delegate to `Reset`
  and `EmergencyReset`. Give cancellations a bounded grace period,
  then liquidate even if some cancellations are unconfirmed: getting flat is
  the priority.
  Controller startup clears Book only after every required liquidation was
  accepted and fully filled, pre-reset orders are terminal, and a fresh broker
  snapshot confirms flatness. Otherwise preserve recovery state and the reset flag,
  and prevent trading from being enabled.
  `--nuke` requests an emergency reset: `EmergencyReset` disables trading before
  any broker operation, then bypasses normal submission policies through
  Controller's registered submission path. Persistence checks still apply;
  failures propagate with trading disabled. It neither verifies completion nor
  clears Book state.

## Storage and configuration

- `AsyncDataStore` operations are awaited; successful mutation return means
  backend completion. `QueuedDataSink.enqueue_*` means queue acceptance only.
  Awaited mutations finish an already-started database call before propagating
  cancellation.
- DRAIN is critical: failures/drain timeouts propagate. DISCARD is best effort.
  Book and default Signal dataframe persistence drain; other queued Arctic
  sinks default to DISCARD. Final queue shutdown is bounded and belongs to
  process shutdown, not an ordinary supervisor reconnect.
- Injected stores are fully configured. Naming is chosen at construction and
  is not replaced afterward. `RuntimeContext.frame_store_provider` supports
  custom strategy-owned stores; default component choices are documented in
  the components guide.
- CLI loads configuration once; owning subsystems construct themselves from
  their mappings. Controller one-run actions live under `controller.startup`;
  logging and dataloader `download` remain subsystem groups.
- Live storage has `base_directory`, `mongodb.client`, `mongodb.database`.
  The dataloader guide owns its narrower schema. Default live libraries belong
  under `market_data_store.library` and `signal_persistence.library`; custom
  libraries belong to strategy composition, save frequency to the consumer.
- Bundled profiles enumerate supported settings, pin effective defaults and
  include concise inline comments. Environment variables select profile files,
  not individual setting overrides. Strategy parameters stay in user Python.

## Conversion safety

`scripts/migrate_components_state.py` is standalone, dry-run by default, and
writes only with explicit `--apply` and distinct source/fresh target database
names. Preserve complete evidence, IB identifiers, source/episode attribution,
honest timestamps and deterministic provenance. Refuse ambiguous allocations,
foreign/mixed target data and incompatible in-flight rolls. Test conversion
with fakes only; never run a real migration during code work.
