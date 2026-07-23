# Components Refactor: Agreed Plan

Last updated: 2026-07-23  
Completed discussion phases: agenda items 1 through 15

## Status and maintenance

This file is the authoritative record of decisions already agreed for the
components refactor. `points_for_discussion.md` is the agenda and may contain
proposals that have not been accepted.

The plan is updated only after the user explicitly declares a discussion phase
complete. During later phases, new proposals must be checked against the
decisions below and contradictions must be pointed out before the plan is
changed.

This is an architecture plan only. No implementation is authorized until the
user explicitly asks to start coding.

## General boundaries

1. `Atom` remains a general-purpose event and composition framework. Users may
   pass arbitrary objects between custom Atoms and create Atoms for any
   purpose.

2. Haymaker's pre-built trading components may use structured messages and
   narrower protocols. Those conventions do not become restrictions on the
   general Atom framework.

3. The user-facing built-ins will ultimately be grouped under
   `haymaker.components`. Package membership is based on whether something is
   part of the public component toolbox, not merely on whether it inherits from
   `Atom`.

4. `Atom`, `Pipe`, `Controller`, runtime services, Book, persistence
   infrastructure and contract management remain outside `components`.
   Bracket legs remain inside because users configure them as part of execution
   composition.

5. The package move will be a direct cutover with no compatibility modules.
   It will happen only after the affected public contracts and names are
   settled.

## 1. Portfolio topology

### Portfolio remains the shared decision maker

`Portfolio` remains the name of the account-wide allocator/manager. It is a
shared Atom with access to runtime services, including Book. A Portfolio
implementation may combine any number of signals, strategies, instruments,
constraints and risk models.

The framework must support analog signals, binary signals and the current
one-strategy/one-contract model. It supplies the composition and execution
boundaries; it does not prescribe how a Portfolio converts inputs into target
positions.

### Two supported composition modes

There is one Portfolio abstraction with two ways to compose it.

#### Wrapper mode

```text
Signal path
    -> PortfolioWrapper
    -> dedicated ExecutionModel
    -> Controller
```

The wrapper submits the path's Signal to the shared Portfolio, asks for the
target for that path, converts the answer into a `PositionTarget`, and emits it
to its dedicated ExecutionModel.

Wrapper mode is deliberately lazy and intended for independently managed
positions. Processing one wrapper does not trigger other wrappers. If one
input must immediately resize several instruments, direct mode should be used
instead.

This mode preserves the convenient current arrangement in which every
strategy/input path can configure its own stateful ExecutionModel and bracket
legs.

#### Direct mode

```text
many Signal paths
    -> shared Portfolio
    -> one ExecutionModel

or

many Signal paths
    -> shared Portfolio
    -> ExecutionRouter
    -> selected ExecutionModel
```

The direct Portfolio updates its input state, recomputes the affected targets
or the whole portfolio, and emits zero or more `PositionTarget` objects. It is
the mode for global allocation, multi-instrument decisions and changes in one
input that must immediately affect other positions.

“Direct” describes graph topology, not whether a Portfolio is stateful. A
direct Portfolio is connected to Signal-producing component chains without a
PortfolioWrapper. A Portfolio used through wrappers may still maintain shared
state, and a direct Portfolio may be stateless.

The initial public hierarchy should remain small:

- `AbstractBasePortfolio` defines the common decision boundary and runtime
  access.
- `DirectPortfolio` is a convenience shell implementing
  `onData -> process -> emit zero or more PositionTargets`.
- `FixedSizePortfolio` provides the current simple ready-made allocation
  behaviour and may be used through wrappers or directly where suitable.
- `PortfolioWrapper` is an adapter and is not a Portfolio subclass.

Persistence is optional behaviour rather than a Portfolio topology or a
mandatory `StatefulPortfolio` base class. The hierarchy may be fine-tuned after
the first implementation reveals which hooks are genuinely shared.

### No separate `NoPortfolio`

There will be no built-in `NoPortfolio` mode. A simple `FixedSizePortfolio`
provides the same effective behaviour while keeping one normal path through
the framework.

`FixedSizePortfolio` must allow quantities to differ by input path. Its sizing
extension point may use `source_key`, Contract or any other Signal field.

### Portfolio source registration

Source registration is optional and is configured in `Portfolio.__init__`.

- Wrapper mode does not require registration.
- A direct Portfolio that needs a known input universe may declare its sources
  at construction time.
- A direct Portfolio that reacts immediately or manages dynamic membership may
  omit registration and own that policy itself.

Conceptually:

```text
sources=None
    registration disabled; no built-in membership completeness check

sources=[...]
    declared membership enabled; Portfolio knows the expected source set
```

Registration is not discovered from the order in which runtime Signals arrive.
Doing so could cause Portfolio to treat a partially started source set as
complete.

### Optional observation synchronization

The built-in `Signal` protocol standardizes optional `as_of` metadata:

```text
source_key: str
as_of: datetime | None
```

`as_of` is the effective time of the market observation, not its arrival time.
When present it must be timezone-aware. Base Portfolio does not interpret it
or impose batching.

Concrete Portfolio implementations own recomputation policy:

- independent and tick-driven inputs may recompute immediately;
- synchronized portfolios may wait for every required source for an `as_of`;
- a Portfolio may reject missing `as_of` values;
- timeout, duplicate and late-arrival behaviour belongs to that
  implementation.

If repeated implementations reveal a common need, an optional batching
coordinator may be added later. Immediate processing requires no coordinator,
and batching will not become mandatory Base Portfolio behaviour.

## 2. Signal, target and position identities

### `Signal`

`Signal` is the structured input used by Haymaker's pre-built trading
components. It does not replace arbitrary Atom messages.

Its common information is:

```text
source_key
Contract
value
form
as_of, optional
created_at
metadata, optional
```

`source_key` is an opaque, stable and serializable identifier for one logical
input stream. A typical value may identify a strategy, parameter variant,
timeframe and instrument. Haymaker does not parse it.

`created_at` is a timezone-aware processing timestamp recording when the
Signal was created. It is distinct from `as_of`, which identifies the market
observation on which the Signal is based.

Every standard Signal declares one of two forms:

```text
SignalForm.STATE
    the value is the source's desired state after this observation

SignalForm.EVENT
    an event with this value occurred at this observation
```

The distinction is semantic and mandatory. Repeating a state value reaffirms
the same desired state; repeating a non-zero event represents another event.
For a state Signal, zero means a flat or neutral desired state. For an event
Signal, zero means that the source was evaluated but generated no event. That
zero event may be required to establish source completeness in a synchronized
Portfolio batch.

This vocabulary is shared with the research package. A research `signal` is a
series of state Signals, while a research `blip` is the concise existing name
for a series of event Signals. The vectorized research representation and the
live runtime representation remain different.

Portfolio implementations declare or document which forms they accept. A
state-aggregating direct Portfolio will commonly retain the latest state Signal
per `source_key`; the current one-to-one event-driven path may consume event
Signals. A component receiving a Signal form it cannot interpret must fail
clearly rather than silently treating state and event semantics as
interchangeable.

The concrete IB `Contract` remains the instrument at the execution boundary.
No separate mandatory `instrument_id` is added. Portfolio implementations may
maintain any logical instrument, sector, currency or allocation groupings they
need internally.

Framework-provided Wrapper and FixedSizePortfolio paths preserve the Signal's
Contract when creating PositionTarget. A custom direct Portfolio remains free
to select another Contract as part of its own allocation logic, but Haymaker
does not add a separate contract-remapping mechanism.

The earlier names `signal_key` and `position_key` are not separate framework
identities. `source_key` identifies the input; there is no universal
`position_key`.

### `PositionTarget`

Every built-in Portfolio path emits the same immutable `PositionTarget` to an
ExecutionModel. It is an absolute setpoint, not an instruction to place a
particular trade. Its common information is:

```text
Contract
target_quantity
created_at
optional PositionIntent
optional source/decision context
optional metadata
```

Source/decision context is optional in the general direct-mode message.
Wrapper mode is stricter: PortfolioWrapper must attach its `source_key`, and
the dedicated ExecutionModel must preserve that `source_key` together with its
`position_id` on every opening, closing, bracket and roll order. This is the
required basis for reconstructing one-to-one strategy episodes and P&L.

It does not contain a captured current quantity or a proposed delta. Those
values can become stale while an earlier target is still being executed and
could cause duplicate or contradictory orders.

`created_at` is the timezone-aware time of the Portfolio decision. Together
with `Signal.created_at`, the locally recorded order-submission time and broker
fill execution times, it provides explicit latency checkpoints without
generic Atom mutation.

For each execution scope, an ExecutionModel owns convergence from accounted
broker state to the latest target. It considers relevant working orders,
serializes target changes for that scope, and decides whether to submit,
modify or cancel orders. A newer target supersedes an older target even while
execution is in progress. Order and fill events reconcile against the latest
target rather than the target that existed when an order was submitted.

The execution scope belongs to the concrete model. It may be one dedicated
logical position in wrapper mode or a Contract managed by a shared direct-mode
model. Controller remains the broker safety, submission and aggregate
reconciliation boundary.

Target-only execution does not solve incomplete Portfolio input views.
Registration, `as_of` batching and recomputation timing remain Portfolio
policies.

### Optional `PositionIntent`

There is no generic `PositionTransition` enum. In particular,
`INCREASE`/`REDUCE` add no required downstream instruction and are ambiguous
for signed short quantities. Exposure changes may be derived for reporting.

The built-in one-to-one path may attach:

```text
PositionIntent.OPEN
PositionIntent.CLOSE
PositionIntent.REVERSE
```

Intent describes the transition from the previous desired target to the new
desired target:

```text
zero -> non-zero       OPEN
non-zero -> zero       CLOSE
sign change            REVERSE
```

The numeric target is authoritative; intent is an optional assertion and
narrow lifecycle semantic. Generic direct Portfolio and ExecutionModel
implementations may omit it.

Intent is runtime information on each `PositionTarget`, not Atom
initialization data. A SignalModel does not know or satisfy downstream
ExecutionModel requirements. Portfolio or PortfolioWrapper attaches intent
when producing the target. An intent-dependent ExecutionModel, including the
current event-driven model, must reject a missing intent before changing its
execution state or submitting an order.

No general producer/consumer capability-negotiation mechanism is introduced
at this stage. Concrete ExecutionModels document and validate their own
requirements on every target. A future class-level capability declaration may
aid startup introspection, but runtime validation would remain authoritative.

### Order side and role

Buy/sell side remains the broker order action and is derived by the
ExecutionModel from the signed execution requirement.

Order role belongs to an actual order created downstream of the
`PositionTarget`, at the ExecutionModel-to-Controller boundary. It is persisted
as a string so custom ExecutionModels may introduce roles. Haymaker provides a
`StandardOrderRole` string enum for roles with framework-recognized meaning,
initially including:

```text
OPEN
CLOSE
TARGET_ADJUSTMENT
STOP_LOSS
TAKE_PROFIT
ROLL
LIQUIDATION
RECONCILIATION
MANUAL
UNKNOWN
```

There is no `REVERSE`, `INCREASE` or `REDUCE` order role. A logical reversal
is normally attributed as a close followed by an open; generic net-position
changes use `TARGET_ADJUSTMENT`.

Book and Controller preserve custom roles for audit but apply framework
behaviour only to recognized roles. Optional one-to-one re-entry state is
updated from authoritative live or replayed fills, not from ExecutionModel
callbacks.

There is no closed framework `OrderType` enum.

### Order configuration

Order-type selection is not part of `PositionTarget`. Order construction belongs
to the selected ExecutionModel.

Configuration precedence is:

```text
built-in fallback
    < global order configuration
    < ExecutionModel constructor parameters
```

ExecutionModel parameters therefore override global defaults. Models may also
calculate prices, quantities, OCA fields and other necessary broker fields as
part of their own implementation.

### `position_id`

`position_id` is not a universal broker-position identity and is not required
on every `PositionTarget`.

It is optional execution-model-owned accounting for a logical, independently
managed position episode. The primary built-in use is the wrapper/dedicated
bracket model:

```text
flat
    -> open P1
    -> increase or reduce P1 while remaining on the same side
    -> close P1

flat
    -> later open P2
```

A reversal closes the old episode and opens a new one. Creating a new
`PositionTarget` does not itself create a new `position_id`.

The ExecutionModel creates and maintains `position_id` because it knows
whether several requests and broker orders belong to one protected position
episode. Portfolio does not create or manage it.

A direct net Portfolio or ExecutionModel may have no logical position episodes
and therefore no `position_id`.

### Broker positions and logical positions

The broker exposes only the net position per concrete Contract. Controller
reconciliation remains aggregate by Contract.

Independently protected logical positions in the same Contract remain
supported, including opposing positions whose broker net is zero. Their orders
and fills must be attributed locally, while the sum of local quantities is
compared with the broker quantity.

No mandatory `ManagedPosition` domain object is introduced merely to represent
this. A specialized ExecutionModel and Book state may track the logical
episode when required.

### Identifier policy

New generated identifiers are added only when they solve a concrete problem.

- IB `orderId` remains the primary operational lookup for incoming events.
- `permId` is retained and used where helpful, including restart matching.
- `execId` deduplicates fills.
- All other useful IB identifiers and the live `ibi.Trade` are retained for
  diagnosis and audit.
- No separate mandatory framework order ID is added.
- A decision or execution correlation ID remains optional and is introduced
  only by features that actually require it.
- Internal Portfolio allocation keys are implementation details, not common
  protocol fields.

## Execution boundary constraints agreed while resolving items 1 through 6

These constraints affect later agenda items but do not close their remaining
design details.

### ExecutionModel

- ExecutionModels remain Atoms and may be stateful.
- They receive new work through `onData`, not `onStart`.
- They are not implicitly bound to strategy data copied during startup.
- Required target facts arrive in `PositionTarget`.
- They own convergence to the latest target, including relevant in-flight
  order handling.
- Concrete models validate any optional protocol information they require
  before taking execution action.
- They may read durable execution and order facts from Book.
- A dedicated wrapper model, a shared multi-contract model and a model that
  manages several execution sessions are all valid implementations.
- ExecutionModels create, modify and cancel orders through Controller.
- Broker `Trade` events are transient and are not replayed by the framework.
  A model must not assume that it will observe every event across a
  disconnection.
- On startup, a model reconnects its callbacks only to still-active Trades
  that Controller has already reconciled and rebound.
- A model whose correctness depends on an effect of a Trade that completed
  offline must reconstruct that effect from authoritative Book state. Effects
  explicitly documented as best-effort may be missed.

The current event-driven bracket implementation intentionally waits for a
complete entry fill before creating stop-loss/take-profit orders; partial fills
do not trigger bracket creation. It is also acceptable for this model to miss
bracket creation when the complete fill occurs while disconnected. This is an
explicit operational exception, not a replay guarantee or a general precedent
for critical execution state. Reversal convergence, accounting and any future
critical event-driven behaviour must remain recoverable from Controller/Book
state rather than depending solely on receipt of one transient event.

### ExecutionRouter

Router is optional. With one ExecutionModel, Portfolio connects directly to it
using an ordinary Pipe.

When used, `ExecutionRouter` is an Atom configured with already constructed
and user-configured ExecutionModel instances plus ordered routing rules. Router
does not construct or configure the models. Rules are evaluated in user-defined
order and the first match wins. A rule may inspect Contract, symbol, security
type, exchange or an arbitrary predicate over the request. An explicit default
route is recommended; an unmatched request must not place an order.

For the initial implementation, routing configuration is fixed for the
programme lifetime. Once an ExecutionModel owns active work for an execution
scope, later targets for that scope continue to use it until the scope is flat
and has no working orders. Dynamic rule replacement and live hand-off between
ExecutionModels are outside this refactor.

Route selection belongs to Router, not Portfolio and not a mandatory
`PositionTarget.execution_route` field.

Router invokes only the selected ExecutionModel's `onData` method. The models
are not ordinary broadcast targets of Router, and the general Atom connection
model is not changed to support selective emission.

Router also invokes every configured ExecutionModel's `onStart` once for each
supervised workload generation. The instances have already been constructed
and configured by the user; this lifecycle call exists so a model can restore
its own lightweight runtime state and reconnect callbacks to still-active
`ibi.Trade` objects after Controller has reconciled and rebound them. It does
not make ExecutionModel responsible for back-accounting fills that completed
while the system was offline; Controller and Book own that recovery.

### Controller and `ibi.Trade`

- Controller remains an Atom and the safety, broker-submission,
  order-registration, fill-accounting and reconciliation boundary.
- ExecutionModels continue to receive the live `ibi.Trade` returned by
  Controller and use its events.
- `ibi.Trade` is transient across IB restarts but must not be replaced or
  hidden behind a competing normalized object.
- Recreated Trades are rebound to the existing order information during
  recovery.
- The existing operational `orderId` lookup is preserved, with `permId` and
  the other IB identifiers retained as complementary evidence.
- The common boundary does not require new `OrderCommand` or `OrderHandle`
  wrapper types.

### Futures rolling

This refactor will not redesign futures rolling into the general
Portfolio/PositionTarget/ExecutionRouter path. The existing Controller-owned
rolling operation remains a specialized contract-migration path and is changed
only as required to work with Book, the new names and the agreed accounting
fields while preserving current behaviour.

A broader roll architecture, including configurable roll execution policies,
may be designed later. It must not expand this already large refactor.

## Atom lifecycle and message hooks

### `onStart`

Generic Atom startup retains the current flexible model. `onStart` receives an
arbitrary user payload plus the immediate upstream Atom, performs base contract
change processing, and propagates startup downstream.

A user Atom may inspect, mutate or enrich the payload before calling
`super().onStart(...)`. It may intentionally suppress propagation by not
calling the base method. The payload is not replaced by a mandatory
`StartInfo` object.

`onStart` remains synchronous and is intended for lightweight initialization
such as setting attributes, inspecting an upstream component and connecting
callbacks. Haymaker will not introduce awaited graph startup in this refactor.
If a custom Atom launches asynchronous initialization, it owns readiness and
buffering until that work completes.

There is no separate `on_source_start` hook. Aggregators and other components
that need their immediate upstream component inspect the existing source
argument in `onStart`.

Base Atom does not deduplicate startup calls. Different upstream paths may
legitimately supply different initialization payloads. Shared components such
as Portfolio, Router and shared ExecutionModels own any merging, readiness or
idempotence policy they require.

`startEvent` is a lifecycle mechanism, not a general runtime-notification
channel. Framework components do not re-emit it after a futures adjustment or
other ordinary data event because doing so could repeat downstream
initialization.

Framework-owned process and supervised-workload generation metadata, when
needed, belongs in `RuntimeContext`; it does not replace or reserve the user
startup payload.

The old generic `Atom.startup` flag is obsolete and will be removed. It is not
used by production components and is unrelated to the
`controller.startup` configuration group.

Generic Atom will also stop automatically acquiring a trading-specific
`strategy` attribute from startup data. Built-in trading messages carry
`source_key` explicitly, avoiding repeated per-component configuration while
keeping trading identity out of the general Atom abstraction. Specialized
components may still define or consume a strategy identity where their own
contract requires it.

### `onData`

The generic `onData` contract accepts any Python object. Base Atom does not
assume a dictionary, mutate the input, append timestamps or automatically emit
an output.

A concrete Atom may mutate the received object, create another object, retain
it, or emit zero, one or many messages. Emission through `dataEvent` remains
explicit. Filters, sinks, batching components and components that suppress a
signal therefore require no special base behaviour. A data-consuming Atom
that has not implemented `onData` should fail clearly with
`NotImplementedError`.

Users remain free to implement cumulative mutable-dictionary pipelines and
selectively append their own timing or audit information. That style is
permitted but is not imposed by Atom.

Haymaker's built-in trading components use structured messages at their
agreed boundaries. Optional metadata must preserve upstream values required
by downstream implementations. In particular, the wrapper-based
EventDrivenExecutionModel must continue to receive inputs such as its
configured volatility field for bracket construction.

The old class-named `*_ts` fields added by `super().onData()` will be removed.
No production logic consumes them. Built-in latency auditing instead uses
explicit semantic timestamps:

```text
Signal.created_at
PositionTarget.created_at
order submitted_at
broker fill execution time
```

`Signal.as_of` remains the effective time of the market observation and must
not be confused with processing latency.

### Fan-out and mutable messages

No new fork mode or automatic copying policy is introduced. Connecting one
Atom to several targets sends the same object reference to every target.

Messages crossing a fan-out are therefore treated as read-only by convention.
A branch that intends to mutate a message must copy it first and emit or retain
the copy within that branch. The correct operation is type-specific:
`dict.copy()` is shallow, nested structures may require a deeper or custom
copy, and dataframes should use their explicit copy operation. The framework
cannot choose a universally safe or efficient policy.

This rule will be stated in the public `Atom.connect` and composition
documentation. Standard built-in boundary messages such as `Signal` and
`PositionTarget` are immutable, which avoids the problem for those paths.
Explicit fork/copying APIs may be reconsidered only after concrete use cases
show a repeated policy that the framework can implement safely.

### Compatibility validation

Compatibility validation is layered and owned by the consuming component. The
general Atom graph remains open to arbitrary user messages and does not gain a
universal input/output type system or capability-negotiation protocol.

Validation occurs at the earliest stage where the answer is reliable:

| Stage | Responsibility |
|---|---|
| constructor | validate the component's own configuration |
| connection | validate structural compatibility with the immediate source |
| `onData` | validate the actual message and conditional requirements |

Base Atom provides a no-op:

```python
def validate_source(self, source: Atom) -> None:
    ...
```

`Atom.connect()` calls `target.validate_source(source)` synchronously before
wiring events. When several targets are supplied in one `connect()` call, all
are validated before any connection is made so one failure cannot leave that
branch partially connected. No graph-wide transactional construction or
rollback mechanism is added.

Only consumers with genuine structural requirements override the hook. The
current aggregator checks should use real classes, ABCs or a minimal runtime
protocol rather than class-name strings, so supported subclasses are not
rejected merely because their names differ. Generic custom Atoms retain the
no-op default.

Facts unavailable during graph construction are not treated as connection
compatibility. The consuming component validates them when they are first
used, normally in `onData`, and raises before changing state or causing an
external action.

Per-message validation remains authoritative. Portfolio validates accepted
Signal forms; an intent-dependent ExecutionModel validates `PositionIntent`;
Router validates that a route exists; and every consumer checks conditional
information before changing state or causing an external action. An invalid
message fails closed with a clear error rather than being silently accepted or
acted upon.

Type annotations and documentation remain advisory. Haymaker does not add
graph-wide type inference, automatic compatibility from annotations or
mandatory input/output declarations. Optional capability declarations may be
reconsidered only if concrete repeated use cases justify them; runtime message
validation would still remain authoritative.

### `onFeedback`

`onFeedback` and the automatic reverse connection remain part of the general
Atom protocol. They provide a valid way for custom linear chains, including
chains of SignalModels, to pass information upstream.

Feedback through branches or shared components may merge or broadcast and is
not inherently correlated. Users must add addressing or correlation when
targeted feedback is required. Portfolio, Router and execution accounting
must not rely on implicit reverse traversal for authoritative execution
feedback.

## Runtime ownership

The supported live deployment owns one installed `RuntimeContext` and one
composed live component graph per process. `Atom.runtime`,
`Streamer.instances` and the shared Portfolio are process-scoped within that
model. The Portfolio singleton is intentional account-wide coordination, not
accidental component state.

Independent live applications, accounts or separately reloadable strategy
graphs in one interpreter are not supported. Supervised workload recovery does
not create a second independent runtime ownership domain. Tests install
isolated runtime services through the project fixtures rather than relying on
same-process production application reuse.

No global-runtime refactor is part of this work. If multi-runtime or
multi-account operation in one process becomes a requirement, runtime
injection and process-global registries must be reconsidered explicitly.

## Signal models

### Purpose and names

The existing generic `Block` abstraction will be replaced by the narrower
`SignalModel` component. A generic computational Block adds no meaningful
contract beyond `Atom`; users who need arbitrary indicators, transformations
or other processing can implement ordinary Atoms and compose them freely.

`SignalModel` has one framework meaning: it produces standard `Signal`
messages. It owns a `source_key` and a Contract because every standard Signal
identifies its logical source and the Contract to which the signal applies.

`PandasSignalModel` is the built-in dataframe-oriented specialization replacing
the current `AbstractDfBlock`. Its calculation returns a dataframe containing
the source data, intermediate calculations and resulting signal information,
and the model emits a `Signal` constructed from the applicable latest row.
Dataframe use is an implementation choice rather than a SignalModel
requirement.

Users may implement incremental, Polars-based or other SignalModels as long as
they emit the standard Signal contract. No separate
`IncrementalSignalModel` abstraction will be introduced until multiple
implementations reveal genuinely shared behaviour.

Where practical, research and live SignalModels should reuse pure calculation
functions. Research may consume the complete returned signal series, while a
live PandasSignalModel packages the current point as a Signal. Batch and
incremental implementations should be testable for equivalent outputs over
the same observations.

Signal generation does not determine execution timing. Research's next-bar
conversion and live Portfolio/ExecutionModel timing remain downstream policy.
The shared Signal value is not restricted to `{-1, 0, 1}`; binary constraints
belong to binary models and processors.

### Calculation-data audit

PandasSignalModel dataframe persistence is optional audit/debug evidence. It
does not participate in signal generation, runtime recovery or Portfolio
state. The saved artifact is the complete calculated dataframe needed to
inspect the source data and intermediate values that produced a questionable
Signal, not merely the latest emitted row.

The framework-provided policy retains a separate immutable Arctic symbol for
each SignalModel run and upstream ACTIVE price-series generation:

```text
{source_key}_{ACTIVE.localSymbol}_{run_started_at}
```

`run_started_at` is an explicit timezone-aware UTC process/component-graph
value, rendered with sufficient precision and collision protection. It is not
an implicit module-import timestamp. An ordinary supervised workload restart
continues the same audit symbol because the component graph and its in-memory
aggregation state survive. A new process/component graph creates a new audit
symbol even when ACTIVE has not changed.

Persistence identity follows the adjusted input series supplied upstream:

- the ACTIVE market-data Contract supplies `localSymbol`;
- SignalModel derives ACTIVE from its own
  `contract_selector.active_contract`;
- the SignalModel's transaction Contract and `Atom.which_contract` do not
  determine the audit symbol;
- a NEXT-only change does not rotate the symbol;
- after ACTIVE changes, a new symbol is created only after back-adjustment
  succeeds;
- the first save to that symbol writes the complete adjusted dataframe;
- later saves append normally;
- the preceding symbol remains unchanged;
- failed adjustment does not create a new symbol;
- a cold start with insufficient adjustment history must report or handle that
  condition explicitly.

For futures, downstream dataframe history may be back-adjusted without a
separate notification. SignalModel treats the first dataframe emitted under a
changed ACTIVE Contract as the new adjusted input generation; it does not
receive a second `onStart` or a standard adjustment event. DfAggregator may
expose adjustment details in a later design if a concrete consumer requires
them, but adjustment value, method and magnitude are not part of the current
component protocol.

Metadata records at least `source_key`, run start, ACTIVE Contract, observation
range and run status. The exact audit symbol is logged when created and may be
included as an optional calculation-data reference in Signal metadata. A
simple lookup helper may select the latest matching run before
`Signal.created_at` and slice it through `Signal.as_of`; no mandatory run
catalogue or observation journal is introduced.

Normal append is valid for causal models whose previously emitted calculated
rows remain immutable within one run and series generation. A custom model
that revises historical calculated rows must supply a snapshot/version
persistence policy capable of preserving those revisions.

The framework audit sink uses a dedicated ordered `DRAIN` queue rather than
the shared best-effort `DISCARD` queue. This keeps the trading path
non-blocking while normal shutdown waits for queued writes and reports
failures. A user who requires persistence completion before Signal emission
may inject an awaited sink; a user may also disable persistence or provide an
entirely custom policy.

## Book and persistence constraints agreed while resolving items 1 through 6

`StateMachine` will be renamed `Book`. It is the central accounting,
information and persistence facade available to Atoms.

Book owns database recovery and persistence. Portfolio, signal processors and
ExecutionModels use Book APIs rather than constructing their own Mongo savers.
Book does not perform Portfolio calculations or broker API work.

The intended minimum physical collections are:

1. `orders`: broker orders, live/recreated `ibi.Trade` information, IB
   identifiers, roles, fills including broker execution times, local
   submission time and source/position correlation when the execution mode
   supplies it; wrapper-mode orders require both `source_key` and the applicable
   `position_id`;
2. `state`: compact current recovery state, stored per entity rather than as
   one whole-system strategy snapshot;
3. `blotter`: the continuing performance/accounting record populated from
   completed broker executions and commission reports.

An append-only `decisions` collection remains optional for explaining
Portfolio decisions and no-trade outcomes. It is not required for basic
recovery. Separate collections for signals, executions and locks are not
required initially.

### Historical blotter continuity

The existing Mongo blotter collection contains approximately two years of
strategy testing and is a required historical input to this refactor. Blotter
is not removed by this work. Its existing records are converted to the new
language and runtime collection so new records can continue to accumulate in
the same schema after cutover.

Haymaker will first implement and validate the new schema and runtime without a
legacy compatibility path. At the end of the refactor, a small one-time
conversion script will read the old database and populate a new database
entirely in the new schema. The application will then switch to that new
database; normal runtime code will not dual-read legacy and new records.

Blotter history is the primary migration requirement. The converter will
translate its records into the new blotter conventions. At minimum:

```text
legacy strategy       -> source_key
legacy position_id    -> position_id, unchanged
legacy action         -> StandardOrderRole where the meaning is known
order_id / perm_id    -> orderId / permId
side                  -> broker BUY/SELL action
fills                 -> execution facts
commission            -> commission facts
realizedPNL           -> realized P&L evidence
```

The original legacy action is retained even when it maps to a standard role.
Unrecognized or custom actions remain custom role strings rather than being
forced into an incorrect standard value. Missing `position_id` values are
preserved as missing; migration must not invent a logical episode when the
source record cannot establish one.

Legacy timestamps retain their actual meanings. In particular, blotter
`sys_time` records report creation after commission processing and must not be
relabeled as order submission time. Submission time may be recovered from
stored Trade evidence only when that derivation is reliable.

Current StateMachine data may also be converted where it has continuing value:

- the latest effective per-strategy state may seed new per-source `state`
  records;
- active or otherwise necessary order records may seed new `orders` records;
- StateMachine orders and converted blotter records must be cross-checked by
  their IB identifiers and execution facts;
- repetitive historical whole-system strategy snapshots need not be migrated
  merely because they exist.

Migration is a controlled cutover:

- the existing database remains unchanged as the source archive and rollback
  reference;
- the converter writes to a fresh target database using only the final new
  schema;
- migrated records retain schema/provenance information sufficient to trace
  them to the legacy source;
- the conversion is deterministic and safely repeatable against a new empty
  target;
- counts, source/position grouping, commissions, realized P&L, active state and
  representative position episodes are reconciled before cutover;
- after validation, configuration is changed to the new database and only the
  new Book/query APIs and new blotter schema are used.

The conversion script will be written only after the new order/state schemas
and their uniqueness and index policies are final.

A stateful direct Portfolio may persist its own normalized recovery state
through Book. Raw received Signals are not the recovery contract. A Portfolio
may retain latest STATE values, accumulate EVENT inputs into its own state, or
use another interpretation entirely. If replayed EVENT inputs require
deduplication, the concrete state also retains whatever source cursor that
implementation needs.

This persistence is required only when a Portfolio cannot reconstruct its
state after restart, for example because sources emit only on change. Records
are scoped by Portfolio identity. Registered Portfolios restore only their
currently configured source set; unregistered/dynamic Portfolio
implementations own their membership and stale-input policy. Wrapper and
stateless immediate implementations need not persist Portfolio state.

For current one-to-one operation, Book must provide a compact, queryable
execution state showing at least:

```text
source_key
signed quantity
active Contract
position_id, when used
re-entry state, when used
updated time
```

This replaces the need to inspect large arbitrary strategy snapshots merely to
determine which strategy/input is in the market and in which direction.

### Optional re-entry blocking

The current `lock` concept is specialized one-to-one behaviour. Its clearer
meaning is a persisted re-entry-blocked direction.

- It is optional; a pipeline may have no lock-aware signal processor.
- It is not ExecutionModel callback state.
- Controller's live and offline fill processing supplies authoritative fill
  events to Book.
- A stop-loss that flattens the logical position blocks re-entry in the
  previous direction.
- An ordinary close or take-profit that flattens it clears the block.
- Partial exits do not apply the flat-position rule prematurely.
- Signal processors may consult or ignore this state.

## Components package

The user-facing package move remains a direct cutover with no compatibility
layer. With the agreed SignalModel terminology, its logical shape is:

```text
haymaker/components/
    messages.py
    streamers.py
    aggregators.py
    signals.py
    signal_models.py
    portfolio.py
    execution_router.py
    execution_models.py
    bracket_legs.py
```

`messages.py` contains the standard immutable boundary messages and their
enums, including Signal, SignalForm, PositionTarget, PositionIntent and
StandardOrderRole. `signals.py` contains built-in signal processors.
`signal_models.py` contains `SignalModel`, `PandasSignalModel` and concrete
signal-producing models. `execution_router.py` contains ExecutionRouter and its
rule helpers. There is no generic public `blocks.py` component module.

Boundary metadata remains an extension mechanism, not an alternative
unstructured protocol. Framework components do not automatically merge or
persist arbitrary metadata wholesale. Information that becomes stable and
framework-relevant is promoted to a named field.

The final move will update all internal imports, tests, public documentation
and the user's strategy modules in one change. `Atom`, `Pipe`, Controller,
Book, runtime services, persistence and contract management remain outside
`components`.

## Implementation and migration order

After the remaining public APIs are agreed, implementation proceeds in this
order:

1. Define and implement the Book schema and Controller boundary.
2. Implement the boundary messages, Portfolio modes, ExecutionRouter and
   ExecutionModels.
3. Adapt futures rolling, re-entry accounting and SignalModel audit
   persistence only as far as required by the agreed design.
4. Perform one direct `components` package cutover, updating internal imports,
   tests, documentation and the user's strategy modules without compatibility
   modules.
5. Convert and reconcile the legacy StateMachine and blotter data into a fresh
   database using the final schemas, then switch configuration after validation.

The separate Controller position-snapshot race documented in `issue_1.md` is
not part of this refactor and does not gate this implementation order. It is
being handled independently.

## Superseded proposals

The following earlier working ideas are not part of the agreed design:

| Superseded idea | Agreed direction |
|---|---|
| A separate `NoPortfolio` Atom | Use a simple `FixedSizePortfolio` |
| Wrappers must react to every global target change | Wrapper mode is lazy; use direct mode for immediate global consistency |
| Source registration is mandatory | Registration is optional and configured by Portfolio |
| `signal_key` plus universal `position_key` | One input `source_key`; no universal `position_key` |
| Portfolio creates a `position_id` per request | ExecutionModel optionally owns one logical position episode |
| Every direct/net position has a `position_id` | Direct net execution may use none |
| Rename `StateMachine` to `TradingState` | Rename it to `Book` |
| Replace `ibi.Trade` with `OrderRecord`/`OrderHandle` | Preserve and expose the live Trade; persist and rebind it |
| Add a mandatory framework order record ID | Preserve operational `orderId` and all useful IB IDs |
| Portfolio chooses an `execution_route` | Router owns ordered route selection |
| Per-request order-type overrides | Global defaults overridden by model parameters |
| ExecutionModels should be stateless | They may be stateful but must not depend on sticky startup strategy data |
| ExecutionModel callbacks own re-entry locking | Book updates optional re-entry state from authoritative fills |
| `TradeRequest` carries current, target and delta quantities | `PositionTarget` carries the absolute target; ExecutionModel owns convergence |
| Generic `PositionTransition` including increase/reduce | Optional one-to-one `PositionIntent` with open/close/reverse only |
| Portfolio supplies an order role | ExecutionModel assigns a standard or custom role to each actual order |
| Blocks or startup wiring guarantee required intent | Portfolio/Wrapper supplies per-target intent and the consuming model validates it |
| Mandatory immutable `StartInfo` replaces startup data | Preserve arbitrary user startup payloads; keep framework run metadata in RuntimeContext |
| Separate `on_source_start` hook | Components inspect the existing immediate-source argument in `onStart` |
| Awaited, deduplicated graph startup | Keep synchronous propagation; shared components own idempotence and multi-source policy |
| Re-emit `startEvent` to announce futures back-adjustment | Treat startup as lifecycle only; futures dataframe history may change without a separate notification |
| Base `onData` appends class timestamps | Base `onData` does not mutate messages; built-ins use semantic latency timestamps |
| Remove the unused reverse event path | Retain `onFeedback`, with explicit caution around branches and shared components |
| Keep a generic public `Block` abstraction | Use ordinary Atoms for arbitrary processing and `SignalModel` for standard Signal production |
| Call persistent desired state and sparse events an undifferentiated Signal | Require `SignalForm.STATE` or `SignalForm.EVENT`; research `signal` and `blip` map to those forms |
| Name calculation audit data from the SignalModel transaction Contract or root symbol | Name each run from the upstream ACTIVE `localSymbol` and rotate only after successful adjusted-series generation |
| Use one shared best-effort queue for SignalModel audit data | Use an optional dedicated ordered `DRAIN` audit sink, with awaited and custom policies available |
| Add a mandatory run catalogue or observation journal | Keep run-scoped dataframe symbols, log/reference the exact symbol and provide a simple lookup helper |
| Add automatic shallow/deep/custom fork modes | Document that fan-out shares one reference and require a mutating branch to copy explicitly |
| Refactor process-global runtime state for hypothetical multi-runtime use | Document and retain the supported one-runtime, one-live-graph-per-process ownership model |
| Add a universal typed graph or capability-negotiation system | Use an optional target-owned `validate_source()` hook plus runtime validation by concrete consumers |
| Keep aggregator compatibility as class-name strings | Validate against real types, ABCs or a minimal runtime protocol so supported subclasses remain compatible |

## Still open

Agenda items 1 through 15 are closed. In particular, later discussion must
still settle or confirm:

- the exact public class and dataclass APIs;
- the exact ExecutionModel/Controller method and recovery APIs;
- the final Book schema and optional decision audit;
- the exact Portfolio-state recovery contract;
- the exact calculation-audit run and adjusted-series generation contract.

Any later proposal may refine an open detail, but it must not silently reverse
the closed decisions above.
