1. **Make `Portfolio` the convergence point**

The current wrapper model in [portfolio.py](/home/tomek/haymaker/haymaker/portfolio.py:60) preserves the appearance of a linear chain, but that stops fitting once Portfolio must:

- combine many strategies and instruments;
- cap aggregate exposure;
- resize positions using volatility and sector limits;
- react to one proposal by changing several instruments;
- route trades through different execution approaches.

I recommend this topology:

```text
strategy/instrument chains
          │
          ▼
     shared Portfolio
          │  zero, one, or many execution requests
          ▼
    ExecutionRouter
       │      │
       ▼      ▼
 execution models
          │
          ▼
 Controller / broker
          │
          └── execution reports back to Portfolio
```

Each chain can connect directly to the singleton `Portfolio`; the chain does conceptually end there, but the overall graph continues through the shared Portfolio output.

Portfolio would maintain a proposal book keyed approximately by:

```text
(source/strategy, logical instrument)
```

A change to any entry causes it to recompute the affected portfolio—or the whole portfolio if necessary.

Examples:

- Analog strategy: `strategy_a, ES → 0.6`
- Binary strategies: `strategy_a, ES → +1`, `strategy_b, ES → -1`
- Current model: `strategy_a, ES → target +1, stated OPEN, bracket policy`

The Portfolio implementation decides what those proposals mean. The Atom framework does not.

Alternatives:

- Keep wrappers: simplest, but awkward when one input causes multiple global trades.
- Make Portfolio a non-Atom runtime service: defensible, but loses useful Atom lifecycle/event infrastructure.
- Wrapper input plus separate global Portfolio output: workable, but introduces two overlapping paths and more correlation problems.

The shared Portfolio Atom is the cleanest option.

2. **Separate proposal ownership, portfolio exposure and managed positions**

The current state model equates “strategy” with several distinct things:

- signal source;
- position owner;
- broker-position attribution;
- execution-model state;
- stop-loss owner.

That works for one strategy owning one future position, but not for aggregated portfolios. [OrderInfo](/home/tomek/haymaker/haymaker/state_machine.py:30) even requires every order to belong to one strategy.

I think the new model needs these separate identities:

- `source_id`: who produced the proposal;
- `instrument_id`: logical portfolio/risk instrument, such as `ES`;
- concrete contract: the actual tradable contract, such as `ESU6`;
- `position_id`: optional independently managed position/protection unit;
- `decision_id`: one Portfolio calculation;
- `execution_id`: one attempt to implement a decision;
- broker order IDs: potentially several per execution.

This accommodates both:

- aggregated ES exposure managed as one net position;
- current strategy-specific protected positions, each with its own `position_id` and stop policy.

There is an unavoidable policy decision here: multiple independently stopped “positions” in the same contract are only logical allocations—the broker holds one net position. Controller reconciliation must therefore compare the sum of managed positions with the broker net position.

3. **Standardize the boundary after Portfolio, not the entire Atom framework**

I agree with your distinction:

- Atom infrastructure should accept arbitrary objects.
- Built-in components may have specific contracts.
- Inputs accepted by a Portfolio implementation need not be globally standardized.
- Output from Portfolio should be standardized because execution, validation, persistence and reconciliation all consume it.

A normalized execution request should contain approximately:

```text
decision_id
execution_request_id
source/contributor references
instrument_id
concrete contract or legs
position_id/owner_id, if applicable

current_quantity
target_quantity
delta_quantity

stated_transition, optional
computed_transition

order_role
execution_policy
policy_parameters
extensions
```

Both `target_quantity` and `delta_quantity` are worthwhile:

- target states the intended final truth;
- delta states what should be traded now;
- Controller can recalculate delta against current positions and pending orders before submitting anything.

For analog signals, `stated_transition` will normally be absent because the upstream Atom does not know the resulting portfolio position. For the current model, it can be supplied and verified.

4. **Use two principal enums, not three closed enums**

I recommend:

**A. `PositionTransition`**

```text
OPEN
CLOSE
REVERSE
INCREASE
REDUCE
UNCHANGED
```

`OPEN/CLOSE/REVERSE` alone is insufficient once quantities can be larger than one.

It is deterministically derived from current and target quantities:

```text
0 → +2     OPEN
+2 → +4    INCREASE
+4 → +1    REDUCE
+1 → 0     CLOSE
+1 → -2    REVERSE
```

An upstream component may supply `stated_transition`. Portfolio calculates `computed_transition`, and a mismatch should reject the request or at least prevent automatic execution.

**B. `OrderRole`**

Something like:

```text
ENTRY
EXIT
REBALANCE
STOP_LOSS
TAKE_PROFIT
ROLL
LIQUIDATION
MANUAL
RECONCILIATION
```

This is independent of transition. A stop-loss and ordinary exit may both close a position, but their operational roles differ. A roll may preserve overall exposure while trading two legs.

**C. Do not create a closed `OrderType` enum**

Market, limit, stop, stop-limit, trailing, adaptive and custom combinations belong to execution implementations and IB orders. A framework enum would become restrictive quickly.

`BUY/SELL` can be a small `OrderSide` enum if useful, but it must always be derived from the signed delta or individual leg and validated. It is not a third independent intention.

5. **Keep ExecutionModels stateful, but remove sticky strategy dependence**

I agree that “stateless” was the wrong term. The undesirable state is implicit strategy data copied from startup, not execution lifecycle state.

Advanced execution models genuinely need state:

- submitted orders;
- fills and remaining quantity;
- replacement history;
- deadlines;
- observed spread and market conditions;
- cancellation state;
- recovery after restart.

The ABC in [execution_models.py](/home/tomek/haymaker/haymaker/execution_models.py:45) should therefore not require every implementation to expose `open()`, `close()` and `reverse()`. Those are useful helpers for the current implementation, not a universal execution protocol.

The minimal general contract should be closer to:

- consume an `ExecutionRequest`;
- return or emit an `execution_id`;
- issue, modify and cancel broker orders through Controller;
- emit correlated execution reports;
- restore or reattach unfinished work after restart.

It should permit three implementation patterns:

- One stateful instance bound to a chain/position owner—closest to the current system.
- One shared model managing multiple executions in an internal dictionary.
- A model that creates one stateful execution session per request—probably the best fit for sophisticated limit-order algorithms.

`ExecutionRouter` selects the implementation using an extensible policy key such as `"market_bracket"` or `"adaptive_limit"`. This should not be a closed enum.

The current bracket execution model then becomes one specialized implementation, retaining stop-loss monitoring and its other existing behaviour.

6. **Do not send trade requests through `onStart`**

`onStart` should mean lifecycle initialization only. Executing a trade is ordinary runtime data and should arrive through `onData` as an `ExecutionRequest`.

For an ExecutionModel:

- `onStart` restores or reattaches existing executions and subscriptions.
- `onData` receives new execution work.
- execution reports travel back to Portfolio with IDs linking them to the request and decision.

This removes the need for BaseExecutionModel to copy the chain’s arbitrary startup dictionary into strategy state.

7. **Replace snapshot persistence with three distinct concepts**

I am not proposing two additional Mongo collections immediately. I recommend:

| Collection | Purpose |
|---|---|
| `state` | Current recoverable state, updated in place |
| `decisions` | Immutable Portfolio decisions and their reasoning |
| `orders` | Broker-order facts and current lifecycle |

So this is one new collection, `decisions`; `state` replaces the current whole-snapshot `models/strategies` collection.

A future `executions` collection becomes worthwhile only when an execution can span multiple child/replacement orders and needs independent recovery.

`state` should contain one document per entity, rather than one enormous document:

```text
kind: proposal
source_id
instrument_id
latest value
readiness/staleness
extensions

kind: allocation
account
instrument_id
broker quantity last observed
pending delta
target quantity
contributor allocations
active constraints

kind: managed_position
position_id
owner_id
instrument_id
held contract
quantity
protection policy/state
active execution IDs
```

`decisions` answers questions that neither state nor orders can:

- Why did four requested ES contracts become two?
- Which contributors were included?
- Which volatility or sector constraint bound?
- Why was no order generated?
- What position did Portfolio believe existed?
- Which execution requests resulted?

A decision record should contain compact inputs and references to evidence, not entire dataframes.

`orders` should remain one updated document per broker order, but link to `decision_id`, `execution_id`, `position_id`, instrument, role and side. It should not copy the complete strategy dataframe and arbitrary upstream dictionary.

Alternatives:

- State plus orders only: simpler, but loses suppressed/no-trade decisions and repeats reasoning in orders.
- Full event sourcing: excellent history, but a substantial projection and recovery architecture.
- Current snapshots: easy to restore, but explains neither causality nor ownership and creates the repetition you are seeing.

8. **Standardize `onStart` with immutable lifecycle context**

The current [Atom.onStart](/home/tomek/haymaker/haymaker/base.py:230) mixes three responsibilities:

- lifecycle notification;
- propagation;
- arbitrary dictionary-based strategy/startup mutation.

I propose a small immutable `StartInfo` shared by every Atom:

```text
process_run_id
workload_generation_id
started_at
reason
```

Possibly also whether this is initial startup or supervised recovery, though that can be derived from generation/reason.

Every Atom would receive it, but most would ignore it.

Conceptually the framework-controlled handler would:

1. Notify the Atom that a particular upstream source has started.
2. Ignore duplicate one-time initialization for the same generation.
3. Process contract changes.
4. Call the user-overridable initialization hook.
5. Propagate the same immutable `StartInfo`.

The duplicate handling matters because one shared Portfolio may have many upstream chains. It must not reinitialize itself or all ExecutionModels once for every strategy chain.

A multi-input Atom may need two concepts:

- `onStart(info)`: once per workload generation;
- `onSourceStart(source, info)`: once for each incoming source, for readiness tracking.

Examples:

- Streamer: establishes subscriptions.
- Aggregator: synchronizes with its source.
- Block: optionally resets incremental state.
- Portfolio: restores its proposal/allocation book and waits for required sources.
- ExecutionModel: reattaches open execution sessions.
- Plain Atom: does nothing.

Global services remain in `RuntimeContext`. They should not be copied into `StartInfo`.

This also eliminates the current footgun where forgetting `super().onStart()` silently stops downstream initialization. The public hook should not be responsible for propagation.

9. **Make `Block` genuinely general and leave dataframe choice open**

Keeping `Block` for now is sensible.

The current Block is not fully general because it requires strategy/trading identity and expects `_signal()` to produce a dictionary. I recommend separating:

- `Atom`: bare event and runtime infrastructure.
- `Block`: user computational component with optional identity, persistence and tracing, but no required trading semantics.
- `DataFrameBlock`: convenience for dataframe-oriented calculations.
- A proposal-emitting component: attaches strategy/source and instrument identity when submitting something to Portfolio.

A Block could then be:

- one reusable indicator;
- an incremental calculation;
- a dataframe calculation;
- part of a multi-stage strategy;
- the final proposal generator.

I would not add an `IncrementalBlock` abstraction until multiple implementations reveal common behaviour. Users can initially implement incremental processing directly in `Block`.

This gives `Block` enough meaning beyond merely aliasing `Atom`: it is the user-computation layer with component identity and optional observation persistence.

10. **Make branching semantics explicit**

The issue is not linear pipelines. Sharing the same mutable object through a linear pipeline is useful. The danger arises at one-to-many branches: [connect()](/home/tomek/haymaker/haymaker/base.py:333) sends the same object to every target.

I propose explicit modes:

- `pipe`: same object passed along a linear chain;
- `broadcast/share`: intentionally send the same object to every branch;
- `fork`: clone independently for each branch.

`fork` should accept a copy policy:

```text
SHALLOW
DEEP
CUSTOM
```

No single default works universally:

- Shallow dict copies still share nested values.
- Deep-copying dataframes and broker objects may be expensive or impossible.
- Dataframes can use a custom `DataFrame.copy()`.
- User objects can provide a cloning callback.

Immutable lifecycle objects such as `StartInfo` are always shared.

Correlated message IDs will also be important. The existing generic backward [feedbackEvent](/home/tomek/haymaker/haymaker/base.py:304) becomes ambiguous once shared Portfolio and ExecutionRouter paths exist. Execution reports should identify their decision and execution rather than relying on “send backwards along whichever chain this came from.”

11. **Keep per-restart Block data, but add a run catalogue and explicit lineage**

Creating a new observation dataset after every supervised workload restart is defensible. It preserves exactly which recomputed stream the system used during that generation.

The current usability problem is discoverability and lineage, not necessarily retention.

I would give every persisted dataset:

```text
process_run_id
workload_generation_id
component_id
source/strategy_id, if applicable
logical instrument
concrete market-data contract per segment
selected trading contract per segment
created_at / completed_at
date range
row count
parameters/configuration hash
code revision, when available
physical Arctic symbol
```

Initially this could live in Arctic metadata with a framework query helper:

```text
list_runs(component=..., instrument=..., date=..., contract=...)
```

Only add a small Mongo `runs` catalogue if scanning Arctic metadata proves too slow.

Regarding today’s `localSymbol → symbol` change: with normal monotonic Arctic append, root `symbol` does not literally overwrite earlier rows. It does combine multiple contract regimes under one physical symbol and makes the stored contract lineage ambiguous; overlapping post-roll recalculations may also be discarded rather than replace earlier rows. So it is still a bug relative to your intended contract-specific preservation.

Short term, reverting to contract-local identity is appropriate. Longer term, I prefer one component/workload dataset with explicit per-row or per-segment contract lineage. That gives you a continuous run without hiding which contract produced each observation.

There is one further decision: does “what the system saw” mean:

- final accepted rows for each workload generation; or
- every revision/emission, including repeated values for the same timestamp?

The second requires an observation/revision ID or event-time plus emitted-time index. An ordinary timestamp-indexed appended dataframe cannot represent it faithfully.

12. **Deprioritize the global-runtime concern**

Given the documented one-runtime/one-strategy-module-per-process model, `Atom.runtime`, `Streamer.instances` and a singleton Portfolio are not current production problems.

The drawbacks are mostly:

- same-process reload and test isolation;
- accidentally retaining dead instances;
- future multi-account runtimes.

Those are valid but speculative here. I would document the ownership invariant and leave it alone. The intentional Portfolio singleton is different from accidental mutable component state and should remain.

13. **Move directly to `components`, after these contracts are agreed**

I agree with a single cut-over and no compatibility package.

Likely shape:

```text
haymaker/components/
    streamers.py
    aggregators.py
    blocks.py
    signals.py
    portfolio.py
    execution_models.py
    bracket_legs.py
```

Core runtime, Atom, Controller, state, persistence and contract management remain outside.

Before moving, I think we need agreement on:

1. Shared Portfolio topology and ownership identities.
2. Standard post-Portfolio execution request.
3. ExecutionModel/ExecutionRouter contract.
4. Generic Block versus strategy-bound proposal component.
5. New state/decision/order persistence boundaries.

Once those are settled, the package move can be done directly alongside import, documentation, test and external strategy updates. The existing Portfolio comparison bug and the root/local-symbol issue should be included in that implementation, but I have not changed anything yet.

14. **Define a general compatibility-validation policy for connected Atoms**

Some existing component pairs already validate structural compatibility, for
example aggregators and dataframe aggregators. The `PositionTarget` discussion
deliberately chose a different policy: the common message keeps
`PositionIntent` optional, while an intent-dependent ExecutionModel rejects a
target without it at runtime.

We should decide whether Haymaker needs one general policy for compatibility
between connected Atoms, including:

- which incompatibilities should be detected while constructing a Pipe, during
  `onStart`, or only when data arrives;
- whether Atoms should declare input/output types or narrower capabilities;
- how declarations would work when custom Atoms may pass arbitrary objects;
- whether compatibility metadata is advisory or enforced;
- which validation must still occur at runtime even after an earlier check;
- whether existing special checks in aggregators should remain local or use a
  shared mechanism.

The policy should improve early diagnosis without turning the general Atom
framework into a closed typed graph or adding negotiation machinery that
duplicates ordinary runtime validation.

15. **Separate transaction-contract identity from adjusted price-series generation**

`block_data` collection identity is currently derived from the strategy
Block's contract. That is the transaction contract selected by
`Atom.which_contract`, but it is not necessarily the contract supplying the
price series. A Block may select `NEXT` for new transactions while still
consuming prices from `ACTIVE`.

Using `localSymbol` created a new collection when `NEXT` changed, before the
market-data contract changed or back-adjustment occurred. Using the root
`symbol` prevents that premature split, but conflates pre-roll and
post-adjustment generations. Append persistence also writes only rows after
the stored `up_to` timestamp, so adjusted historical rows held in memory are
not rewritten in Arctic.

The required persistence semantics to evaluate are:

- a NEXT-only change must not rotate the persisted price series;
- a new series is created only after the ACTIVE market-data contract changes
  and back-adjustment completes successfully;
- the first save to the new series contains the complete adjusted dataframe;
- later saves append normally;
- the previous series remains unchanged;
- metadata records the source ACTIVE contract, preceding contract, adjustment
  basis and adjustment time;
- a failed adjustment does not create a new series;
- cold starts explicitly handle or report insufficient history for the
  adjustment.

The refactor should model transaction-contract identity and
price-series-generation identity separately. Persistence routing follows the
upstream adjusted-data generation, not `Block.contract` or
`Atom.which_contract`.

This issue must be resolved together with item 11. Its generation-per-adjustment
requirement conflicts with item 11's tentative suggestion of one continuous
component/workload dataset with per-segment lineage. The current project note
that root `contract.symbol` is the intentional collection identity is also
reopened by this issue.
