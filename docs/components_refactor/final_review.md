# Concrete-contract and futures-roll refactor: final review

Reviewed 2026-09-07 after implementation checkpoints 1–8. This review does not
silently implement additional behavioral changes. The implementation is not a
claim of live-production readiness: the first finding needs attention before
relying on rolling during in-flight one-to-one execution.

## Remaining findings

Follow-up status (2026-09-08): the stale bracket-roll finding is fixed. Pending
episodes are refreshed together after active work/accounting settles; flat
sources are skipped and completed physical offsets are preserved. Independent
entry/close regressions and offset/replaced-episode tests pass. The findings
below retain the original evidence; further resolution is recorded below.

Query/configuration follow-up: logical_positions now replays direct evidence
once per compound query, with no cache lifecycle added. Zero allocation and
bare-string source declarations are rejected. All five override parameter
diagnostics are fixed; Pyright reports zero errors (the two intentional
aggregated-export warnings remain). Full pytest: 1,647 passed; mypy: 112 files
clean. The stale target-key docstring is corrected.

### High: waiting bracket rolls retain a stale participant quantity

`haymaker/components/execution/future_roll.py:659–697` waits for source OPEN or
CLOSE work, then submits the previously captured RollParticipant unchanged.
Neither quantity nor episode identity is refreshed before submission.

A read-only reproduction using the focused broker harness produced:

```text
Initially: WAITING_FOR_ACTIVE_WORK
Held after entry: 2.0
Roll submitted: 1.0
```

The roll was discovered with one unit filled and another entry unit pending.
After entry completion, it still rolled one. The later position-consistency
check can block, but only after the wrong-sized broker roll. A CLOSE completing
while waiting can similarly leave an obsolete planned roll.

Recommendation: after active work finishes, re-read all unprocessed source
episodes before any broker side effect. Skip flat episodes, reject unexpected
episode replacement, and rebuild the remaining physical/net allocation from
current quantities. Defer the resume callback until Controller accounting has
settled, as the direct executor already does. Add independent tests for partial
entry completion and close-to-flat while a roll is waiting. Do not simply change
one participant quantity without reconsidering offsetting-source allocation.

### Medium: direct position queries repeatedly replay the complete fill history

`haymaker/book.py:1053` calls `direct_positions()` for each exact quantity query.
`logical_positions()` then calls aggregate queries for multiple Contracts;
`Portfolio.positions_for_blueprint()` invokes that full-account query before
filtering. Cost grows with retained order/Fill evidence and with the number of
Contracts, on the event-loop thread.

Recommendation: first calculate the direct position mapping once per compound
query. If measurement justifies more, maintain a rebuildable cached projection
invalidated by Fill application, restore, and reset. Keep normalized Fill
evidence authoritative; no new source-allocation ledger is needed.

### Low: Portfolio configuration has two avoidable surprises

- `FixedSizeAllocator.target_for()` permits zero although its docstring says
  positive. For an OPEN proposal it produces a zero target with OPEN intent,
  which BracketExecutionModel rejects. Recommend rejecting zero size with an
  actionable message; a custom allocator can return None to suppress execution.
- `Portfolio(sources="alpha")` interprets the string as individual source
  characters. Recommend explicitly rejecting a bare string and asking for
  `sources={"alpha"}`. This is a configuration error, not a source-accounting
  feature.

### Low: typing and reference documentation checks are not fully clean

With the actual project interpreter available, Pyright reports five existing
parameter-name override diagnostics: ExecutionModel.onData, both Portfolio
onData methods, the signal processor onData method, and a dataframe grouper
override. Their signatures were checked against pre-refactor commit `2e88442`.
Two warnings concern the deliberately module-owned, aggregated `__all__` lists.

Recommendation: align overriding parameter names with their bases, retaining
descriptive local variables if useful. Keep the approved public-export design;
do not introduce a second manually maintained export list to silence Pyright.

Strict Sphinx with external inventories available reports 81 unresolved
references. These include old shorthand references in Atom/SignalModel,
unindexed framework classes (including FutureSelector), third-party private
type paths, and research documentation references. The three newly exposed
heading-underline warnings were corrected. Recommend a focused documentation
reference pass with canonical type names and explicit API targets; do not
globally suppress missing references. The HTML output was generated, but the
strict validation command failed and is not counted as a clean pass.

Minor documentation cleanup: `execution.router.where` still says “persisted
target key”; the final recovery fields are concrete Contract, quantity, and
creation time.

## Verified implementation and deliberate boundaries

- Held and pending one-to-one Contracts/inputs are separate. OPEN selects the
  incoming Contract; CLOSE uses Book's episode Contract; REVERSE closes before
  opening the incoming Contract under a new episode ID.
- PortfolioWrapper allocates and transfers intent, without held-contract logic.
- Direct targets are concrete conId setpoints. Portfolio owns cross-expiry and
  source allocation; default direct execution does not infer or substitute an
  existing holding.
- Blueprint snapshots and qualified membership are explicit and overlapping
  declarations fail atomically. Explicit nth selection does not clamp.
- Default rolling selects only past contracts. Custom policies select triggers
  and destinations; fixed-schedule occurrence completion is durable.
- Direct target-transfer snapshots survive interruption between their writes and
  yield to newer explicit targets. Bracket roll tests cover replacement stops,
  episode preservation, and reversal requested during an already active roll.
- Optional Portfolio persistence and completion feedback are available, not
  enforced. Roll completion callbacks are not a durable message queue; custom
  Portfolio recovery must reconcile with Book.
- The converter is standalone, defaults to dry-run, preserves explicit evidence,
  and refuses ambiguous allocation or in-flight roll schema changes. No real
  database, production configuration, credentials, or environment files were
  changed. Nothing was pushed.

## Validation

- Full pytest: **1,637 passed**.
- Mypy: **112 source files clean**, including the converter and broker harness.
- Black: **38 changed Python files clean** (`--check --fast`, Python 3.12).
- `git diff --check`: clean.
- Public-export docstring checks and the retained processor matrices run as part
  of the full suite. Public docs were also manually reviewed for purpose,
  composition, Contract semantics, policy extension points, and recovery limits;
  the mismatches above are not hidden by the existence test.
- Pyright: **5 errors, 2 warnings**, detailed above. Sandbox runs initially could
  not launch the interpreter; the reported final run used the real environment.
- Sphinx `-n -W --keep-going`: **81 reference warnings**, with inventories fetched
  outside the sandbox. Not a successful strict build.

The original integration suite and full processor matrices remain. New
independent paths are `test_episode_end_to_end.py`, `test_direct_end_to_end.py`,
and `test_roll_end_to_end.py`, driven through real Controller/Trader events and
an independent broker ledger. They supplement, rather than certify every
possible interleaving of, the existing tests.
