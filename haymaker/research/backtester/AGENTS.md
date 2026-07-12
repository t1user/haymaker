# Backtester Package Guidance

This package is the active research backtester. It evaluates theoretical
strategies prepared in pandas; it does not model broker state or consume live
trades. Preserve the package's accounting and timing semantics when cleaning up,
extending, or optimizing it.

## Active Implementation

- `pipeline.py` owns the public preparation and reporting API: `no_stop()`,
  `auto_perf()`, `perf()`, and `Results`.
- `engine.py` owns the single-pass accounting engine. Keep the Numba and Python
  implementations behaviorally identical.
- `metrics.py` owns reporting-day aggregation, account/fixed returns,
  drawdowns, and summary statistics. Keep these calculations pure and testable
  from hand-calculated PnL examples.
- Result-analysis helpers, including excursions and entry-time factor
  extraction, live in `haymaker.research.result_analysis` and are re-exported
  here only for compatibility.
- `retired/vector_backtester.py` is reference code retained for parity checks.
  Do not import it from active package code or add new features to it.

## Input Contract

- `perf()` accepts a bar-indexed transaction DataFrame containing
  `bar_price`, `open_price`, `close_price`, `stop_price`, and `position`.
- `bar_price` is the reference price used for mark-to-market accounting. When
  `capital` is omitted, its first value is also the initial capital assumption.
- `position` is the position held at the end of the bar and is limited to
  `-1`, `0`, or `1` under the current contract.
- `open_price` is signed by the opened position: positive for a long entry and
  negative for a short entry.
- `close_price` and `stop_price` are signed by the closing transaction: negative
  when closing a long and positive when closing a short.
- Use `no_stop()` to prepare position/blip strategies and `stop_loss()` for
  strategies requiring exact intra-bar exit prices. Do not make `perf()` infer
  stop execution prices from coarse OHLC bars.
- A blip is generated information and becomes executable on the following bar.
  Do not move its execution to the signal bar.

## Transaction Ordering

- A bar may contain no action, one entry or exit, a reversal, an entry followed
  by an immediate stop, or a reversal followed by an immediate stop.
- On a reversal, close the existing position at `close_price` before opening the
  new position at `open_price`.
- If the new position also stops on that bar, close it at `stop_price` after the
  reversal open. In that case both `close_price` and `stop_price` are valid on
  the same row because they close different positions.
- Same-bar open/stop trades are real zero-duration trades. Their gross PnL is
  `-(open_price + stop_price)` and they incur two transaction legs.
- An open position at the end of the input is synthetically closed at the final
  reference price. `skip_last_open=True` excludes that synthetic trade from the
  reported result according to the public contract.

## Accounting Invariants

- Slippage is a non-negative, conservative cost expressed as a multiple of the
  inferred minimum tick and charged per non-zero transaction leg.
- Slippage always worsens execution: buys cost more and sells receive less. It
  also applies to stop exits because a stop level is a trigger, not a guaranteed
  fill price.
- Bar-level mark-to-market PnL must reconcile with trade-level PnL, including
  slippage and same-bar trades.
- Bar PnL is the canonical engine output. Do not calculate returns in the
  engine or compound returns at bar frequency.
- Reporting uses only observed calendar dates. By default Sunday observations
  are combined with the following Monday; `sunday_to_monday=False` retains
  Sunday as an independent reporting day. Do not synthesize missing weekdays.
- Standard session return is session PnL divided by beginning session equity.
  Fixed return is session PnL divided by unchanged initial capital. Capital and
  PnL must use the same units.
- Account equity and intraday drawdown are derived directly from cumulative bar
  PnL. Standard account statistics become undefined after equity is
  nonpositive, while fixed-capital and point-PnL analysis continues.
- Do not change transaction ordering, price signs, timing, slippage, or final
  position treatment based on intuition alone. Add a focused example first and
  ask when the intended market semantics are unclear.

## Deferred Portfolio Work

- General portfolio construction is intentionally out of scope for the current
  backtester and should not be added to `grid_search.py`.
- A future portfolio module must distinguish capital-weighted strategy return
  streams from fixed-contract portfolios. Fixed-contract aggregation requires
  point PnL, quantities, reference prices, and contract point values; price-only
  or equal-return weighting is not an economic substitute.
- `GridSearchResult.combined_returns()` remains a convenience for equal-weight,
  daily-rebalanced simulation sleeves, not a fixed-contract portfolio engine.

## Compatibility And Validation

- Preserve public imports from `haymaker.research.backtester` and
  `haymaker.research` unless a compatibility break is explicitly requested.
- Treat `retired` as non-public. It exists only for comparison, benchmarks, and
  migration evidence.
- For accounting changes, test the Python engine first, then assert exact or
  tolerance-appropriate parity with the Numba engine.
- Run focused checks first:

```bash
python -m pytest tests/test_research/test_backtester.py \
  tests/test_research/test_parity_old_new.py
```

- For broader changes, follow the validation commands in
  `haymaker/research/AGENTS.md`.
