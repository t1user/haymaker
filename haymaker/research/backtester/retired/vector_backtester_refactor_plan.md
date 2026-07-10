# Refactoring Plan: Vector Backtester (`perf`)

---

## 1. Background & Objective

### What is `perf`?
`perf` is a **research tool**. A researcher codes a strategy in pandas that indicates, at each price
bar, what position the strategy would hold. `perf` takes this signal and estimates:

- At what prices would positions have been opened and closed?
- What PnL would have resulted?
- What are the key statistics (Sharpe, win rate, drawdown, etc.)?

There is no broker. There are no real trades. It is a pure simulation designed to be called
interactively in Jupyter notebooks.

### Why refactor?
The current `vector_backtester.py` has grown into a single 1000-line file with:
- A "God Function" (`perf`) that does input normalization, mark-to-market, trade matching, and stats.
- A dual code path: simple signals vs. stop-loss output, interleaved throughout.
- A `stop_adj` patch applied post-hoc to correct PnL for intra-bar stop prices.
- Trade matching (`pos()`) reconstructed from a position series via complex pandas joins instead
  of being recorded directly as trades happen.
- An ambiguous input API where `perf` accepts many different things with no single contract.

### Goal
Replace the dual-path with a **single, clean pipeline**:

```
researcher input → TransactionFrame (normalised) → Numba engine → Results
```

Old code stays untouched until the new implementation is verified correct.

---

## 2. Terminology

Defined in `signal_converters.py` (summarised here for reference):

| Term | Values | Meaning |
|:---|:---|:---|
| **indicator** | continuous | Raw computation the strategy is based on (e.g. MACD) |
| **signal** | -1, 0, 1 | What the strategy *wants* to be at each bar; stateful, persists until changed |
| **blip** | -1, 0, 1 | Momentary intent to trade; non-zero only when action is required |
| **transaction** | -1, 0, 1 | Bar where actual execution occurs (typically one bar after signal/blip) |
| **position** | -1, 0, 1 | Actual holding after each bar (-1=short, 0=flat, 1=long) |

`signal → position` via `sig_pos` (shift by 1 bar).
`blip → signal` via `blip_sig(always_on=...)`.
`position → transaction` via `pos_trans`.

### Always-on vs. not-always-on
- **Always-on**: a signal/position in the *opposite* direction to the current position triggers an
  immediate reversal — position goes -1→+1 in a single step (close short, open long, same bar).
- **Not always-on**: an opposite signal closes the current position to flat; opening in the other
  direction requires a separate, subsequent signal.

---

## 3. Why Stop-Loss Needs Separate Transaction Prices

With coarse bars (e.g. hourly), a stop may fire intra-bar. The next bar's open price could be far
from where the stop actually triggered. The `stop_loss()` function determines *exactly* where
the stop level was hit (from high/low data), and records that price. Without this, PnL would be
over-optimistic.

This is why `stop_loss()` returns explicit `open_price`, `close_price`, `stop_price` columns
rather than just a position series.

---

## 4. Per-Bar Transaction Column Combinations

Output of `stop_loss()` has columns: `position`, `open_price`, `close_price`, `stop_price`.

`close_price` and `stop_price` are **mutually exclusive** — a position can only be closed one way
per bar. Valid combinations:

| Scenario | `open_price` | `close_price` | `stop_price` | `t_count` |
|:---|:---:|:---:|:---:|:---:|
| No action | 0 | 0 | 0 | 0 |
| New position, no immediate stop | ≠0 | 0 | 0 | 1 |
| Normal close (no reversal) | 0 | ≠0 | 0 | 1 |
| Stopped out | 0 | 0 | ≠0 | 1 |
| Opened + immediately stopped (same bar) | ≠0 | 0 | ≠0 | 2 |
| Reversal: normal close + reopen | ≠0 | ≠0 | 0 | 2 |
| Reversal: normal close + reopen + immediate stop | ≠0 | ≠0 | ≠0 | 3 |

The "open + immediate stop" case exists to handle sparsely sampled data: a bar may have made a
significant adverse move *and recovered* within one bar. The stop engine records the exact
intra-bar stop price so `perf` can use it directly.

---


## 5. Input API: `_TransactionFrame` (internal)

### Design decision
The **public API uses plain `pd.DataFrame`** throughout. Both factory functions return a regular
DataFrame with a defined column schema. `perf` accepts a plain DataFrame and validates it
internally using a private `_TransactionFrame` dataclass.

This keeps the interface simple for researchers (no new types to learn) while enforcing the
schema at the boundary inside `perf`.

### Required DataFrame schema

All columns are bar-indexed (same index as the input `df`):

| Column | Type | Meaning |
|:---|:---|:---|
| `bar_price` | float64 | Reference price for every bar (mark-to-market and log return denominator) |
| `open_price` | float64 | Signed entry price; non-zero only when a position is opened this bar |
| `close_price` | float64 | Signed normal exit price; non-zero only when a position is closed normally |
| `stop_price` | float64 | Signed stop/tp/time-stop exit price; non-zero only when a bracket fires |
| `position` | int8 | Resulting position at end of bar (-1, 0, 1) |

`close_price` and `stop_price` are mutually exclusive per bar.
All price columns are signed by trade direction (positive for long entry, negative for long exit).

### Internal validation (`_TransactionFrame`)

```python
@dataclass
class _TransactionFrame:
    """Internal validator. Not part of public API."""

    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"bar_price", "open_price", "close_price", "stop_price", "position"}
    )

    data: pd.DataFrame

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self.data.columns)
        if missing:
            raise ValueError(
                f"perf() received a DataFrame missing required columns: {missing}. "
                "Use signals_to_transactions() or stop_loss() to produce the input."
            )
```

`perf` wraps the input immediately:
```python
def perf(data: pd.DataFrame, slippage: float = 1.5, ...) -> Results:
    return _PerfCalculator(_TransactionFrame(data), slippage, ...).run()
```

### Factory 1: Simple path (`no_stop`)

```python
def no_stop(
    df: pd.DataFrame,
    price_column: str = "open",
) -> pd.DataFrame:
```

**Dispatch Logic & Precedence**:
1. **`position` column**: If present, it is used directly as the desired position.
2. **`blip` column**: If `position` is absent but `blip` is present, it uses the blip path.
3. **Validation**: If neither is present, it raises a `ValueError`.

**Blip Path Logic**:
To maintain consistency with the stop-loss engine, blips are treated as "information signals" generated at bar N and acted upon at bar N+1:
- `blip` (and optional `close_blip`) are **shifted by 1 bar** forward.
- The shifted blips are passed to `blip_sig()` to derive a stateful position series.
- This results in a `position` series where changes occur at the execution bar.

**Output Schema Generation**:
Returns a DataFrame with the required schema. Internally generates:
- `bar_price = df[price_column]`
- `position = position_series` (either direct or derived from blips)
- `open_price = df[price_column]` where `position` opens or reverses
- `close_price = df[price_column]` where `position` closes normally
- `stop_price = 0` always

This ensures that regardless of whether the researcher provides raw positions or blips, `perf` always receives a fully reconciled `position` series and the corresponding execution prices.

### Factory 2: Stop path (`stop_loss`)

`stop_loss()` already returns a DataFrame with `open_price`, `close_price`, `stop_price`,
`position`. The **only change required** is to add `bar_price` as a column (a copy of
`df[price_column]`). The return type remains `pd.DataFrame`. No other changes to `stop_loss`.

```python
# In stop/interface.py, _build_output — add one line:
out_df["bar_price"] = price_series  # df[price_column], passed in from stop_loss()
```

### Optional convenience wrapper (future)

```python
def perf_from_signals(df, price_column, position_column, slippage):
    return perf(no_stop(df, price_column), slippage)
```

---




## 6. Slippage

Slippage always makes the transaction **less advantageous** for the trader:

| Trade direction | Adjustment |
|:---|:---|
| Long (buy) | `execution_price = price + slippage` |
| Short (sell) | `execution_price = price - slippage` |

Slippage is expressed as a multiple of `min_tick` (estimated from price data via `get_min_tick`).
`cost = min_tick * slippage_multiple`.

**Applied uniformly to all transaction types**, including stop exits. Rationale: slippage is also
meant to cover transaction fees and we want conservative (pessimistic) estimates. A stop price is
the trigger level, not the final filled price.

Slippage per bar = `t_count * cost`, where `t_count` is the number of non-zero price columns
(`open_price`, `close_price`, `stop_price`) on that bar.

---

## 7. Architecture

### 7.1 New module
Implementation goes in a new file: `haymaker/research/backtester.py`.
`vector_backtester.py` is **not modified** until the new implementation is verified correct.

### 7.2 Public interface

```python
def perf(
    tx: TransactionFrame,
    slippage: float = 1.5,
    skip_last_open: bool = False,
    raise_exceptions: bool = True,
) -> Results:
    """Return performance statistics and underlying data for debugging."""
    return _PerfCalculator(tx, slippage, skip_last_open, raise_exceptions).run()
```

`Results` NamedTuple is unchanged — same fields as today for compatibility.

### 7.3 Internal `_PerfCalculator` class

```python
class _PerfCalculator:
    def __init__(self, tx: TransactionFrame, slippage: float, ...): ...

    def _prepare_arrays(self) -> ...:
        """Convert TransactionFrame to numpy arrays for Numba."""

    def _run_engine(self) -> tuple[np.ndarray, np.ndarray]:
        """Call Numba core. Returns (bar_log_returns, trade_records)."""

    def _build_positions(self, trade_records: np.ndarray) -> pd.DataFrame:
        """Convert raw trade records back to positions DataFrame."""

    def _build_daily(self, bar_log_returns: np.ndarray) -> pd.DataFrame:
        """Resample bar returns to daily."""

    def _build_stats(self, positions: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        """Calculate all statistics (Sharpe, EV, win rate, etc.)."""

    def run(self) -> Results: ...
```

Each method is independently testable.

### 7.4 Numba core: `_perf_engine`

New file: `haymaker/research/_perf_engine.py`.

```python
@jit(nopython=True)
def _perf_engine(
    bar_price: np.ndarray,      # reference price for every bar (mark-to-market)
    open_price: np.ndarray,     # signed entry price (0 if no entry)
    close_price: np.ndarray,    # signed normal exit price (0 if no close)
    stop_price: np.ndarray,     # signed stop exit price (0 if no stop)
    cost: float,                # slippage cost per transaction
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-pass engine over price bars.

    State maintained: current_position, entry_price, entry_bar_index.

    For each bar:
    1. Apply any close/stop transactions → update position to 0, record trade.
    2. Apply any open transaction → update position, record entry.
    3. Calculate bar PnL (mark-to-market against bar_price).
    4. Calculate log return for the bar.

    Returns:
        bar_log_returns: shape (n_bars,) float64
        trade_records:   shape (n_trades, 6) float64
                         columns: entry_bar, exit_bar, entry_price, exit_price,
                                  gross_pnl, slippage
    """
```

**Why this is better than the current implementation:**
- No `stop_adj` needed: we use the actual stop price directly for PnL on stop bars.
- No post-hoc `open_stop` calculation: the open+stop same-bar case is handled by processing
  open_price then stop_price sequentially within the same bar iteration.
- No `pos()` trade matching: trades are recorded as they happen.
- No pandas `.shift()` or `.join()` in the hot path.

---

## 8. `skip_last_open` Behaviour

If `skip_last_open=True`: if the last position in the test period is still open at the final bar,
exclude it from stats and returns. Rationale: an unclosed position at the end of the backtest
period represents a mark-to-market value, not a realised result. If that position is large relative
to total PnL, it may make the strategy appear better (or worse) than it actually is.

**Implementation**: the Numba engine always records the "last open" as a trade closed at the final
bar price. The `skip_last_open` flag is applied in `_build_positions` by filtering out the last
trade if it corresponds to a position that was still open at bar[-1].

**Note**: the current implementation's `_skip_last_open` function contains a broken
`assert isinstance(i, int)` that fails with datetime indices. The new implementation handles
this correctly without the assertion.

---

## 9. Implementation Steps

### Phase 1: Data layer
1. Define `TransactionFrame` NamedTuple in `backtester.py`.
2. Implement `signals_to_transactions()`.
3. Modify `stop_loss()` in `stop/interface.py` to return `TransactionFrame`.
4. Write unit tests for both factories covering all 7 transaction scenarios.

### Phase 2: Numba engine
5. Implement `_perf_engine` in `_perf_engine.py` (Numba JIT).
6. Implement a reference Python version of the same loop for testing/debugging.
7. Verify both versions produce identical results on test data.

### Phase 3: Calculator and output
8. Implement `_PerfCalculator` class.
9. Implement `perf()` public function.
10. Replicate all stats from current `Results` output.

### Phase 4: Verification
11. Run `perf` (old) and `perf` (new) in parallel on a range of test scenarios:
    - Simple long-only strategy
    - Always-on (reversing) strategy
    - Strategy with stop-loss
    - Strategy with open+immediate-stop bars
12. Assert `abs(old.positions.pnl.sum() - new.positions.pnl.sum()) < epsilon` for each.
13. When verified, update `__init__.py` to expose new `perf` as the default.

### Phase 5: Utilities review
Review each utility function in `vector_backtester.py`:
- `summary()`, `v_backtester()`: decide whether to port or discard.
- `excursions()`, `profitable_excursions()`: port if still needed; no Numba required.
- `factor_extractor()`, `blip_extractor()`: likely keep as-is.
- `duration_warning()`, `last_open_position_warning()`: port if kept.
- `optimize()`: delete (dead code, no type annotations, not used anywhere).
- `efficiency()` variants: simple scalar functions, port as-is.

---

## 10. Resolved Correctness Flags

### `open_stop` PnL — **Correct**
For bars where a position is opened and immediately stopped (scenario 5, 6, 7), the formula
`-(open_price + stop_price)` correctly computes the net PnL. Since prices are signed by
direction, `open_price + stop_price` yields the net cash flow, and negating gives gross PnL.
Verified by tracing through all 7 scenarios algebraically.

### `pos()` price reconstruction — **Design flaw, not correctness bug**
The current code reconstructs a price series for trade matching that mixes `df["price"]`
(caller-supplied) and `stop_price` (from stop engine). If the caller passes a price that matches
what stop used internally, results are correct. The new API eliminates this entirely by design:
`TransactionFrame` carries the correct prices, so reconstruction is unnecessary.

### Slippage on stop exits — **Intentional**
Slippage is applied to all transaction types including stop exits. This is deliberately
conservative: slippage covers both market impact and transaction fees, and we want to err on
the side of caution. Stop trigger price is where the stop fires, not necessarily the final fill.

### `_skip_last_open` broken assert — **Will be fixed in new implementation**
The `assert isinstance(i, int)` fails with datetime-indexed data. The new implementation
handles this via a flag in `_build_positions`, no index type assumption needed.

### `optimize()` — **Dead code, to be deleted**
Not used anywhere in the codebase. No type annotations. Will not be ported.
