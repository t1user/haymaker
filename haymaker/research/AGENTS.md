# Research Package Guidance

This package is timing-sensitive trading research code. Small alignment changes can
materially change results. Prefer narrow, well-tested edits and verify semantics
against focused examples before refactoring.

## Core Timing Rules

- Treat event timing as the primary invariant. Do not "simplify" by moving
  signals, stops, or execution prices across bars without an explicit test.
- Read definitions of key terms used in the package from module level docstrings in `haymaker.research.signal_converters.py`
- Left-labeled lower-frequency data means each timestamp marks the beginning of
  its group. A grouped value becomes usable only when that grouped bar completes.
- `blip` and `close_blip` are event columns. Preserve their raw provenance as
  `raw_blip` / `raw_close_blip` when upsampling.
- Only canonical `blip` and `close_blip` get special event treatment by default.
  Do not infer special semantics from arbitrary `*blip*` column names.
- `position` is executable state. Do not upsample `position`; upsample generated
  signals/events/features first, then derive `position` on the upsampled frame.

## `upsample()`

- Public function lives in `haymaker.research.upsampling`.
- Use `hf_df` / `lf_df` names for high-frequency and low-frequency frames.
- Ordinary lower-frequency columns are propagated from their availability point.
- `sparse` is the single exception-list knob for event-like columns. Do not
  restore older `keep` / `propagate` style options.
- Passing a literal `position` column must raise. Columns containing
  `"position"` should warn because they are often already-shifted state.
- Tests for timing regressions belong first in `tests/test_research/test_utils.py`.

## `stop_loss()`

- Public entrypoint is `haymaker.research.stop.stop_loss`.
- The caller is responsible for pre-shifting blips to the intended execution bar;
  keep future-leakage warnings explicit in docs.
- `distance` is already in final price units. Do not reintroduce multiplier
  handling inside `stop_loss`.
- If `distance` is a Series, its index must exactly match `df.index`. Raise on
  mismatch; do not silently reindex.
- `scheduled_close` accepts `datetime.time`, a tuple accepted by `datetime.time`,
  a same-index boolean Series, `BeforeClose`, or `None`.
- `scheduled_close` is a scheduled close, not a stop. It should close an existing
  position and suppress an open on the same bar.
- `before_close(...)` returns a lazy `BeforeClose` helper resolved against the
  dataframe/index inside `stop_loss`. It is intraday-only, infers bar duration
  from the unique mode of positive index diffs, defaults `session_gap` to
  30 minutes, and should raise on bad data.
- Keep Python and Numba stop implementations behaviorally identical. Parity
  tests should cover scalar/Series distance, fixed/trailing stops, take profit,
  time stop, scheduled close, blip and position inputs, and edge cases.

## Backtester / `perf()`

- Current `perf()` expects a transaction dataframe with the prepared columns
  produced by `stop_loss()` / `no_stop()`:
  `bar_price`, `open_price`, `close_price`, `stop_price`, `position`.
- Do not reintroduce old tuple-unpacking call paths such as `perf(*data, ...)`.
- Same-bar open/stop rows are real zero-duration trades:
  open at `open_price`, close at `stop_price`, end flat, charge two slippage
  legs, and include `-(open_price + stop_price)` in gross trade PnL.
- Reversal plus same-bar stop is ordered: close existing position first via
  `close_price`, open the new position via `open_price`, then close the new
  position via `stop_price` if present.
- Keep `_perf_engine` and `_perf_engine_python` identical. Tests should assert
  Python/Numba parity and reconciliation between bar-level PnL and trade PnL.

## Bootstrap Package

- Synthetic-data code lives under `haymaker.research.bootstrap`.
- Shared OHLC preparation/reconstruction is in `bootstrap/data.py`; block
  bootstrap is in `bootstrap/block.py`; regime/Markov empirical generation is in
  `bootstrap/regime.py`; simple state helpers are in `bootstrap/states.py`.
- All generators return `list[pd.DataFrame]`, even for one path.
- Output length is always `len(data) - 1`, indexed as `data.index[1:]`, anchored
  by `data["close"].iloc[0]`.
- OHLC is represented as log distances from previous close. `volume` and
  `barCount` are sampled as raw bar attributes. Unknown columns, including
  `average`, are dropped unless a future change explicitly defines semantics.
- Avoid `iterrows()` and row-wise `apply()` for reconstruction. Prefer vectorized
  pandas/NumPy operations; use Numba only when vectorization is not suitable.
- `regime_bootstrap()` accepts user-provided hard state labels. Labels can be
  strings, ints, tuples, categoricals, etc., but must be one non-null label per
  generated bar and each state must pass minimum row/transition counts.
- Advanced HMM/GMM/KDE/MCMC fitting should remain external unless explicitly
  requested. Helpers may prepare feedable state Series, but should not hide
  statistical modeling decisions.

## Validation Defaults

- Focused research checks:
  `python -m pytest tests/test_research`.
- For typing/formatting when touching research code:
  `python -m mypy haymaker/research tests/test_research`
  and `python -m flake8 haymaker/research tests/test_research --select=F401,F821,F841,E501`.
- Preserve imports from `haymaker.research` when moving public
  functions, unless the user explicitly asks to break compatibility.
