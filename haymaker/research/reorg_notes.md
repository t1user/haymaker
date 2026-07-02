# Research Utility Reorganization Notes

This note records the current working taxonomy for reorganizing research helpers.
It is intentionally provisional: before moving indicator code, review each
indicator for correctness, usefulness, live-trading usage, and import cost.

## Import Boundaries

- Keep `haymaker.research.__init__` light. Avoid re-exporting modules that import
  backtester, pyfolio, plotting libraries, numba, or other optional/heavy
  dependencies.
- Keep `haymaker.indicators` as a compatibility facade over
  `haymaker.research.indicators` for now because live trading code may import
  from it. Remove the facade only after live usage is audited and migrated.
- Importing general indicator helpers should not import libraries that are only
  needed by specific indicators. In particular, numba-backed helpers should be
  isolated or imported lazily.
- Do not keep a `research/utils.py` compatibility layer unless there is a real
  caller to preserve. Prefer explicit imports from the final module names.
- Leave old backtester utilities in place until the old backtester/reference code
  is removed or individual helpers are deliberately promoted.

## Working Categories

### Market Data Tools

Tools that reshape, align, resample, sample, normalize, or validate market data.
They do not encode trading logic.

Examples: `resample`, `weighted_resample`, `downsampled_func`,
`downsampled_atr`, `sampler`, `gap_tracer`, `round_tick`.

### Indicators

Functions that calculate market-derived features. They answer: "what is the
state of the market or series?"

Mean examples: `mmean`, `rolling_weighted_mean`, `rolling_weighted_std`,
`weighted_zscore`.

Technical examples: `true_range`, `atr`, `rsi`, `macd`, `tsi`, `carver`, `adx`,
`strength_oscillator`, `chande_ranking`, `chande_momentum_indicator`, `spread`,
`momentum`, `divergence_index`.

Breakout examples: `min_max_blip`, `min_max_index`, `breakout`,
`breakout_blip`.

### Generic Indicator Transformers

Functions that interpret prices or indicators into trading-intent-like outputs:
signals, blips, entries, exits, long/short/flat decisions, or signal filters.

Moved to `haymaker.research.indicators.transformers`: `crosser`,
`extreme_reversal_blip`, `range_blip`, `signal_generator`, `combine_signals`,
`inout_range`.

### Performance Metrics

Functions that compute risk, return, or statistical metrics from returns or
backtest output.

Examples: `true_sharpe`; future local replacements for pyfolio-style metrics.

### Result Analysis

Functions that inspect, decompose, or present backtest results. They answer:
"what happened in this simulation?"

Examples: `paths`, `long_short_returns`, `excursions`,
`profitable_excursions`, `adverse_excursions`, `factor_extractor`.

### Backtest Tools

Convenience functions that run simplified backtests or threshold summaries.

Examples: `v_backtester`, `summary`.

### Glue Or Review Before Moving

Decorators, private helpers, deprecated names, unclear old helpers, and
numba-backed wrappers should not define the architecture until reviewed.

Examples: `mmean`, `join_swing`.

## Initial Shape To Consider

Current broad modules:

- `haymaker.research.indicators.mean`: moving-average, weighted-mean, and
  weighted z-score helpers.
- `haymaker.research.indicators.technical`: market-data helpers and technical
  indicator calculations.
- `haymaker.research.indicators.breakout`: breakout/channel indicators and
  strategy-ready breakout wrappers.
- `haymaker.research.indicators.transformers`: generic helpers that transform
  indicators/prices into discrete events, filters, or desired exposure.
- `haymaker.research.tools`: market-data tools, result analysis, and backtest
  tools that are not tightly owned by a specific package.
- `haymaker.research.metrics`: performance metrics.
