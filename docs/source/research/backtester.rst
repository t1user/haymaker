==========
Backtester
==========

Use the research backtester after a strategy has produced a ``position`` or
``blip`` series. It reports one-unit PnL, completed trades, account returns,
fixed-capital returns, and drawdowns without requiring a live-trading setup.

Quick Start
===========

For a strategy without stop logic, convert its desired position into the
transaction columns accepted by :func:`haymaker.research.backtester.perf`:

.. code-block:: python

   from haymaker.research.backtester import no_stop, perf

   strategy = bars.copy()
   strategy["position"] = position

   transactions = no_stop(strategy, price_column="open")
   result = perf(
       transactions,
       slippage=1,
       min_tick=0.25,
   )

``position`` must contain ``-1`` for short, ``0`` for flat, or ``1`` for long.
If the strategy produces ``blip`` events instead, ``no_stop()`` converts them
to next-bar positions. See :doc:`signals` for the timing conventions.

For exact stop, take-profit, or scheduled-close prices, prepare transactions
with :func:`haymaker.research.stop.stop_loss` and pass its result directly to
``perf()``. See :doc:`stop` for that workflow.

For exploratory work where the input may already be prepared, use
:func:`haymaker.research.backtester.auto_perf`:

.. code-block:: python

   from haymaker.research.backtester import auto_perf

   result = auto_perf(
       strategy,
       price_column="open",
       slippage=1,
       min_tick=0.25,
   )

Choosing Capital
================

``perf()`` evaluates one unit of the instrument. ``capital`` determines how
its point PnL is expressed as a return.

If ``capital`` is omitted, the first ``bar_price`` is used. This is a convenient
initially unlevered assumption: one unit of capital is allocated for one unit
of instrument price.

Pass ``capital`` when a different funding assumption is more useful:

.. code-block:: python

   result = perf(
       transactions,
       capital=5_000,
       slippage=1,
       min_tick=0.25,
   )

Capital and PnL must use the same units. The backtester keeps futures PnL in
price points and does not apply a contract multiplier. To analyze returns on a
dollar margin assumption, convert the margin to point-equivalent capital:

.. code-block:: python

   point_value = 50       # dollars per point
   margin_dollars = 22_000
   capital_points = margin_dollars / point_value

   result = perf(
       transactions,
       capital=capital_points,
       min_tick=0.25,
       slippage=1,
   )

This changes return and percentage-drawdown statistics. It does not change
trade PnL, trade count, or PnL reported in points.

Transaction Costs
=================

``slippage`` is the cost of each transaction leg in multiples of ``min_tick``.
For example, ``min_tick=0.25`` and ``slippage=1.5`` charge ``0.375`` points on
entry and again on exit.

Pass the known instrument tick whenever possible. If ``min_tick`` is omitted,
the backtester estimates it from observed prices. Estimation can be unreliable
when the data is sparse or prices move in multiples of several ticks. If the
price never changes, a tick cannot be inferred; explicit ``min_tick`` is then
required when slippage is non-zero.

Two Return Views
================

Every call reports both return views; there is no mode switch.

Account returns
   ``result.daily["returns"]`` uses the account equity available at the start
   of each reporting day. Use the unprefixed annual return, volatility, Sharpe,
   Sortino, Calmar, and drawdown metrics for conventional compounded account
   performance.

Fixed-capital returns
   ``result.daily["fixed_return"]`` always uses the original ``capital``.
   Metrics prefixed with ``fixed_`` answer the non-compounding question: how
   variable and productive was the strategy if the same capital base was used
   throughout?

The two views use the same trades and PnL. ``total_return`` is therefore simply
total PnL divided by initial capital. Their annual return and risk statistics
differ because their daily denominators differ.

Returns are calculated at reporting-day frequency, not at input-bar frequency.
Intraday bars are still used for equity and drawdown, so a loss that recovers
before the daily close remains visible in ``max_drawdown`` and
``max_drawdown_pnl``.

Annualization And Units
-----------------------

Return volatility and risk ratios use 252 reporting days per year. Fixed
annual return uses the same 252-day convention. ``monthly_pnl`` uses 21
reporting days and ``annual_pnl`` uses 252. The package does not replace those
conventions with the exact number of observed calendar days.

Sharpe uses a zero risk-free rate and Sortino uses a zero required return.
Return and drawdown percentages are stored as decimal fractions, so ``-0.12``
means a 12% loss. PnL and expectancy remain in instrument price points unless
the caller converts them after the run.

Reporting Days
==============

Only calendar dates represented by input observations are reported. Missing
weekdays and holidays are not inserted.

By default, Sunday observations are combined with the following Monday. This
is useful for futures data where Sunday evening begins Monday's activity:

.. code-block:: python

   result = perf(transactions, sunday_to_monday=True)

For a market where Sunday is an independent trading day, disable the behavior:

.. code-block:: python

   result = perf(transactions, sunday_to_monday=False)

This option only handles Sunday labeling. It does not infer exchange sessions,
overnight cutoffs, holidays, or timezones. Weekdays are determined in the
timezone already carried by the dataframe index.

Working With Results
====================

``perf()`` and ``auto_perf()`` return a
:class:`haymaker.research.backtester.Results` object with five fields.

.. list-table:: Result fields
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Use
   * - ``stats``
     - A Series of compact ``snake_case`` metrics for filtering, comparison,
       and grid-search tables.
   * - ``daily``
     - Reporting-day PnL, account returns, fixed returns, equity, and balance.
   * - ``positions``
     - One row per completed trade, including prices, PnL, and duration.
   * - ``df``
     - Original prepared bars enriched with slippage, PnL, equity, and
       drawdown paths.
   * - ``warnings``
     - Non-fatal conditions that limit how some metrics should be interpreted.

Typical access patterns are ordinary pandas operations:

.. code-block:: python

   headline = result.stats[
       ["annual_return", "sharpe_ratio", "max_drawdown", "net_pnl"]
   ]

   return_comparison = result.stats[
       [
           "annual_return",
           "fixed_annual_return",
           "sharpe_ratio",
           "fixed_sharpe_ratio",
       ]
   ]

   equity_curve = result.daily["equity"]
   losing_trades = result.positions.query("pnl < 0")
   intraday_drawdown = result.df["drawdown"]

A Practical Review Sequence
---------------------------

When evaluating a new strategy:

#. Read ``result.warnings`` before comparing headline statistics.
#. Compare ``gross_pnl`` with ``net_pnl`` to understand the effect of slippage.
#. Review account and fixed return metrics together; a large difference means
   changing equity materially changes the effective performance denominator.
#. Compare ``max_drawdown`` with ``median_21d_drawdown`` to distinguish the
   worst event from a more typical rolling month.
#. Check ``net_pnl_ex_best``, ``best_trade_share``, and ``top5_trade_share`` for
   dependence on a few outliers.
#. Inspect trade durations, time in market, and the completed trades themselves
   before changing stop or scheduling rules.

Daily Data
----------

.. list-table:: ``result.daily`` columns
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Meaning
   * - ``pnl``
     - Net one-unit PnL for the reporting day, in price points.
   * - ``returns``
     - Account return using beginning-of-day equity.
   * - ``lreturn``
     - ``log(1 + returns)``, useful for additive analysis. It is undefined at
       or below a 100% account loss.
   * - ``equity``
     - Initial capital plus cumulative net PnL.
   * - ``balance``
     - Equity normalized by initial capital; it begins around ``1.0``.
   * - ``fixed_return``
     - Daily PnL divided by unchanged initial capital.

Metric Reference
================

Return And Drawdown Metrics
---------------------------

.. list-table:: Account and fixed-capital metrics
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Interpretation
   * - ``total_return``
     - Total net PnL divided by initial capital.
   * - ``annual_return``
     - Compounded annual growth rate of account equity.
   * - ``annual_volatility``
     - Annualized standard deviation of daily account returns.
   * - ``sharpe_ratio``
     - Annualized account return per unit of total volatility, using a zero
       risk-free rate.
   * - ``sortino_ratio``
     - Annualized account return per unit of downside risk, using a zero
       required return.
   * - ``calmar_ratio``
     - Compounded annual return divided by absolute maximum account drawdown.
   * - ``max_drawdown``
     - Worst peak-to-trough account decline, measured on the bar-level equity
       path and reported as a negative fraction.
   * - ``median_21d_drawdown``
     - Median maximum drawdown across rolling 21-reporting-day windows; a view
       of typical monthly drawdown rather than the single worst event.
   * - ``skew`` / ``kurtosis``
     - Shape diagnostics for daily account returns. ``kurtosis`` is excess
       kurtosis.
   * - ``fixed_annual_return``
     - Mean fixed return multiplied by 252; it is arithmetic, not CAGR.
   * - ``fixed_annual_volatility``
     - Annualized volatility of fixed-capital daily returns.
   * - ``fixed_sharpe_ratio`` / ``fixed_sortino_ratio``
     - Risk-adjusted statistics using the unchanged capital base.
   * - ``fixed_max_drawdown``
     - Worst peak-to-trough PnL decline divided by initial capital.
   * - ``fixed_median_21d_drawdown``
     - Median rolling 21-day drawdown using the fixed capital denominator.
   * - ``max_drawdown_pnl``
     - Worst peak-to-trough decline in price points.

PnL And Trade Metrics
---------------------

.. list-table:: PnL and trade metrics
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Interpretation
   * - ``gross_pnl`` / ``net_pnl``
     - Total one-unit PnL before and after slippage.
   * - ``monthly_pnl`` / ``annual_pnl``
     - Mean reporting-day PnL scaled by 21 or 252. These are run rates, not
       realized calendar-month or calendar-year totals.
   * - ``trade_count`` / ``session_count``
     - Number of completed trades and observed reporting days.
   * - ``win_rate``
     - Winning trades divided by all trades. Breakeven trades remain in the
       denominator but are neither wins nor losses.
   * - ``avg_win`` / ``avg_loss``
     - Mean net PnL of strictly profitable or strictly losing trades.
   * - ``payoff_ratio``
     - Absolute average win divided by absolute average loss.
   * - ``profit_factor``
     - Total net PnL from winning trades divided by the absolute total net PnL
       from losing trades.
   * - ``trade_expectancy``
     - Mean net PnL per completed trade, in price points.
   * - ``trade_expectancy_ticks``
     - Mean trade PnL divided by ``min_tick``.
   * - ``long_expectancy`` / ``short_expectancy``
     - Mean trade PnL split by entry direction.
   * - ``trades_per_session``
     - Completed trades divided by observed reporting days.
   * - ``avg_duration`` / ``median_duration``
     - Typical completed-trade holding time.
   * - ``p90_duration`` / ``max_duration``
     - Long-tail and longest completed-trade holding time.
   * - ``time_in_market``
     - Fraction of input bars with exposure or a transaction.
   * - ``best_trade`` / ``worst_trade``
     - Largest and smallest individual net trade PnL.
   * - ``net_pnl_ex_best``
     - Total net PnL after removing the best trade, useful for detecting a
       strategy whose result depends on one outlier.
   * - ``best_trade_share`` / ``top5_trade_share``
     - Share of all winning net PnL contributed by the best winner or five
       best winners.

Warnings And Undefined Metrics
==============================

Warnings do not mean the PnL calculation failed. They explain why a statistic
is unavailable or potentially fragile. Always inspect them when reviewing a
new strategy:

.. code-block:: python

   for warning in result.warnings:
       print(warning)

Common cases include:

* Fewer than 21 reporting days: rolling drawdown metrics are ``NaN``.
* Fewer than two reporting days or zero volatility: volatility-based ratios
  are undefined.
* No negative returns: the corresponding Sortino ratio is undefined.
* Account equity becomes nonpositive: compounded account statistics are
  undefined from that point, but fixed-capital statistics and point PnL remain
  available.
* No trades or no inferable tick: trade or tick metrics are unavailable.
* Many zero- or one-bar trades, or a dominant final forced-close trade: inspect
  transaction timing and the end of the test period before trusting the result.

Invalid input, invalid capital or tick values, and failed PnL reconciliation
raise exceptions instead of producing a partial result.

Final Open Position
===================

An open position on the last bar is normally closed at that bar's reference
price so it contributes to PnL and trade statistics. Use
``skip_last_open=True`` when the final trade is incomplete and should be
excluded:

.. code-block:: python

   result = perf(transactions, skip_last_open=True)

This can materially shorten the reported bar range when no later completed
trade exists. Review ``result.df.index[-1]`` and ``result.positions.tail()``
after using it.

Primary API
===========

.. autofunction:: haymaker.research.backtester.no_stop

.. autofunction:: haymaker.research.backtester.perf

.. autofunction:: haymaker.research.backtester.auto_perf

.. autoclass:: haymaker.research.backtester.Results

The :doc:`examples` page links to notebooks with executed outputs.
