==========
Backtester
==========

The backtester is the final step in the basic research workflow. It consumes a
transaction dataframe and returns performance statistics, daily returns,
trade-level records, an enriched bar dataframe, and warnings. The accounting
engine produces PnL in instrument price points; percentage returns are derived
after PnL has been grouped into reporting dates.

For strategies without stop logic, use
:func:`haymaker.research.backtester.no_stop` to
convert a dataframe containing ``position`` or ``blip`` into the transaction
schema expected by :func:`haymaker.research.backtester.perf`.

For notebook work, :func:`haymaker.research.backtester.auto_perf` is a
convenience wrapper
that accepts either an already prepared transaction dataframe or a raw strategy
dataframe that can be passed through
:func:`haymaker.research.backtester.no_stop`.

Primary API
===========

.. autofunction:: haymaker.research.backtester.no_stop

.. autofunction:: haymaker.research.backtester.perf

.. autofunction:: haymaker.research.backtester.auto_perf

.. autoclass:: haymaker.research.backtester.Results
   :members:

Return Models
=============

``perf()`` reports two views of the same reconciled one-unit PnL stream. Let
``C`` be initial capital, ``P_t`` the PnL for reporting session ``t``, and
``E_(t-1)`` beginning session equity.

Account return
   ``P_t / E_(t-1)``. These returns geometrically link to the account balance
   and feed CAGR, annual volatility, Sharpe, Sortino, and Calmar.

Fixed return
   ``P_t / C``. The denominator never changes. Fixed annual return is the
   arithmetic mean session return multiplied by 252, rather than CAGR.

Initial capital defaults to the first ``bar_price``. Pass ``capital`` to use a
different assumption, such as a futures margin amount expressed in units
consistent with the point PnL produced by the backtester. The package does not
apply contract multipliers.

Returns are calculated only at reporting-session frequency. Bar PnL is used to
construct account equity and retain intraday drawdowns without implying that
capital is reinvested on every input bar.

Reporting Dates
===============

Only dates represented by input observations are included. Missing weekdays
and holidays are not synthesized. By default, Sunday observations are assigned
to the following Monday before aggregation. Pass ``sunday_to_monday=False``
when Sunday is an independent trading day.

This is a reporting convention, not an exchange-calendar session model. In
particular, it does not infer overnight session boundaries from trading hours.

Result Data
===========

``result.daily`` contains session ``pnl``, account ``returns``, ``lreturn``,
``equity``, normalized ``balance``, and ``fixed_return``. ``result.df`` retains
the prepared transaction data and adds bar-level gross/net PnL, slippage,
equity, balance, standard drawdown, fixed-capital drawdown, and drawdown in
price points.

Statistic names use compact ``snake_case`` keys. Drawdown percentages are
negative, PnL values remain in price points, and durations are pandas timedeltas.

Typical Pattern
===============

.. code-block:: python

   from haymaker.research.backtester import no_stop, perf

   strategy = df.copy()
   strategy["signal"] = signal
   strategy["position"] = strategy["signal"].shift().fillna(0).astype(int)

   tx = no_stop(strategy, price_column="open")
   result = perf(tx, slippage=1, min_tick=0.25, capital=5_000)

The :doc:`examples` page links to notebooks with executed outputs.
