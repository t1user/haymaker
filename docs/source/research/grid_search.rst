==========
GridSearch
==========

Purpose
=======

``haymaker.research.grid_search`` provides a small grid-search workflow for
running many dataframe backtests and comparing their results. It does not choose
an optimum automatically. It builds a set of simulations, runs each simulation
through :func:`haymaker.research.backtester.perf`, and returns a
``GridSearchResult`` with statistic tables and raw backtest outputs.

Basic Usage
===========

The three supported construction paths all return a ``GridSearch`` object.
Call ``run()`` to execute the simulations and get a ``GridSearchResult``.

Progressions
------------

Use ``GridSearch.from_progressions`` when you want a Cartesian product of two
parameter progressions:

.. code-block:: python

   from haymaker.research.grid_search import GridSearch, show_grid_table

   search = GridSearch.from_progressions(
       df,
       strategy_func,
       [(0.5, 0.5, "lin"), (30, 10, "lin")],
       param_names=("threshold", "lookback"),
       fixed_kwargs={"bias": 0},
       multiprocess=False,
   )

   result = search.run()
   show_grid_table(result)

Each progression is ``(start, step, mode)``. ``mode="lin"`` creates ten values
by adding ``step``. ``mode="geo"`` creates ten values by multiplying by
``step``. If ``start`` is itself a sequence, those explicit values are used
directly and ``step``/``mode`` are ignored.

For example, this uses the exact first-axis values ``0.5``, ``1.0``, and
``2.0`` instead of generating ten values:

.. code-block:: python

   search = GridSearch.from_progressions(
       df,
       strategy_func,
       [((0.5, 1.0, 2.0), None, "lin"), (30, 10, "lin")],
       param_names=("threshold", "lookback"),
   )

Pairs
-----

Use ``GridSearch.from_pairs`` when you already know the exact pairs to test:

.. code-block:: python

   from haymaker.research.grid_search import GridSearch

   search = GridSearch.from_pairs(
       df,
       strategy_func,
       [(0.5, 30), (1.0, 40), (1.5, 50)],
       param_names=("threshold", "lookback"),
   )

   result = search.run()

The result keys are the pairs themselves, so ``result.tables["annual_return"]``
is indexed by the first searched value and has columns for the second searched
value.

Dataframes
----------

Use ``GridSearch.from_dfs`` when you want to run one fixed parameter tuple
against many dataframes, for example bootstrap paths or different sample
periods:

.. code-block:: python

   from haymaker.research.grid_search import GridSearch

   search = GridSearch.from_dfs(
       {"sample_a": df_a, "sample_b": df_b},
       strategy_func,
       params=(0.5, 30),
       param_names=("threshold", "lookback"),
   )

   result = search.run()

Mapping keys become the first part of the result key. If you pass a sequence of
dataframes instead of a mapping, integer positions are used as labels.

Function Contract
=================

The strategy function receives the selected input data as its first argument and
must return a transaction dataframe accepted by
:func:`haymaker.research.backtester.perf`.

By default, GridSearch passes ``df["close"]`` as the first argument. Set
``pass_full_df=True`` when the function needs the complete dataframe.

Searched parameters are passed positionally unless ``param_names`` is provided.
``fixed_kwargs`` are included in every simulation call and are the recommended
way to supply parameters that are not part of the search. If a searched
``param_names`` entry overlaps with ``fixed_kwargs``, GridSearch raises
``ValueError``.

GridSearch API
==============

``GridSearch`` is the simulation plan. It does not run anything until
``run()`` is called.

``GridSearch.from_progressions(df, func, progressions, ...)``
    Builds a two-axis parameter grid from two progression specs. This is the
    usual 10 by 10 heatmap workflow.

``GridSearch.from_pairs(df, func, pairs, ...)``
    Builds simulations from exact parameter pairs. Use this when the parameter
    combinations are irregular or preselected.

``GridSearch.from_dfs(dfs, func, params, ...)``
    Runs one parameter tuple against many dataframes. Use this for bootstrap
    samples, different time windows, or different instruments prepared into the
    same dataframe shape.

``GridSearch.run()``
    Executes every simulation and returns ``GridSearchResult``.

Common constructor options:

``param_names``
    Names for the searched parameters. If provided, searched values are passed
    as keyword arguments. If omitted, searched values are passed positionally.

``fixed_kwargs``
    Keyword arguments passed to every strategy call. These must not overlap
    with ``param_names``.

``pass_full_df``
    By default the strategy receives ``df["close"]``. Set this to ``True`` when
    the strategy needs the whole dataframe.

``multiprocess``
    Runs simulations in separate processes when ``True``. Set to ``False`` for
    easier debugging or notebook work with non-picklable callables.

``save_mem``
    Omits daily return data and enriched bar-level data from the result. This
    keeps memory use lower but disables ``returns``, ``log_returns``, ``paths``,
    ``corr``, ``rank``, and combined-return helpers.

``**perf_kwargs``
    Extra keyword arguments forwarded to
    :func:`haymaker.research.backtester.perf`.

GridSearchResult
================

``GridSearch.run()`` returns ``GridSearchResult``. The main access path is:

.. code-block:: python

   result.stats_frame
   result.tables["annual_return"]
   result.tables["sharpe_ratio"]

``stats_frame`` shows every statistic returned by ``perf`` in one dataframe.
Rows are the original statistic names and columns are simulations. Parameter
searches use the parameter pair as the column label. Searches created with
``from_dfs`` use only the dataframe label, because the searched parameter tuple
is the same for every column.

For plotting compatibility, statistic tables are also available as dynamic
attributes:

.. code-block:: python

   result.annual_return
   result.sharpe_ratio

The result also stores raw backtester outputs:

``raw_stats``
    Statistics from each call to ``perf``.

``raw_dailys``
    Daily return data from each call to ``perf``.

``raw_positions``
    Position/trade records from each call to ``perf``.

``raw_dfs``
    Enriched bar-level performance data from each call to ``perf``.

``raw_warnings``
    Backtester warnings keyed by simulation key.

Convenience properties:

``stats_frame``
    All ``perf`` statistics in one dataframe, with statistic names as rows and
    simulations as columns.

``returns``
    Daily simple returns for every simulation, arranged by simulation key.

``log_returns``
    Daily log returns for every simulation.

``paths``
    Daily balance paths for every simulation.

``corr``
    Correlation matrix of simulation log returns.

``rank``
    Top 20 simulations by final balance minus one.

``return_mean`` and ``return_median``
    Formatted summaries of non-zero annual returns.

``warnings``
    Only the non-empty warning lists from ``raw_warnings``.

When ``save_mem=True``, daily return data and bar-level data are omitted. In
that mode, ``returns``, ``log_returns``, ``paths``, ``corr``, and ``rank`` raise
``ValueError``.

Display Helpers
===============

Existing display helpers accept ``GridSearchResult``:

.. code-block:: python

   from haymaker.research.grid_search import plot_grid, show_grid, show_grid_table

   fig = plot_grid(result)
   show_grid(result)
   show_grid_table(result)

``plot_grid(result, fields=("annual_return", "sharpe_ratio"))``
    Creates the 10 by 10, two-field heatmap layout and returns the Matplotlib
    figure. ``plot_grid`` expects both selected statistic tables to be 10 by 10,
    so it is meant for ``from_progressions`` output using two ten-value axes.

``show_grid(result, fields=..., plotting_function=plot_grid)``
    Notebook helper that displays the figure and closes it with
    ``plt.close(fig)`` afterwards. Use this instead of manually calling
    ``display(fig)`` when you do not want the figure to remain open in
    Matplotlib state.

``show_grid_table(result, fields=None)``
    Displays statistic tables as plain formatted notebook tables. ``fields=None``
    displays the preferred default fields. Passing a string displays
    ``annual_return`` plus that field. Passing a sequence displays exactly those
    fields.

Combined Portfolio Analytics
============================

Combined analytics are methods on ``GridSearchResult`` over selected simulation
keys:

.. code-block:: python

   keys = [(0.5, 30), (1.0, 40), (1.5, 50)]
   returns = result.combined_returns(keys)
   path = result.combined_path(keys)

The combined return stream represents a daily-rebalanced equal-weight portfolio
of completed simulation return streams. It is not a directly simulated strategy.

The default missing-data policy is ``missing="zero"``, which treats missing
sleeve returns as idle cash. Use ``missing="raise"`` to reject missing selected
returns, or ``missing="drop"`` to average over available sleeves only.

``result.combined_returns(keys, missing="zero")``
    Returns the selected equal-weight daily return stream.

``result.combined_path(keys, missing="zero")``
    Returns ``(result.combined_returns(...) + 1).cumprod()``.

``result.combined_stats(keys, missing="zero")``
    Returns the standard lean metric set for the combined return stream.

The module-level ``combined_returns(result, keys)``,
``combined_path(result, keys)``, and ``combined_stats(result, keys)`` functions
remain available as convenience wrappers around the result methods.

Lower-Level Helpers
===================

``GridSearch.progression(spec)``
    Converts one progression spec into values. Most callers should use
    ``from_progressions`` instead.

``GridSearch.get_pairs(sp_1, sp_2)``
    Returns the Cartesian product of two parameter sequences. Most callers
    should use ``from_progressions`` or ``from_pairs`` instead.

API Reference
=============

.. autoclass:: haymaker.research.grid_search.GridSearch
   :members:

.. autoclass:: haymaker.research.grid_search.GridSearchResult
   :members:

.. autofunction:: haymaker.research.grid_search.combined_returns

.. autofunction:: haymaker.research.grid_search.combined_path

.. autofunction:: haymaker.research.grid_search.combined_stats

.. autofunction:: haymaker.research.grid_search.plot_grid

.. autofunction:: haymaker.research.grid_search.show_grid

.. autofunction:: haymaker.research.grid_search.show_grid_table
