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

Use ``GridSearch.from_pairs`` when you already know the exact pairs to test:

.. code-block:: python

   search = GridSearch.from_pairs(
       df,
       strategy_func,
       [(0.5, 30), (1.0, 40), (1.5, 50)],
       param_names=("threshold", "lookback"),
   )

Use ``GridSearch.from_dfs`` when you want to run one fixed parameter tuple
against many dataframes, for example bootstrap paths or different sample
periods:

.. code-block:: python

   search = GridSearch.from_dfs(
       {"sample_a": df_a, "sample_b": df_b},
       strategy_func,
       params=(0.5, 30),
       param_names=("threshold", "lookback"),
   )

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

Results
=======

``GridSearch.run()`` returns ``GridSearchResult``. The main access path is:

.. code-block:: python

   result.tables["annual_return"]
   result.tables["sharpe_ratio"]

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

``plot_grid`` keeps its original 10 by 10, two-field heatmap layout.
``show_grid_table`` displays plain formatted notebook tables.

Combined Portfolio Analytics
============================

Combined analytics are explicit functions over selected simulation keys:

.. code-block:: python

   from haymaker.research.grid_search import combined_path, combined_returns

   keys = [(0.5, 30), (1.0, 40), (1.5, 50)]
   returns = combined_returns(result, keys)
   path = combined_path(result, keys)

The combined return stream represents a daily-rebalanced equal-weight portfolio
of completed simulation return streams. It is not a directly simulated strategy.

The default missing-data policy is ``missing="zero"``, which treats missing
sleeve returns as idle cash. Use ``missing="raise"`` to reject missing selected
returns, or ``missing="drop"`` to average over available sleeves only.

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
