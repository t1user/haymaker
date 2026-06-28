***************
Research Module
***************

The :mod:`haymaker.research` package is a lightweight layer for strategy
research with pandas dataframes. It is meant for fast iteration: generate a
hypothetical strategy state, convert it into transactions, apply optional stop
logic, and inspect performance without building a full live-trading pipeline.

The tools in this package deliberately stay close to ordinary
:class:`pandas.DataFrame` and :class:`pandas.Series` objects. That makes it easy
to mix Haymaker with external data sources, indicator libraries, plotting
tools, and notebooks.

Use this section as the public guide to the research module. The pages below
explain the core workflow, the timing vocabulary, the backtester and stop
engine, frequency alignment, and synthetic data generation.

.. toctree::
   :maxdepth: 2

   overview
   signals
   backtester
   stop
   upsampling
   grid_search
   bootstrap
   examples
