****************
Research Package
****************

The :mod:`haymaker.research` package is a lightweight layer for strategy
research with pandas dataframes. It is meant for fast iteration: generate a
hypothetical strategy state, convert it into transactions, apply optional stop
logic, and inspect performance without building a full live-trading pipeline.

The tools in this package deliberately stay close to ordinary
:class:`pandas.DataFrame` and :class:`pandas.Series` objects. That makes it easy
to mix Haymaker with external data sources, indicator libraries, plotting
tools, and notebooks.

.. toctree::
   :maxdepth: 2

   overview
   signals
   backtester
   stop
   upsampling
   bootstrap
   examples

