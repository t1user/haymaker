==========
Backtester
==========

The backtester is the final step in the basic research workflow. It consumes a
transaction dataframe and returns performance statistics, daily returns,
trade-level records, an enriched bar dataframe, and warnings.

For strategies without stop logic, use :func:`haymaker.research.no_stop` to
convert a dataframe containing ``position`` or ``blip`` into the transaction
schema expected by :func:`haymaker.research.perf`.

For notebook work, :func:`haymaker.research.auto_perf` is a convenience wrapper
that accepts either an already prepared transaction dataframe or a raw strategy
dataframe that can be passed through :func:`haymaker.research.no_stop`.

Primary API
===========

.. autofunction:: haymaker.research.no_stop

.. autofunction:: haymaker.research.perf

.. autofunction:: haymaker.research.auto_perf

.. autoclass:: haymaker.research.Results
   :members:

Typical Pattern
===============

.. code-block:: python

   from haymaker.research import no_stop, perf

   strategy = df.copy()
   strategy["signal"] = signal
   strategy["position"] = strategy["signal"].shift().fillna(0).astype(int)

   tx = no_stop(strategy, price_column="open")
   result = perf(tx, slippage=1)

The :doc:`examples` page links to notebooks with executed outputs.

