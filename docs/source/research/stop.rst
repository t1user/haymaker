===========
Stop Engine
===========

The stop engine sits between strategy generation and
:func:`haymaker.research.backtester.perf`.
It accepts a dataframe with either ``position`` or ``blip`` input, applies stop
logic, and emits the same transaction-frame schema consumed by the backtester.

It supports fixed and trailing stops, optional take-profit distance, stop
adjustment after favorable movement, bar-count time stops, scheduled close
times, and lazy before-session-close masks for intraday data.

Primary API
===========

.. autofunction:: haymaker.research.stop.stop_loss

.. autofunction:: haymaker.research.stop.before_close

.. autoclass:: haymaker.research.stop.BeforeClose
   :members:

Typical Pattern
===============

.. code-block:: python

   from haymaker.research.backtester import perf
   from haymaker.research.stop import stop_loss

   tx = stop_loss(
       strategy,
       distance=stop_distance,
       mode="trail",
       price_column="open",
   )
   result = perf(tx, slippage=1)

If ``distance`` is supplied as a series, it must already have the same index as
the input dataframe. Align or upsample it explicitly before calling
``stop_loss``.
