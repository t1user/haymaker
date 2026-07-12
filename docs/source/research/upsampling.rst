=======================
Upsampling and Grouping
=======================

Strategies are often easier to define on bars that are not simple clock-time
bars. For example, a strategy may generate cleaner signals on volume bars while
execution and mark-to-market should still happen on the original higher
frequency data.

The typical workflow is:

#. Build grouped bars from the source dataframe.
#. Generate ``signal`` or ``blip`` columns on the grouped dataframe.
#. Use :func:`haymaker.research.upsample` to align generated information back
   to the source dataframe.
#. Convert the upsampled dataframe to transactions with
   :func:`haymaker.research.backtester.no_stop` or
   :func:`haymaker.research.stop.stop_loss`.

Grouped lower-frequency values become available when their grouped bar is
complete. Ordinary columns are propagated from that availability point onward.
Sparse event columns are kept sparse.

Primary API
===========

.. autofunction:: haymaker.research.upsample

.. autofunction:: haymaker.research.numba_tools.volume_grouper

Typical Pattern
===============

.. code-block:: python

   from haymaker.research import upsample
   from haymaker.research.numba_tools import volume_grouper

   bars = volume_grouper(df, target=target_volume, field="volume", label="left")
   bars["signal"] = grouped_signal
   bars["blip"] = sig_blip(bars["signal"])

   upsampled = upsample(
       df[["open", "high", "low", "close", "volume", "barCount"]],
       bars[["signal", "blip"]],
       label="left",
   )

Do not pass ``position`` into :func:`haymaker.research.upsample`. A position is
already executable state. Upsample generated signals or blips first, then derive
or execute the resulting state on the upsampled dataframe.
