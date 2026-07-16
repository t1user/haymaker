========
Overview
========

Purpose
=======

The research package provides lightweight wrappers around pandas dataframes for
testing hypothetical trading strategies. The central workflow is:

#. Start with OHLC data in a dataframe.
#. Add strategy information such as ``signal``, ``blip``, or ``position``.
#. Convert the strategy dataframe into the transaction schema expected by
   :func:`haymaker.research.perf`, either directly with
   :func:`haymaker.research.no_stop` or through
   :func:`haymaker.research.stop.stop_loss`.
#. Inspect performance statistics, trade records, enriched bar-level results,
   and path plots.

This is intentionally lighter than the live execution framework. The research
package is designed for answering questions like "what would this rule have
done on this dataframe?" while preserving explicit timing semantics.

Dataframe First
===============

Most research functions accept and return pandas objects. There is no special
data container for ordinary strategy research. A typical input dataframe has at
least:

``open``, ``high``, ``low``, ``close``
    Price columns used for execution, stop logic, and mark-to-market.

``volume`` and ``barCount``
    Optional bar attributes used by utilities such as bootstrap generation or
    custom grouping.

``signal``, ``blip``, ``close_blip``, ``position``
    Optional strategy columns. Their meaning is described in
    :doc:`signals`.

Few Indicators Are Included
===========================

The package contains very few actual indicators. That is deliberate. Many
indicator packages already operate directly on pandas objects, and many common
indicators are only a few lines of pandas code. The research package focuses on
the mechanics around strategy timing, transaction generation, stop handling,
performance calculation, frequency alignment, and synthetic data generation.

As the cleaned-up utility and indicator modules stabilize, they can be added to
this documentation tree as separate pages without changing the core research
workflow described here.

Core Areas
==========

Backtester
    Convert strategy columns into transaction frames and calculate performance.

Stop engine
    Apply fixed or trailing stops, take profit, bar-count time stops, and
    scheduled closes.

Upsampling and grouping
    Generate signals on custom bars such as volume bars, then align generated
    information back to a higher-frequency execution dataframe.

Bootstrap
    Generate synthetic OHLC paths using block bootstrap or regime/state-based
    empirical sampling.

