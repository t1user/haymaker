==================
Data Bootstrapping
==================

Data bootstrapping creates synthetic OHLC price paths from a real historical
dataframe. "Synthetic" means the path is artificial, but its pieces come from
patterns observed in the source data. In ELI5 terms, it cuts history into
price-move cards, shuffles those cards in controlled ways, and then rebuilds a
new price story.

In Haymaker this is part of :mod:`haymaker.research`, not the live trading
runtime. The generated paths are ordinary :class:`pandas.DataFrame` objects that
can be used like historical data in research notebooks, strategy experiments,
or backtester runs.

What Problem Are We Solving?
============================

Historical market data is limited. If a strategy is developed, tuned, and
judged on the same exact path, it can accidentally learn quirks of that path
instead of a durable trading idea. That is over-fitting: the strategy looks
clever on the data you showed it, but fails when the next unseen market period
arrives.

Bootstrap data helps with the middle part of research:

* It gives more test paths than the one historical path you started with.
* It keeps strategy development from repeatedly peeking at the final holdout
  period.
* It lets you ask "does this idea survive many plausible reshufflings of the
  same market behavior?"
* It can reveal fragile strategies whose result depends on one lucky ordering
  of trades.

The recommended split is simple:

#. Use a training or development slice to design the idea.
#. Generate bootstrap paths from that development slice.
#. Use those artificial paths to stress the idea and reject fragile versions.
#. Save truly unseen real data for the final test.

The final real-data test still matters because bootstrap paths are not new
market history. They are new arrangements of information extracted from old
history.

Risks and Limits
================

Bootstrap data is useful, but it is not magic.

Will outcomes be the same as real future outcomes?
    No. Bootstrap paths estimate "what could have happened if observed
    historical moves appeared in a different order." Future markets can contain
    shocks, liquidity changes, volatility regimes, correlations, and execution
    behavior that are not in the source sample.

Are all possible scenarios covered?
    No. The generator can only resample from what it has seen. If the source
    data has no crash-like bars, no overnight gaps, no high-volatility period,
    or no low-liquidity session, the generated paths will not invent a complete
    version of those missing cases.

Are we tainting the original data?
    Yes, in the statistical sense. Synthetic paths are derived from the source
    data, so they are not independent evidence. If the same real period is used
    for design, bootstrap generation, parameter selection, and final reporting,
    the final result is still contaminated by repeated exposure to that period.
    Keep a separate real holdout set for the final check.

Can bootstrap data hide bad assumptions?
    Yes. A method may preserve some properties while damaging others. For
    example, block bootstrap keeps short local sequences but can break longer
    macro structure. Regime bootstrap keeps user-defined state behavior but is
    only as good as the state labels.

How Haymaker Generates Data
===========================

Haymaker currently provides two bootstrap families.

Block Bootstrap
---------------

:func:`haymaker.research.bootstrap` resamples rows in blocks. Blocks are used
because single-bar shuffling destroys too much market structure. Nearby bars
often belong together: momentum, volatility clustering, and intraday flow all
depend on sequence.

Supported methods:

``"stationary"``
    Variable-length blocks. The generator usually continues to the next
    historical row, but sometimes jumps to a new random row. This is the
    default.

``"moving"``
    Fixed-length blocks that must fit inside the original data. They do not
    wrap around the end of the sample.

``"circular"``
    Fixed-length blocks that may wrap from the end of the sample back to the
    beginning.

``block_length="auto"`` estimates a reasonable block length. If the optional
``arch`` package is installed, Haymaker asks ``arch`` for the estimate. Without
``arch``, it uses an internal autocorrelation cutoff fallback.

Regime Bootstrap
----------------

:func:`haymaker.research.regime_bootstrap` first follows a Markov-style state
path, then samples historical rows from matching state buckets. A state is a
hard label such as ``"up"``, ``"down"``, ``"high_vol"``, ``"low_vol"``, or a
tuple like ``("up", "high_vol")``.

This is useful when you want generated paths to respect a simple market
condition process. For example, high-volatility bars should mostly be sampled
while the synthetic state path is in a high-volatility state.

If you omit ``states``, Haymaker uses a deliberately simple default:
:func:`haymaker.research.return_states`, which labels bars as positive or
negative from close-to-close log returns. This makes
``regime_bootstrap(df)`` useful out of the box without pretending to infer the
"true" market regimes.

For more controlled experiments, provide labels with a :class:`pandas.Series`,
or build them with the simple helper functions:

* :func:`haymaker.research.trend_states`
* :func:`haymaker.research.volatility_states`
* :func:`haymaker.research.range_states`
* :func:`haymaker.research.return_states`
* :func:`haymaker.research.combine_states`
* :func:`haymaker.research.hmm_states`

``hmm_states()`` is only a convenience wrapper around ``hmmlearn``. It requires
the optional ``hmmlearn`` dependency and should not be treated as a universal
regime detector.

Shared OHLC Preparation
-----------------------

Both generator families use the same OHLC representation:

#. The first real ``close`` is kept as the anchor price.
#. For every later row, ``open``, ``high``, ``low``, and ``close`` are stored as
   log distances from the previous real close.
#. The generator samples those prepared rows.
#. Reconstruction applies sampled log distances to the previous synthetic close.

This means the next synthetic bar is built from the previous synthetic close,
not by copying historical prices directly. It preserves the shape of sampled
bars while allowing the synthetic path to move away from the original price
level.

``volume`` and ``barCount`` are sampled as raw bar attributes when present.
Unknown columns, including ``average``, are dropped unless you explicitly pass
them in ``raw_columns`` and accept the semantics of raw sampling for those
columns.

Input and Output Contract
=========================

Input data must be an OHLC dataframe with:

* ``open``, ``high``, ``low``, and ``close`` columns.
* At least two rows.
* A sorted increasing index.
* Positive OHLC prices, because log distances are used.

Every generator returns ``list[pandas.DataFrame]``. This is true even when
``paths=1``.

Each generated path:

* Has length ``len(data) - 1``.
* Uses index ``data.index[1:]``.
* Starts reconstruction from ``data["close"].iloc[0]``.
* Contains reconstructed ``open``, ``high``, ``low``, and ``close`` columns.
* Contains available raw columns from ``raw_columns``.

Ready-Made One-Liners
=====================

For basic block bootstrap, this is the preferred one-liner:

.. code-block:: python

   from haymaker.research.bootstrap import bootstrap

   paths = bootstrap(df, paths=20, random_state=42)

That uses the defaults: ``method="stationary"`` and
``block_length="auto"``. The result is a list, so the first synthetic dataframe
is ``paths[0]``.

If you want one path only:

.. code-block:: python

   synthetic_df = bootstrap(df, random_state=42)[0]

If you want a fixed block length:

.. code-block:: python

   paths = bootstrap(df, block_length=100, paths=20, random_state=42)

If you want fixed non-wrapping blocks:

.. code-block:: python

   paths = bootstrap(df, method="moving", block_length=100, paths=20)

For regime bootstrap, the default one-liner is:

.. code-block:: python

   from haymaker.research.bootstrap import regime_bootstrap

   paths = regime_bootstrap(df, paths=20, random_state=42)

That uses ``return_states(df)`` internally. If you want explicit state labels,
use a compact helper-based call:

.. code-block:: python

   from haymaker.research.bootstrap import (
       combine_states,
       regime_bootstrap,
       trend_states,
       volatility_states,
   )

   paths = regime_bootstrap(
       df,
       states=combine_states(trend_states(df), volatility_states(df)),
       paths=20,
       random_state=42,
   )

For a production-quality research notebook, keep the state-building step
separate so you can inspect state counts before generating data:

.. code-block:: python

   states = combine_states(
       trend_states(df, window=80),
       volatility_states(df, window=40),
   )
   print(states.value_counts())

   paths = regime_bootstrap(df, states=states, paths=20, random_state=42)

Parameter Guide
===============

``data``
    Source OHLC dataframe. The generator samples behavior from this dataframe,
    so use a development slice, not the final holdout period.

``paths``
    Number of synthetic dataframes to return. Always returns a list.

``random_state``
    Integer seed, :class:`numpy.random.Generator`, or ``None``. Set an integer
    for reproducible notebooks and tests.

``raw_columns``
    Columns copied by raw row sampling instead of transformed as prices.
    Defaults to ``("volume", "barCount")``.

``method``
    Block-bootstrap method: ``"stationary"``, ``"moving"``, or
    ``"circular"``.

``block_length``
    Average block length for ``"stationary"``, fixed block length for
    ``"moving"`` and ``"circular"``, or ``"auto"``.

``states``
    Optional regime-bootstrap state labels. If omitted, defaults to
    ``return_states(data)``. If supplied, the index must cover
    ``data.index[1:]`` and every generated bar must have one non-null hashable
    label.

``min_state_count``
    Minimum number of rows required per regime state. This protects against
    sampling from tiny buckets.

``min_transition_count``
    Minimum number of outgoing transitions required per regime state. This
    protects against a transition matrix built from too little evidence.

Practical Workflow
==================

#. Start with a clean OHLC dataframe.
#. Keep a real holdout period out of the bootstrap source data.
#. Generate 10-100 paths with a fixed ``random_state`` while developing.
#. Run the same strategy and backtester flow on each path.
#. Look for stability across paths, not one best synthetic result.
#. Only after the idea survives this process, run the final test on unseen real
   data.

API Reference
=============

Block bootstrap:

.. autofunction:: haymaker.research.bootstrap

.. autofunction:: haymaker.research.optimal_block_length

Regime bootstrap:

.. autofunction:: haymaker.research.regime_bootstrap

State helpers:

.. autofunction:: haymaker.research.combine_states

.. autofunction:: haymaker.research.trend_states

.. autofunction:: haymaker.research.volatility_states

.. autofunction:: haymaker.research.range_states

.. autofunction:: haymaker.research.return_states

.. autofunction:: haymaker.research.hmm_states

Data preparation:

.. autofunction:: haymaker.research.prepare_bootstrap_frame
