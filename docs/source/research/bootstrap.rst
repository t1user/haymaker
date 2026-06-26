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

The sampled rows are not raw prices. Haymaker first converts each generated
bar, meaning every row after the first input row, into OHLC log distances from
the previous close. If ``len(data)`` is 101, there are 100 prepared rows to
sample, indexed conceptually as ``0`` through ``99``. The bootstrap method
chooses a sequence of prepared-row positions, then reconstruction turns those
sampled rows back into an OHLC path.

Why use log distances instead of raw prices? Because a sampled bar should
describe a move relative to where the synthetic path is now, not copy an old
absolute price level. A log distance is:

.. code-block:: python

   log(current_value / previous_close)

For example, if yesterday's close was ``100`` and today's close is ``102``, the
prepared close value is ``log(102 / 100)``. If that prepared row is later
sampled when the synthetic previous close is ``250``, reconstruction applies
the same relative move to ``250`` instead of copying the old price ``102``.
This is why the generated path can drift away from the original price level
while still using observed bar shapes and returns.

The same idea is used for ``open``, ``high``, and ``low``. If the previous
close is ``100`` and the next bar is ``open=101``, ``high=104``, ``low=99``,
``close=102``, the prepared row stores:

.. code-block:: python

   open  = log(101 / 100)
   high  = log(104 / 100)
   low   = log(99 / 100)
   close = log(102 / 100)

That prepared row means: "relative to the previous close, this bar opened 1%
higher, traded roughly 4% above it, traded roughly 1% below it, and closed
roughly 2% above it." The exact values are logs, not simple percentages, because
logs compose cleanly when many returns are chained together.

Supported methods:

``"stationary"``
    Variable-length blocks. The generator usually continues to the next
    historical row, but sometimes jumps to a new random row. This is the
    default. With ``block_length=20``, the jump probability at each generated
    row is ``1 / 20``. Longer block lengths mean fewer jumps and longer
    historical runs. The stay probability is therefore ``19 / 20``.

``"moving"``
    Fixed-length blocks that must fit inside the original data. They do not
    wrap around the end of the sample. With ``block_length=20``, every sampled
    block is exactly 20 consecutive prepared rows, except the final block may
    be truncated when the synthetic path is already full.

``"circular"``
    Fixed-length blocks that may wrap from the end of the sample back to the
    beginning. This is like ``"moving"``, but a block starting near the end can
    continue from the beginning.

``block_length="auto"`` estimates a reasonable block length. If the optional
``arch`` package is installed, Haymaker asks ``arch`` for the estimate. Without
``arch``, it uses an internal autocorrelation cutoff fallback.

The jump/stay probability itself is not learned as a separate probability from
the original data. For stationary bootstrap, it is mechanically derived from
the resolved block length:

.. code-block:: python

   jump_probability = 1 / block_length
   stay_probability = 1 - jump_probability

So ``block_length=5`` means a 20% jump chance and an 80% stay/continue chance
at each generated row. ``block_length=100`` means a 1% jump chance and a 99%
stay/continue chance. Here "stay" means "stay in the current block and move to
the next historical row"; it does not mean repeat the same row. What can be
learned from the data is the block length when ``block_length="auto"`` is used.
With a manual integer ``block_length``, this probability is your modeling
choice.

The decision is made independently at every generated row. Haymaker draws a
fresh random number: if it falls below ``1 / block_length``, the next prepared
row comes from a new random historical position; otherwise the next prepared
row is the next historical position after the current one. This is the standard
stationary-bootstrap rule. It is not trying to detect "jump points" in the
historical sample. A new random historical position is chosen uniformly from
the prepared rows.

How Blocks Are Created
----------------------

For a tiny prepared sample with positions ``0, 1, 2, 3, 4, 5`` and
``block_length=3``, the methods behave like this:

``stationary``
    Starts at a random position. On each next generated row it either continues
    to the next position, wrapping if needed, or jumps to a fresh random
    position. One possible sampled position sequence is::

       5, 0, 1, 3, 1, 5, 0, 1, 2, 2

    Read this as blocks of uneven length: ``5, 0, 1`` then ``3`` then ``1``
    then ``5, 0, 1, 2`` then ``2``. The block boundaries are random.

``moving``
    Samples fixed blocks that fit inside the sample. With six prepared rows and
    block length three, valid starts are ``0``, ``1``, ``2``, and ``3``. One
    possible sampled position sequence is::

       3, 4, 5, 2, 3, 4, 2, 3, 4, 3

    The first three rows come from the block starting at ``3``. The next three
    come from the block starting at ``2``. Each new block start is chosen
    uniformly from the valid starts.

``circular``
    Samples fixed blocks but allows wrapping. With six prepared rows and block
    length three, a block can start at ``5`` and continue as ``5, 0, 1``. One
    possible sampled position sequence is::

       5, 0, 1, 3, 4, 5, 4, 5, 0, 5

    Each new block start is chosen uniformly from all prepared-row positions.

``block_length`` Choices
------------------------

``block_length`` can be either ``"auto"`` or a positive integer.

``"auto"``
    Good first choice. Haymaker estimates a block length from the prepared close
    log-distance series. It uses
    :func:`haymaker.research.optimal_block_length` and then applies the result
    to the selected method.

Positive integer
    Manual block length. Use this when you want the synthetic paths to preserve
    a particular amount of local sequence, for example about one trading
    session, one hour, or one signal lookback window.

Choosing a value is a tradeoff:

* Smaller values reshuffle more aggressively. This gives more variety but
  destroys more serial dependence.
* Larger values preserve more local structure. This is useful when volatility,
  trend, or session behavior lasts many bars, but very large values can make
  paths look too much like copied historical chunks.
* For ``"moving"``, the value cannot exceed the number of prepared rows,
  because every fixed block must fit inside the sample.
* For ``"circular"``, the value may exceed the number of prepared rows because
  blocks can wrap around.

What ``arch`` Does For ``"auto"``
---------------------------------

When ``arch`` is installed, Haymaker delegates the estimate to
``arch.bootstrap.optimal_block_length``. That function is from the optional
``arch`` package and implements an automatic block-length estimator for
dependent time-series bootstraps. In plain terms, it looks at how quickly
autocorrelation dies out and returns block-length recommendations for
stationary and circular-style bootstraps.

Autocorrelation means "does this row resemble recent previous rows?" If returns
or OHLC log distances are still related several bars later, a block should
usually be longer so the bootstrap does not break that dependence too quickly.
If dependence dies out quickly, a shorter block is usually enough.

The ``arch`` estimator is not guessing a magic profitable value. It is trying
to choose a statistically reasonable block size for dependent data. It inspects
autocovariances/autocorrelations, chooses a lag window where dependence has
mostly died down, and computes recommended block sizes for stationary and
circular bootstrap formulas. The result is data-dependent, but it is still an
estimate. It should be treated as a default starting point, not as proof that
one block length is correct for every strategy.

Haymaker then:

#. Selects the stationary recommendation for ``method="stationary"``.
#. Selects the circular recommendation for ``method="circular"``.
#. Also uses the circular recommendation for ``method="moving"``, because both
   moving and circular use fixed-size blocks.
#. Rounds the estimate up to a positive integer.
#. For ``method="moving"``, caps the automatic value at the number of prepared
   rows, because moving blocks cannot be longer than the available sample.
#. Falls back to Haymaker's internal estimator if ``arch`` is unavailable,
   errors, or returns an unsupported/non-finite result.

The internal fallback is intentionally simpler: it checks autocorrelation up to
about ``sqrt(n)`` lags, where ``n`` is the number of non-null observations. It
uses a noise band of ``2 / sqrt(n)`` and picks the first lag whose absolute
autocorrelation is inside that band. If no lag clears that band, it uses the
maximum checked lag. This is a conservative fallback, not a full replacement
for ``arch``.

For example, if the fallback checks lags ``1`` through ``10`` and lag ``4`` is
the first lag where autocorrelation is small enough to look like noise, the
estimated block length is ``4``. If none of the checked lags look like noise,
the estimate is ``10``.

One important detail: in ``bootstrap(df, block_length="auto")``, Haymaker
prepares the data first and calls ``optimal_block_length`` on the prepared
``close`` log-distance column, not on raw absolute prices. If you call
:func:`haymaker.research.optimal_block_length` yourself, you can pass either a
return series or a dataframe and choose the column to inspect.

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

State Helper Outputs
--------------------

State helpers return a :class:`pandas.Series` of labels. The labels are not
used as numbers. They are bucket names. During generation,
:func:`haymaker.research.regime_bootstrap` learns how often each bucket appears,
learns how buckets transition into each other, generates a synthetic bucket
path, and samples prepared OHLC rows from the matching bucket.

``return_states(data)``
    Labels each bar as ``"positive"`` or ``"negative"`` from close-to-close
    log return. Zero returns are labeled ``"positive"``. This is the default
    used by ``regime_bootstrap(data)``.

``trend_states(data, window=50)``
    Labels each bar as ``"up"`` or ``"down"`` from
    ``data["close"].pct_change(window)``. A larger ``window`` asks for a slower
    trend regime; a smaller window reacts faster.

``volatility_states(data, window=20, q=0.5)``
    Computes rolling close-to-close log-return volatility and labels bars as
    ``"low_vol"`` or ``"high_vol"``. ``q`` is the quantile threshold. With
    ``q=0.5``, roughly the lower half of volatility observations are
    ``"low_vol"`` and the upper half are ``"high_vol"``.

``range_states(data, q=0.5)``
    Labels each bar as ``"narrow_range"`` or ``"wide_range"`` from
    ``log(high / low)``. ``q`` is the quantile threshold.

``combine_states(first, second, ...)``
    Combines labels into tuple labels. For example, combining return and range
    states may produce buckets such as ``("positive", "narrow_range")`` and
    ``("negative", "wide_range")``. Rows where any input state is missing are
    dropped.

``hmm_states(data, n_states=...)``
    Fits a Gaussian hidden Markov model with ``hmmlearn`` and returns integer
    labels such as ``0``, ``1``, and ``2``. Use this only when you intentionally
    want an HMM-derived hard state model and have inspected the resulting state
    counts.

Here is a small example. Suppose the close prices are:

.. code-block:: python

   close = [100, 101, 100, 103, 102, 106]

Then the simple helpers produce labels like:

.. code-block:: python

   return_states(df)
   # ["positive", "positive", "negative", "positive", "negative", "positive"]

   trend_states(df, window=2)
   # ["up", "up", "up", "up", "up", "up"]

   volatility_states(df, window=2, q=0.5)
   # ["low_vol", "low_vol", "low_vol", "high_vol", "high_vol", "high_vol"]

   range_states(df, q=0.5)
   # [
   #     "narrow_range",
   #     "narrow_range",
   #     "wide_range",
   #     "narrow_range",
   #     "wide_range",
   #     "wide_range",
   # ]

Combining states creates more specific buckets:

.. code-block:: python

   combine_states(return_states(df), range_states(df))
   # [
   #     ("positive", "narrow_range"),
   #     ("positive", "narrow_range"),
   #     ("negative", "wide_range"),
   #     ("positive", "narrow_range"),
   #     ("negative", "wide_range"),
   #     ("positive", "wide_range"),
   # ]

The more helpers you combine, the more buckets you create. That can be useful,
but each bucket then has fewer historical rows and fewer observed transitions.
If buckets become too small, ``regime_bootstrap`` raises through
``min_state_count`` or ``min_transition_count`` instead of generating fragile
paths.

Shared OHLC Preparation
-----------------------

Both generator families use the same OHLC representation:

#. The first real ``close`` is kept as the anchor price.
#. For every later row, ``open``, ``high``, ``low``, and ``close`` are stored as
   log distances from the previous real close.
#. The generator samples those prepared rows.
#. Reconstruction applies sampled log distances to the previous synthetic close.
#. Reconstructed OHLC prices are rounded back to the tick grid inferred from
   the source OHLC prices.

This means the next synthetic bar is built from the previous synthetic close,
not by copying historical prices directly. It preserves the shape of sampled
bars while allowing the synthetic path to move away from the original price
level.

Tick rounding matters because the backtester estimates transaction costs from
minimum tick size. Without rounding, repeated log/exp reconstruction can create
prices such as ``100.2499999997`` or ``100.251381`` even when the original
instrument traded in ``0.25`` increments. Haymaker estimates the source tick
size from the OHLC price columns and rounds generated ``open``, ``high``,
``low``, and ``close`` values to that grid. If the source data does not contain
enough price variation to infer a tick size, no tick rounding is applied.

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
* Rounds reconstructed OHLC prices to the inferred source tick size when one
  can be inferred.
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

Parameter Examples
==================

Block bootstrap examples:

``bootstrap(df, paths=20, random_state=42)``
    Generates 20 stationary-bootstrap paths. The average block length is chosen
    automatically. This is the best first call when you want synthetic paths
    quickly and do not yet have a view on how long historical dependence should
    persist.

``bootstrap(df, method="stationary", block_length=10, paths=20)``
    Generates 20 paths where each generated row has a ``1 / 10`` chance of
    jumping to a fresh historical position. Expected blocks are shorter than
    with ``block_length=100``. The paths should be more shuffled and less tied
    to long historical runs.

``bootstrap(df, method="stationary", block_length=100, paths=20)``
    Generates 20 paths with fewer jumps and longer copied runs of prepared
    rows. This preserves more local trend and volatility clustering, but gives
    less variety.

``bootstrap(df, method="moving", block_length=50, paths=20)``
    Samples fixed 50-row blocks that must fit inside the source sample. Use
    this when you want every block to be a real contiguous slice, with no
    wrap-around from the end of the sample to the beginning.

``bootstrap(df, method="circular", block_length=50, paths=20)``
    Samples fixed 50-row blocks that may wrap. This gives every prepared row a
    chance to be near the beginning of a sampled block, including rows close to
    the end of the source sample.

Regime bootstrap examples:

``regime_bootstrap(df, paths=20, random_state=42)``
    Uses ``return_states(df)``. The generator learns transitions between
    positive-return and negative-return bars, then samples rows from the
    matching return-sign bucket.

``regime_bootstrap(df, states=volatility_states(df), paths=20)``
    Generates paths that switch between low-volatility and high-volatility
    buckets. High-volatility synthetic states sample rows from historically
    high-volatility bars.

``regime_bootstrap(df, states=combine_states(trend_states(df), volatility_states(df)))``
    Generates paths from composite buckets such as ``("up", "low_vol")`` and
    ``("down", "high_vol")``. This is more expressive, but each bucket has
    fewer rows. Check ``states.value_counts()`` before trusting the result.

``regime_bootstrap(df, states=hmm_states(df, n_states=3), paths=20)``
    Generates paths from three HMM-inferred integer states. Use this only after
    checking whether the HMM states are stable and economically meaningful.

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
