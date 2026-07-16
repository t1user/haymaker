=========
Bootstrap
=========

The bootstrap tools generate synthetic OHLC dataframes for research workflows.
They accept ordinary OHLC data and return a list of synthetic dataframes.

The shared preparation step stores ``open``, ``high``, ``low``, and ``close``
as log distances from the previous close. Reconstruction then anchors each
sampled row to the previous synthetic close, while raw columns such as
``volume`` and ``barCount`` are sampled as observed bar attributes.

Block Bootstrap
===============

.. autofunction:: haymaker.research.bootstrap

.. autofunction:: haymaker.research.optimal_block_length

Regime Bootstrap
================

.. autofunction:: haymaker.research.regime_bootstrap

State Helpers
=============

These helpers produce state series that can be passed directly to
:func:`haymaker.research.regime_bootstrap`.

.. autofunction:: haymaker.research.combine_states

.. autofunction:: haymaker.research.trend_states

.. autofunction:: haymaker.research.volatility_states

.. autofunction:: haymaker.research.range_states

.. autofunction:: haymaker.research.return_states

.. autofunction:: haymaker.research.hmm_states

Data Preparation
================

.. autofunction:: haymaker.research.prepare_bootstrap_frame

Typical Pattern
===============

.. code-block:: python

   from haymaker.research.bootstrap import bootstrap

   paths = bootstrap(
       df,
       method="stationary",
       block_length="auto",
       paths=3,
       random_state=42,
   )

For regime-based generation, provide one state label for every generated bar:

.. code-block:: python

   from haymaker.research.bootstrap import (
       combine_states,
       regime_bootstrap,
       trend_states,
       volatility_states,
   )

   states = combine_states(
       trend_states(df, window=80),
       volatility_states(df, window=40),
   )
   paths = regime_bootstrap(df, states=states, paths=3, random_state=7)

