===========================
Signals, Blips, Positions
===========================

The research package uses a small vocabulary for binary trading state. Keeping
these terms separate is important because they describe different moments in
the strategy lifecycle.

``indicator``
    A computed input to a strategy, such as a moving average, volatility
    estimate, breakout level, or any other dataframe column.

``signal``
    The desired strategy state after seeing a completed bar. A signal is
    information, not yet an executable position.

``blip``
    A sparse event on the bar where the strategy learns that it wants an
    action. A blip is usually ``0`` and becomes ``1`` or ``-1`` only when a
    trade event is generated.

``close_blip``
    A sparse close event. When supplied together with ``blip``, it allows a
    strategy to distinguish opening events from closing events.

``transaction``
    The bar where an action is executed.

``position``
    The executable or held state after timing conversion. A position is already
    aligned to the bar where the position exists.

For binary strategies, ``signal``, ``blip``, ``close_blip``, ``transaction``,
and ``position`` use ``-1`` for short, ``0`` for flat, and ``1`` for long.

Timing Convention
=================

Research signals are normally generated after a bar is complete. If a signal is
computed from the current bar close, it cannot be executed at that same bar
open. Convert it to an executable position with an explicit timing shift, for
example with :func:`haymaker.research.sig_pos`.

Blips are generated events. They should be recorded where the information
becomes known. Functions that accept blips document whether they perform the
execution timing conversion internally.

Positions are different: they are already executable state. Do not pass shifted
positions into helpers that expect generated signals or events.

Conversion Helpers
==================

.. autofunction:: haymaker.research.sig_pos

.. autofunction:: haymaker.research.sig_blip

.. autofunction:: haymaker.research.blip_sig

.. autofunction:: haymaker.research.pos_trans

.. autofunction:: haymaker.research.pos_trans_numpy

