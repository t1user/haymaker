Updating strategies and saved state
===================================

Use :doc:`terminology` for the supported vocabulary and :doc:`execution` for
composition examples. Strategies from ``dev`` need to be adapted to the
component APIs and their execution model names before startup. Preserve each
strategy's identity as its ``signal_key`` when converting existing positions.
For converted signal positions, use the execution model name
``legacy:<signal_key>`` and recovery-compatible protection configuration.

A custom execution model declares ``mode = ExecutionMode.SIGNAL`` or
``ExecutionMode.CONTRACT`` and calls the base initializer. Custom roll
executors declare the matching mode. Import ``ExecutionMode`` from
``haymaker.components``. Query filled positions through
``book.positions.for_signal`` or ``book.positions.for_contract``; accepted
targets and protection are exposed separately through ``book.targets`` and
``book.protection``.

Follow the settled-source, dry-run and fresh-target workflow in
:ref:`state-conversion`. The standalone converter reads ``dev`` strategy
snapshots, orders and blotter records. It preserves broker identifiers, fill
and commission evidence, signal position identity and saved protection inputs.
It reconstructs contract positions through Book restoration and verifies the
complete target readback. It refuses component Book records as source input.
