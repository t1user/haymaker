Updating strategy names and saved state
=======================================

Use :doc:`terminology` for the supported vocabulary. Update strategy Python and
convert saved accounting together before starting the new runtime. The runtime
accepts only Book schema version 1; it has no old-keyword or import aliases.

.. list-table:: Strategy edits
   :header-rows: 1

   * - Previous interface
     - Supported interface
   * - ``source_key``; ``Portfolio(sources=...)``; ``expected_sources``
     - ``signal_key``; ``Portfolio(signal_keys=...)``; ``expected_signal_keys``
   * - ``PositionProposal``; ``PositionIntent``
     - ``PositionRequest``; ``TargetIntent``
   * - ``contract_blueprint``; registry ``blueprint_*`` and ``register_blueprint``
     - ``contract_spec``; registry ``contract_spec_*`` and ``register_contract_spec``
   * - ``positions_for_blueprint``
     - ``positions_for_contract_spec``
   * - ``book.positions.for_source``; ``source_states``; ``source_states_for_contract``
     - ``for_signal``; ``by_signal``; ``signal_positions_for_contract``
   * - ``book.positions.quantity(contract)``; position ``quantity``
     - ``book.positions.filled_quantity(contract)``; position ``filled_quantity``
   * - ``PositionState``
     - ``SignalPosition``, ``SignalTarget`` and ``ProtectionState`` through their Book queries
   * - ``TargetState``
     - ``ContractTarget``; reset cutoffs remain internal to Book
   * - ``FutureRollMode.BRACKET``; ``BracketFutureRollExecutor``
     - ``ExecutionMode.SIGNAL``; ``SignalFutureRollExecutor``
   * - ``FutureRollMode.DIRECT``; ``DirectFutureRollExecutor``
     - ``ExecutionMode.CONTRACT``; ``ContractFutureRollExecutor``

Update imports, constructor keywords, Portfolio membership declarations, query
calls and custom roll executors' ``mode`` declarations. Keep ``contract=...``,
``Atom.contract``, ``which_contract``, ``BracketExecutionModel`` and
``SerialTargetExecutionModel``. Signal key values and position IDs retain their
identity. Query accepted targets with ``book.targets.for_signal`` or
``for_contract`` and current protection with ``book.protection.for_signal``;
these fields no longer belong to the public position record.

Import ``ExecutionMode`` from ``haymaker.components``. A custom execution model
declares its ``mode`` as a class attribute and calls the base initializer.
Custom roll executors declare the same enum in their existing ``mode``
attribute. If registering an executor explicitly, call
``controller.future_roller.register_executor(executor)`` after constructing a
model; omit the executor to use the default. Schema version 1 rolls already
storing ``SIGNAL`` or ``CONTRACT`` need no conversion for this API change.

For custom accounting integrations, use Book's coordinated mutation methods.
``create_signal_position`` starts an independently managed position;
``close_signal_position`` closes its accounting record.
``correct_signal_position`` aligns filled and desired quantities, while
preserving direction blocking and fill checkpoints. Never replace recovery
envelopes or call collection-private mutation hooks.

Saved state
-----------

Follow the settled-source, dry-run and fresh-target workflow in
:ref:`state-conversion`. The standalone converter supports legacy strategy
snapshots and the preceding component schema. It preserves explicit execution
owner names, IB identifiers, fill evidence, corrections, checkpoints, protection
inputs and reset cutoffs. It rebuilds derived contract positions through Book
restoration and verifies the complete target readback. Unfinished old roll work
must be completed before conversion. Historical completed executor names can
remain historical values; new default names follow the supported executor classes.

The collections remain ``orders``, ``state`` and ``blotter``. Framework-owned
state names change as follows:

.. list-table:: State document names
   :header-rows: 1

   * - Previous identity / state type
     - Schema version 1 identity / state type
   * - ``position:{source_key}`` / ``position``
     - ``signal_execution:{signal_key}`` / ``signal_execution``
   * - ``balance:{conId}`` / ``balance``
     - ``contract_position:{conId}`` / ``contract_position``
   * - ``target:{conId}`` / ``target``
     - ``contract_execution:{conId}`` / ``contract_execution``

``roll:{series_key}`` and ``portfolio:{portfolio_key}`` retain their identities.
All new order/state documents include ``schema_version: 1``. The converter
renames framework attribution and filled-quantity fields, plus
``bracket_inputs`` / ``target_bracket_inputs`` to
``protection_inputs`` / ``target_protection_inputs``. It translates roll-family
values to ``SIGNAL`` / ``CONTRACT``. Broker quantities, signal key and position ID
values, opaque Portfolio payloads and arbitrary strategy metadata retain their
meaning. Older conversion targets use a different migration version and cannot
be reused as this conversion's target.
