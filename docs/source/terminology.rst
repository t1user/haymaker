Terminology
===========

Use these names in strategy APIs, framework code and operational diagnostics.
A new synonym requires an explicit terminology decision and coordinated updates
here, in the interfaces and in the terminology test.

.. list-table:: Filled and desired exposure
   :header-rows: 1

   * - Scope
     - Filled exposure
     - Accepted desired exposure
   * - Signal
     - ``SignalPosition``
     - ``SignalTarget``
   * - Concrete contract
     - ``ContractPosition``
     - ``ContractTarget``

A **signal**, identified by ``signal_key``, is a stable named stream. A
``Signal`` message is one observation on that stream. ``PositionRequest``
adds direction and intent before allocation supplies an absolute
``PositionTarget``. Book exposes the accepted target using the typed records
above; those records are separate from the allocation-to-execution message.

A **position** describes filled exposure. ``filled_quantity`` is the signed
remaining quantity after fills and accepted accounting corrections; it excludes
pending orders and targets. ``target_quantity`` is accepted desired exposure.
``effective_quantity`` includes relevant working orders and is a different query.

``position_id`` identifies an independently managed signal position. Rolling
preserves it; a subsequent opening or reversal creates a new ID.
``ProtectionState`` stores that position's identity and protection inputs;
order records remain authoritative for protective order status and fills.

**Signal execution** manages independent signal positions.
**Contract execution** manages aggregate positions for concrete contracts.
One process uses one execution family. Brackets name protective order
mechanisms; ``BracketExecutionModel`` and ``SerialTargetExecutionModel`` retain
those mechanism-specific names. Models declare ``ExecutionMode.SIGNAL`` or
``ExecutionMode.CONTRACT``; Controller enforces this choice at model registration.
Futures rolling uses the same mode; the corresponding roll executors are
``SignalFutureRollExecutor`` and ``ContractFutureRollExecutor``.

A **contract specification**, ``contract_spec``, is the user's instrument
declaration, potentially unqualified. Constructing an Atom with ``contract=...``
or assigning ``atom.contract`` supplies that specification. Reading
``atom.contract`` during initialized operation returns the selected qualified
contract. Before initialization it retains the declaration behavior;
``atom.contract_spec`` returns a copy of the saved declaration. Qualification
membership and ACTIVE/NEXT selection determine resolution.

A qualified **contract** is identified by ``conId``. A futures **series** groups
related expiries and retains ``series_key`` for selection and rolling.
``ContractPosition`` always belongs to one concrete ``conId``, including stocks.
A specification-wide query returns separate positions for its member contracts.

Target intent
-------------

``TargetIntent`` is asserted when a signal execution target is accepted:

* ``OPEN`` establishes the requested position or reaffirms an already matching
  position.
* ``CLOSE`` requests zero target quantity.
* ``REVERSE`` requests the opposite direction. Signal execution closes the held
  position before opening its replacement, possibly on another contract.

Acceptance validates intent against effective quantity, including relevant
working orders. Numeric target state is authoritative afterward. Recovery
computes remaining work from positions, targets and orders; it does not replay
intent. Contract execution targets omit intent.
