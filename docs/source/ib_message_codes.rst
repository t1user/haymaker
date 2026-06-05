IB Message Codes
================

Interactive Brokers sends true errors, warnings, and informational status
messages through ``IB.errorEvent`` / ``EWrapper.error``. Haymaker treats these
as broker messages first; logging level and recovery action are separate
decisions.

The official reference is Interactive Brokers' `TWS API message codes
<https://interactivebrokers.github.io/tws-api/message_codes.html>`_.

Connection and Data Recovery
----------------------------

These codes are most relevant to Haymaker connection supervision.

.. list-table::
   :header-rows: 1
   :widths: 12 34 54

   * - Code
     - Meaning
     - Haymaker recovery interpretation
   * - ``1100``
     - Connectivity between IB and TWS/Gateway has been lost.
     - Recoverable broker-degraded context. Wait after timeout before restart.
   * - ``1101``
     - Connectivity restored, data lost.
     - Request restart/rebuild because subscriptions must be resubmitted.
   * - ``1102``
     - Connectivity restored, data maintained.
     - Clear broker-degraded context; no rebuild required by this message alone.
   * - ``1300``
     - TWS socket port has been reset and the connection is being dropped.
     - Reconnect path is required; check the configured port if recovery fails.
   * - ``2103``
     - Market data farm is disconnected.
     - Recoverable data-farm degradation; use as recent context on timeout.
   * - ``2104``
     - Market data farm connection is OK.
     - Informational recovery/startup message.
   * - ``2105``
     - Historical data farm is disconnected.
     - Recoverable HMDS degradation; use as recent context on timeout.
   * - ``2106``
     - Historical data farm is connected.
     - Informational recovery/startup message.
   * - ``2107``
     - Historical data farm is inactive but available on demand.
     - Normal idle state; usually safe to ignore.
   * - ``2108``
     - Market data farm is inactive but available on demand.
     - Normal idle state; usually safe to ignore.
   * - ``2110``
     - TWS/Gateway connection to IB servers is broken and should restore
       automatically.
     - Strong recoverable broker-degraded context. Wait after timeout.
   * - ``2119``
     - Market data farm is connecting.
     - Recovery progress, not proof that all subscriptions are active.
   * - ``2157``
     - Security definition data farm connection is broken.
     - Recoverable sec-def degradation; relevant to contract-detail requests.
   * - ``2158``
     - Security definition data farm connection is OK.
     - Informational recovery/startup message.
   * - ``10182``
     - Failed to request live updates because IB reports disconnected state.
     - Strong request-level degradation; use as recent context on timeout/probe
       failure.

Order and Request Messages
--------------------------

These commonly affect logging and user attention rather than socket recovery.

.. list-table::
   :header-rows: 1
   :widths: 12 34 54

   * - Code
     - Meaning
     - Haymaker logging interpretation
   * - ``201``
     - Order rejected.
     - Critical; user attention required.
   * - ``202``
     - Order cancelled.
     - Usually expected because Haymaker cancels orders frequently.
   * - ``321``
     - Server validation message for an API request.
     - Often noisy; debug unless classified as actionable by context.
   * - ``347``
     - Short sale slot validation failed.
     - Potentially actionable order/request validation issue.
   * - ``500+``
     - Client-side API message range.
     - Usually debug unless it causes failed recovery or failed trading action.

Operational Rule of Thumb
-------------------------

Use ``timeoutEvent`` and probes as the first active health checks. Recent broker
messages decide whether a failed health check should wait for broker recovery or
request a reconnect/rebuild immediately.
