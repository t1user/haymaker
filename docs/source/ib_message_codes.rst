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
     - Enter broker auto-recovery wait while connected.
   * - ``1101``
     - Connectivity restored, data lost.
     - Request restart/rebuild because subscriptions must be resubmitted.
   * - ``1102``
     - Connectivity restored, data maintained.
     - During broker auto-recovery wait, trigger a recovery probe; if
       ``restart_on_recovered_connection`` is enabled, request restart/rebuild.
   * - ``1300``
     - TWS socket port has been reset and the connection is being dropped.
     - Reconnect path is required; check the configured port if recovery fails.
   * - ``2103``
     - Market data farm is disconnected.
     - Enter broker auto-recovery wait while connected.
   * - ``2104``
     - Market data farm connection is OK.
     - Informational recovery/startup message.
   * - ``2105``
     - Historical data farm is disconnected.
     - Enter broker auto-recovery wait while connected.
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
     - Enter broker auto-recovery wait while connected.
   * - ``2119``
     - Market data farm is connecting.
     - Recovery progress, not proof that all subscriptions are active.
   * - ``2157``
     - Security definition data farm connection is broken.
     - Enter broker auto-recovery wait while connected.
   * - ``2158``
     - Security definition data farm connection is OK.
     - Informational recovery/startup message.
   * - ``10182``
     - Failed to request live updates because IB reports disconnected state.
     - Enter broker auto-recovery wait while connected.

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

Use ``timeoutEvent`` and probes as active health checks. Broker-degraded message
codes enter broker auto-recovery wait immediately while connected, and
``updateEvent`` or ``1102`` is a useful hint that traffic has resumed and should
trigger a recovery probe. Failed recovery probes should keep the original grace
timer intact. A successful probe only proves current connectivity; stale
subscriptions may still be detected later by streamer timeouts.
