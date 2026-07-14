Connection Supervisor
=====================

Purpose
-------

Long-running Interactive Brokers connections can lose the API socket, lose
broker connectivity, or keep a connected socket while market-data
subscriptions stop updating. Haymaker's connection supervisor detects these
conditions and restores a usable connection and workload without requiring the
application to be restarted manually.

The supervisor manages only Haymaker-owned API connections. It reconnects to
TWS or IB Gateway, but it does not start, stop, or restart the gateway process.

Recovery Policies
-----------------

* Work starts only after Haymaker has connected and successfully tested broker
  request availability.
* An API socket disconnect, IB ``1101`` (data lost), or ``1300`` (API port
  reset) rebuilds the connection and workload.
* IB ``1100`` and ``2110`` enter a short recovery wait. IB ``1102`` (data
  maintained), or expiry of the wait, causes a connection test. By default, a
  successful test keeps the existing workload and subscriptions.
* IB ``10182`` indicates a failed live-update subscription. Haymaker waits for
  180 quiet seconds after the last ``10182`` and then rebuilds the workload.
  Streamer timeouts remain an independent fallback.
* Data-farm status messages such as ``2103``/``2104``, ``2105``/``2106``, and
  ``2157``/``2158`` are informational and do not initiate recovery.
* Repeated restart requests are combined, failed connection attempts are
  retried, and controller synchronization is aborted cleanly while the
  connection is unavailable.

Usage
-----

The live-trading command and managed dataloader create and run the supervisor
automatically. Strategy modules do not need to instantiate or control it.
Configure live trading under the ``app`` section of the YAML file. Managed
dataloader connection settings use the same names at the top level.

.. code-block:: yaml

   app:
     host: 127.0.0.1
     port: 4002
     connectTimeout: 15
     retryDelay: 30
     appTimeout: 90
     probeTimeout: 15
     connection_lost_retry: 90
     auto_recovery_grace_period: 120
     restart_on_recovered_connection: false
     log_datafarm_status: true

Use the normal configuration precedence described in :doc:`configuration` to
override these values. The framework uses separate IB client IDs for live
trading and managed dataloader connections.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Setting
     - Default
     - Effect
   * - ``host``
     - ``127.0.0.1``
     - TWS or IB Gateway API host.
   * - ``port``
     - ``4002``
     - TWS or IB Gateway API port.
   * - ``connectTimeout``
     - ``15``
     - Seconds allowed for one connection attempt.
   * - ``retryDelay``
     - ``30``
     - Seconds between failed connection attempts.
   * - ``appTimeout``
     - ``90``
     - Seconds without IB traffic before Haymaker tests the connection.
   * - ``probeContract``
     - ``EURUSD``
     - Contract used for the small historical-data connection test.
   * - ``probeTimeout``
     - ``15``
     - Seconds allowed for a connection test.
   * - ``connection_lost_retry``
     - ``90``
     - Delay before reconnecting after a failed connection test.
   * - ``auto_recovery_grace_period``
     - ``120``
     - Maximum wait for broker connectivity to recover before testing it.
   * - ``restart_on_recovered_connection``
     - ``false``
     - Rebuild after ``1102`` instead of first testing and preserving the
       existing workload.
   * - ``log_datafarm_status``
     - ``true``
     - Log informational data-farm status messages. This does not alter
       recovery behavior.

See :doc:`ib_message_codes` when interpreting broker messages in logs.
