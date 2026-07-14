*****************
Dataloader Module
*****************

Interactive Brokers imposes strict throttling on historical data downloads.While IB is not designed for data collection, the Dataloader Module works within IB limitations to download available historical price and volume data.

Connection Recovery
===================

The current managed dataloader path uses the same Interactive Brokers socket
connection supervisor package as live execution, but with its own supervisor
instance and owned socket. The supervisor reconnects the API client but does not
manage or restart the TWS or IB Gateway process.
See :doc:`supervisor` for connection recovery policies and settings.

The ``run_mode`` configuration option accepts:

* ``reconnect``: clean up interrupted work and run it again after reconnection.
* ``wait``: reconnect the socket but allow existing work to wait for data flow
  to resume.
