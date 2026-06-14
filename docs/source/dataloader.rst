*****************
Dataloader Module
*****************

Interactive Brokers imposes strict throttling on historical data downloads.While IB is not designed for data collection, the Dataloader Module works within IB limitations to download available historical price and volume data.

Connection Recovery
===================

The dataloader uses the same Interactive Brokers socket connection supervisor
package as live execution, but with its own supervisor instance, ``IB`` client,
and owned socket. The supervisor reconnects the API client but does not manage
or restart the TWS or IB Gateway process.
See :doc:`supervisor` for the supervisor lifecycle and state transition chart.

The dataloader client ID defaults to ``1`` so it is distinct from the live
runtime's expected ``clientId=0``. Override ``clientId`` in the dataloader
config only when another process already uses that ID. A duplicate client ID is
treated as a
connection configuration failure.
