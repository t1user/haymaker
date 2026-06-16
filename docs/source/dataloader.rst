*****************
Dataloader Module
*****************

Interactive Brokers imposes strict throttling on historical data downloads.
While IB is not designed for data collection, the Dataloader Module works
within IB limitations to download available historical price and volume data.

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

Request Pacing
==============

The dataloader keeps a client-side request pacer for IB historical-data,
head-timestamp, historical-schedule, and contract-details requests. The pacer
uses module-level limits that model current IBKR historical-data guidance and
keeps retry handling for pacing violations out of worker code.

Two dataloader config keys adjust that machinery:

``pacer_allowance_fraction``
   Multiplies the module-level client-side capacities. Values below ``1.0``
   reserve more broker allowance for other IB clients; values above ``1.0``
   deliberately make the local pacer more aggressive.

``pacer_no_restriction``
   Disables client-side pacing waits while keeping the same request-routing API.
   Use this only when relying on Gateway/TWS pacing or for targeted testing.

Contract selectors use paced ``reqContractDetailsAsync`` calls for
qualification and futures-chain discovery. The resolved contracts returned by
IB are then used for historical requests so pacing follows ``ib_insync``
contract hash semantics.
