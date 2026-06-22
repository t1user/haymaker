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

Workload Resume and Process Stop
================================

Within one running dataloader process, supervisor recovery preserves discovered
unfinished work in memory. If the socket has to reconnect and the workload is
restarted, the dataloader resumes active jobs before discovering new contracts.

A full process stop does not write a separate checkpoint file or checkpoint
collection. The next dataloader process reads the persisted Arctic-backed
datastore, derives boundaries from the stored data, and schedules any remaining
backfill, update, or gap-fill work from those datastore boundaries.

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

Request Sizing
==============

The dataloader sizes each ``reqHistoricalData`` call from the planned missing
range, the configured ``max_bars`` value, and IBKR's documented maximum
duration for the selected ``barSize``. IB's step-size rules are caps on valid
request shapes; they are not a promise that every contract/session will return
that many bars. Exact returned counts still depend on the instrument, session
calendar, holidays, early closes, market activity, and IB availability.

The dataloader uses canonical IB bar-size spelling such as ``1 secs`` and
``1 hour``. Noncanonical spellings such as ``1 sec`` are rejected before a
historical request is built.

Historical Availability Limits
==============================

The dataloader applies IBKR's documented hard availability limits when they can
be known from the contract and configured bar size:

* bars of ``30 secs`` or smaller are not requested more than six months back;
* expired futures are not requested earlier than two years before the exact
  contract expiry date;
* expired options, futures options, and warrants are skipped when their exact
  expiry date is known.

Other IB unavailability cases, such as delisted securities, exchange moves,
expired future spreads, or contract-specific gaps in intraday futures history,
are not reliably knowable from the dataloader's local inputs. Those cases fall
back to IB's response: when backfill reaches a no-data boundary for a series
that already has stored data, the dataloader marks that series with
``backfill_exhausted: true`` and avoids repeating older backfill requests on
future runs.

Datastore
=========

The dataloader currently supports the Arctic datastore backend only. There is
no datastore backend config key; the dataloader derives the Arctic library name
from ``wts`` and ``barSize`` such as ``TRADES_30_secs``. Collections use the
datastore module's default
``simple_collection_namer(contract)`` naming policy.

The dataloader talks to the async datastore interface. Arctic still owns data
cleaning and metadata updates when data is written. The current target
responsibility split is recorded in ``docs/dataloader-object-boundaries.md``.

Backfill Exhaustion Metadata
============================

When a backfill request reaches a point where IB returns no older bars for a
series that already has stored data, the dataloader writes
``backfill_exhausted: true`` to that series metadata. Later dataloader runs use
that marker to skip older backfill requests for the same Arctic library and
collection, while still allowing updates and optional gap filling.

Missing metadata never disables downloads. If the marker is absent, the
dataloader proceeds normally from the stored data boundaries and IB
``headTimeStamp``. The dataloader does not write a ``from`` metadata key; that
lower-bound metadata belongs with datastore-maintained series boundaries, in
the same spirit as the existing ``up_to`` field.

Failure Handling
================

Download workers distinguish broker request failures from session-level
failures. A broker request failure for one job is recorded and summarized when
the session finishes, while the remaining queued jobs continue. Empty
historical responses keep their normal no-data behavior, including
``backfill_exhausted`` marking for older backfill ranges.

Connection-class failures, such as ``ConnectionError`` or ``TimeoutError``, are
not recorded as ordinary job failures. They escape the dataloader session so
the supervised workload boundary can handle the broken connection. Local
processing failures, including datastore write failures, also abort the session
instead of being hidden behind a completed run summary.

Historical Date Policy
======================

The dataloader intentionally requests historical bars with Interactive Brokers
``formatDate=2``. For intraday bars, ``ib_insync`` returns timezone-aware UTC
``datetime`` values. For daily, weekly, and monthly bars, IB returns date-only
values. The dataloader keeps that split as its scheduling and storage policy:

* intraday ranges use UTC-aware ``datetime`` values;
* ``1 day``, ``1 week``, and ``1 month`` ranges use ``date`` values;
* naive intraday ``datetime`` values are rejected before scheduling.

Do not configure ``formatDate`` per run in this dataloader path. Other IB date
formats can depend on Gateway/TWS settings or instrument time zones and would
need a separate datastore policy to avoid mixing incompatible indexes in one
collection.
