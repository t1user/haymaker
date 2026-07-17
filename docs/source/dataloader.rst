*****************
Dataloader Module
*****************

The dataloader is Haymaker's standalone command for collecting Interactive
Brokers historical data into the configured Arctic datastore. It reads a CSV of
IB contract definitions, downloads the missing history for each contract, and
can later refresh the same series or fill detected gaps.

It does not place orders.

Quick Start
===========

1. Make sure TWS or IB Gateway is running and API connections are enabled.
2. Create a source CSV with one contract specification per row.
3. Create a dataloader YAML settings file, or use the defaults with command-line
   overrides.
4. Run ``dataloader``.

Example source file:

.. code-block:: text
   :caption: nq.csv

   secType,symbol,exchange,currency
   FUT,NQ,CME,USD

Example settings file:

.. code-block:: yaml
   :caption: nq_30s.yaml

   connection:
     client_id: 1
   download:
     source: nq.csv
     bar_size: 30 secs
     what_to_show: TRADES
     max_lookback_days: null
     gap_fill_mode: "off"
     use_rth: false
   futures:
     selector: current_and_expired

Run it:

.. code-block:: bash

   dataloader -f nq_30s.yaml

The first run backfills the available range allowed by IB and the datastore
state. Re-running the same command updates the same datastore series from its
stored boundary.

Source CSV
==========

The source CSV columns should be valid ``ib_insync.Contract`` fields. The most
important columns are usually:

``secType``
   IB security type, such as ``STK``, ``FUT``, ``CONTFUT``, ``CASH``, ``IND``,
   ``CFD``, ``OPT``, or ``FOP``.

``symbol``
   IB symbol, such as ``AAPL``, ``NQ``, or ``EUR``.

``exchange``
   Exchange or routing destination, such as ``SMART``, ``CME``, ``IDEALPRO``,
   or ``NASDAQ``.

``currency``
   Contract currency, such as ``USD`` or ``EUR``.

``lastTradeDateOrContractMonth``
   Optional contract month or exact expiry for futures/options. Use this when
   you want a specific contract instead of the configured futures selector.

Examples:

.. code-block:: text
   :caption: stocks.csv

   secType,symbol,exchange,currency
   STK,AAPL,SMART,USD
   STK,MSFT,SMART,USD

.. code-block:: text
   :caption: fx.csv

   secType,symbol,exchange,currency
   CASH,EUR,IDEALPRO,USD
   CASH,GBP,IDEALPRO,USD

.. code-block:: text
   :caption: exact_futures.csv

   secType,symbol,lastTradeDateOrContractMonth,exchange,currency
   FUT,NQ,20260918,CME,USD
   FUT,ES,20260918,CME,USD

Futures Selection
=================

Rows with ``secType`` set to ``FUT`` or ``CONTFUT`` use the dataloader futures
selector. Configure it with ``futures.selector``:

``current_and_expired``
   Default. Download the current contract and expired contracts returned by IB.
   This is the usual choice for building a historical futures library.

``current``
   Download the current contract. Use ``futures.current_index`` to offset from
   the current contract in the futures chain.

``fullchain``
   Download contracts from the full chain returned by IB. Use
   ``futures.full_chain_spec`` with ``full``, ``active``, or ``expired`` to narrow
   the chain.

``exact``
   Use the contract exactly as specified in the CSV row.

``contfuture``
   Download IB's continuous future contract. IB requires continuous-future
   historical requests to use an empty ``endDateTime``, so the dataloader makes
   one latest-ended request range and does not schedule internal gap fills for
   ``CONTFUT``.

``current_and_contfuture``
   Download both the current explicit future and IB's continuous future.

Common futures setup:

.. code-block:: yaml

   download:
     source: nq.csv
     bar_size: 30 secs
     what_to_show: TRADES
   futures:
     selector: current_and_expired

Exact-contract setup:

.. code-block:: yaml

   download:
     source: exact_futures.csv
     bar_size: 1 hour
     what_to_show: TRADES
   futures:
     selector: exact

Command Line
============

Run with a source CSV and default settings:

.. code-block:: bash

   dataloader contracts.csv

Run with a YAML settings file:

.. code-block:: bash

   dataloader -f settings.yaml

Override the gap-fill mode for one run:

.. code-block:: bash

   dataloader -f settings.yaml -g auto

Set simple config values from the command line:

.. code-block:: bash

   dataloader contracts.csv \
       -s download.bar_size "1 hour" \
       -s download.what_to_show MIDPOINT

Command-line values are parsed as YAML, preserving booleans, numbers, lists,
and ``null``. Quote values containing spaces.

Configuration
=============

Target constructors own intrinsic defaults and validation. The bundled
``haymaker/config/dataloader_base_config.yaml`` is a complete, commented
reference profile that pins the dataloader's effective defaults; user files
need include only the values changed for a run.

Common settings:

``download.source``
   CSV file containing contract rows.

``download.bar_size``
   IB bar size. Examples: ``30 secs``, ``1 min``, ``1 hour``, ``1 day``,
   ``1 week``, ``1 month``.

``download.what_to_show``
   IB ``whatToShow`` value. Common choices are ``TRADES``, ``MIDPOINT``,
   ``BID``, ``ASK``, and ``BID_ASK``.

``download.max_lookback_days``
   Optional maximum lookback span, in calendar days, considered by one
   dataloader run. Omit it or set it to ``null`` to load all data available
   under IB's limits and the datastore's own backfill state. Set it to a
   positive integer, such as ``30``, to deliberately limit a run to recent
   history.

   When IB does not provide a head timestamp, the dataloader still probes for
   historical data because IB can return bars in that situation. The fallback
   starts at most five calendar years back and remains clamped by this setting
   and known IB small-bar or expired-future availability limits.

``download.gap_fill_mode``
   Gap-fill behavior for already stored data. See :ref:`dataloader-gap-fill`.

``download.use_rth``
   Passed to IB historical-data and historical-schedule requests.

``connection.host``
   Hostname or IP address of TWS or IB Gateway. The default is ``127.0.0.1``.

``connection.port``
   TWS or IB Gateway API port. The default is ``4002``.

``connection.client_id``
   IB API client ID. The dataloader default is ``1`` so it is distinct from the
   live runtime's expected ``connection.client_id=0``.

``download.number_of_workers``
   Number of worker tasks consuming planned downloads. The default is ``10``.
   Increasing it can keep more contracts active, but does not bypass local or IB
   pacing limits.

``download.save_every_chunks``
   Number of downloaded chunks buffered before creating a new datastore
   version. The default is ``10``. A range completion or orderly dataloader
   shutdown also saves any remaining chunks. Larger values reduce full-series
   rewrites but increase the amount of recently downloaded data that a forced
   process termination may lose.

``pacing.allowance_fraction``
   Multiplies the dataloader's local pacing capacity. The default is ``1.0``.
   Use values below ``1.0`` to leave more room for other IB clients. Values above
   ``1.0`` are allowed for experimentation, but deliberately exceed IB's
   published pacing limits and may trigger broker throttling or pacing
   violations.

Example: daily stock bars:

.. code-block:: yaml
   :caption: stocks_daily.yaml

   download:
     source: stocks.csv
     bar_size: 1 day
     what_to_show: TRADES
     max_lookback_days: null
     gap_fill_mode: "off"
     use_rth: true

Example: hourly FX midpoint bars:

.. code-block:: yaml
   :caption: fx_hourly.yaml

   download:
     source: fx.csv
     bar_size: 1 hour
     what_to_show: MIDPOINT
     max_lookback_days: 365
     gap_fill_mode: auto
     use_rth: false

.. _dataloader-gap-fill:

Gap Filling
===========

Gap filling is optional. It is intended for a second pass over already stored
dataloader data.

``off``
   Do not schedule internal datastore gaps. This is the safest default for a
   first run.

``heuristic``
   Detect gaps from stored bar regularity. Repeated short local-time gaps and
   simple weekend gaps are treated as typical non-trading gaps.

``schedule``
   Ask IB for historical schedules and fill only gaps that overlap reported
   trading sessions. If IB does not return a usable schedule for a contract,
   planning for that contract fails.

``auto``
   Try schedule mode first, then fall back to heuristic mode when schedule data
   is unavailable.

Suggested workflow:

.. code-block:: bash

   dataloader -f nq_30s.yaml
   dataloader -f nq_30s.yaml -g auto

The first command builds or updates the library. The second command performs a
gap-fill pass over the stored data.

Datastore
=========

The dataloader currently writes to the Arctic-backed Haymaker datastore. The
library name is derived from ``download.what_to_show`` and
``download.bar_size``:

* ``TRADES`` + ``30 secs`` -> ``TRADES_30_secs``
* ``MIDPOINT`` + ``1 hour`` -> ``MIDPOINT_1_hour``
* ``TRADES`` + ``1 day`` -> ``TRADES_1_day``

Collections use Haymaker's default contract naming policy.

Each save creates a complete new Arctic version of the series. The dataloader
does not append fragments directly, so a saved version remains independently
readable. ``save_every_chunks`` controls how frequently a long download creates
these versions.

Read data back with the datastore API:

.. code-block:: python

   from ib_insync import Future
   from haymaker.datastore import ArcticStore

   store = ArcticStore("TRADES_30_secs")
   contract = Future("NQ", "20260918", "CME", currency="USD")

   df = store.read(contract)
   metadata = store.read_metadata(contract)

Use ``store.keys()`` to inspect the available collection names when you do not
know the exact contract key.

Availability And Limits
=======================

IB historical data has hard limits and contract-specific availability gaps. The
dataloader avoids requests that are known to be unavailable:

* bars of ``30 secs`` or smaller are not requested more than six months back;
* expired futures are not requested earlier than two years before exact expiry;
* expired options, futures options, and warrants are skipped when exact expiry
  is known.

Historical-data request chunk size is an internal dataloader policy. It is
targeted at approximately 10,000 bars and remains capped by IB's documented
duration rules for the selected bar size. IB recommends smaller responses, so
if large requests are slow, review the dataloader log before increasing worker
count or pacing allowance.

The dataloader reads existing data and applies known contract-age limits before
requesting ``headTimeStamp``. It requests that timestamp only when older
backfill remains possible; update-only and ``backfill_exhausted`` series do not
spend a discovery request on it.

When a backfill request reaches a no-data boundary for a series that already
has stored data, the dataloader writes ``backfill_exhausted: true`` to that
series metadata. Later runs skip older backfill for that series while still
allowing updates and optional gap filling.

Date Policy
===========

The dataloader uses IB ``formatDate=2``.

* Intraday bars are stored and compared as timezone-aware UTC ``datetime``
  values.
* Daily, weekly, and monthly bars use ``date`` values.
* Other IB date formats are not exposed in this dataloader path.

Connection And Pacing
=====================

``DataloaderRuntime`` creates its own ``ib_insync.IB`` client and session, then
runs under the Haymaker connection supervisor. The supervisor reconnects the
IB API socket but does not start, stop, or restart TWS/IB Gateway.

There is no dataloader connection-mode setting. Configure the gateway address,
port, and ``connection.client_id`` used by this standalone connection. The default client ID
is ``1``; choose another value when that ID is already in use.

The dataloader locally enforces IB's hard historical pacing rules for bars of
``30 secs`` or less. Bars of ``1 min`` or longer retain a concurrency bound but
do not use the small-bar 60-request/10-minute window. Head timestamps and
contract details use a separate conservative discovery allowance. Historical
schedules remain in the historical bucket because IB exposes them through the
historical-data API without publishing a separate quota. One small-bar request
slot is reserved for supervised connection probes.

With DEBUG logging enabled, request messages distinguish these stages:

* ``prepared ... handing it to the local pacer`` means a worker has built the
  request but has not sent it;
* ``Local pacer delaying`` names the local rule and remaining delay;
* ``Submitted ... to IB; awaiting response`` means IB has received the request;
* ``IB request ... completed`` reports broker response latency.

Every five minutes, ``Dataloader status`` reports engaged workers, workers and
requests waiting for IB, local pacing or concurrency waits, and contracts waiting
for a worker. Follow-up lines name each outstanding IB request with its elapsed
wait and each queued contract. Producer state distinguishes active planning from
backpressure such as ``blocked_on_full_queue (next=...)``. Availability skips
explicitly say ``local policy`` when they are inferred from documented IB limits
rather than returned by IB.

Completion And Failures
=======================

Normal completion means all planned contracts were processed, but individual
historical requests can fail while other contracts continue. Review the final
log summary for failed download jobs before treating a large run as complete.
Re-running the same command plans remaining work again from the data already
saved in Arctic.

Pressing ``Ctrl-C`` requests an orderly supervisor stop. Active requests are
cancelled, downloaded chunks are flushed, and queued datastore metadata writes
are drained before the process exits. Avoid sending a second forced interrupt
while these final writes are being logged.

The run stops rather than continuing when it cannot safely plan or persist data,
including these cases:

* a source row identifies an unknown or ambiguous contract;
* ``schedule`` gap mode cannot obtain a usable historical schedule;
* local date or dataframe processing fails;
* an Arctic read or write fails.

IB connection failures are handled by the connection supervisor. During an
orderly restart or shutdown, buffered chunks are saved before the session
finishes. A forced process kill can lose only the chunks accumulated since the
last threshold or range-completion save.

Operational Notes
=================

* Use a distinct ``connection.client_id`` if another process already uses ``1``.
* Keep ``download.gap_fill_mode: off`` for initial large downloads unless you are
  intentionally running a gap pass.
* Use ``download.use_rth: true`` only when you want regular-trading-hours data and
  schedules.
* Re-run the same settings file to update an existing library.
* If IB returns no older backfill data for an existing series, future runs will
  honor ``backfill_exhausted: true`` metadata.
* A full process stop does not write a separate checkpoint. The next process
   derives remaining work from the persisted datastore boundaries.
