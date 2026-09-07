*****************
DataFrame Storage
*****************

Haymaker provides separate storage defaults for broker market-data history and
calculated Signal data. Most strategies only select whether to use those
defaults; custom storage objects are an advanced extension point.

Default storage
===============

The live YAML profile selects one Arctic library for each purpose:

.. code-block:: yaml

   market_data_store:
     library: market_data

   signal_persistence:
     library: signal_data

Use the component boolean options to select these runtime defaults:

* ``HistoricalDataStreamer(datastore=False)`` does not inspect persisted
  history. ``datastore=True`` uses the configured market-data store to shorten
  later IB requests.
* ``FuturesPandasAggregator()`` uses the configured market-data store. Its
  datastore cannot be disabled because persisted history is part of its
  aggregation contract.
* ``PandasSignalModel(persistence=False)`` emits Signals without retaining the
  calculated dataframe. ``persistence=True`` uses a model-owned default policy
  backed by the configured Signal library.

The streamer and aggregator resolve the same cached market-data store when
their bar size, data type, and regular-hours setting match. Signal persistence
is separate because it queues an audit generation and returns the reference
placed in Signal metadata.

.. _advanced-storage-configuration:

Advanced storage configuration
==============================

Pass a custom object only when the runtime defaults are unsuitable. Market-data
components accept an :class:`~haymaker.datastore.AsyncDataStore`, while
``PandasSignalModel`` accepts a
:class:`~haymaker.datastore.SignalFramePersistence`. These interfaces are
deliberately different: market-data history is read and written with awaited
completion, whereas Signal calculation persistence accepts work synchronously
on the trading path and returns an optional lookup reference.

Strategy modules are imported after the live runtime has installed its
``RuntimeContext``. Advanced composition can therefore obtain the configured
backend provider from ``Atom.runtime.frame_store_provider``. Do not call
``Atom.set_runtime_context()`` from strategy code.

Custom market-data history
--------------------------

Construct one awaited store and pass the same instance to the historical
streamer and futures dataframe aggregator:

.. code-block:: python

   import ib_insync as ibi

   from haymaker.base import Atom
   from haymaker.components import (
       FuturesPandasAggregator,
       HistoricalDataStreamer,
   )
   from haymaker.datastore import MarketDataSymbolNamer


   contract = ibi.ContFuture("ES", exchange="CME")
   provider = Atom.runtime.frame_store_provider
   history_store = provider.datastore(
       "custom_market_data",
       symbol_namer=MarketDataSymbolNamer(
           barSizeSetting="1 day",
           whatToShow="TRADES",
           useRTH=False,
       ),
   )

   streamer = HistoricalDataStreamer(
       contract=contract,
       durationStr=100,
       barSizeSetting="1 day",
       whatToShow="TRADES",
       useRTH=False,
       datastore=history_store,
   )
   frames = FuturesPandasAggregator(datastore=history_store)

The symbol namer's bar size, data type, and regular-hours policy must match the
streamer request. Reusing the store instance ensures that the streamer reads
the endpoint maintained by the aggregator.

A completely custom implementation may be supplied instead. Its mutation
methods must be awaitable, and a successful return must mean that the backend
operation has completed. Symbol naming is fixed when the store is constructed.

Custom Signal calculation persistence
-------------------------------------

The standard custom policy can be built from a dedicated ordered ``DRAIN``
sink:

.. code-block:: python

   from haymaker.async_wrappers import QueueShutdownPolicy
   from haymaker.base import Atom
   from haymaker.datastore import (
       QueuedSignalFramePersistence,
       simple_symbol_namer,
   )


   provider = Atom.runtime.frame_store_provider
   signal_sink = provider.queued_sink(
       "custom_signal_data",
       symbol_namer=simple_symbol_namer,
       shutdown_policy=QueueShutdownPolicy.DRAIN,
   )
   signal_persistence = QueuedSignalFramePersistence(signal_sink)

   signals = MySignalModel(
       SOURCE,
       CONTRACT,
       SIGNAL_TYPE,
       persistence=signal_persistence,
   )

Create a separate ``SignalFramePersistence`` instance for every persisted
model. The policy retains its current generation and last accepted dataframe
index; sharing it would mix independent model state.

A custom policy's ``save()`` method runs synchronously on the Signal-emission
path. It must only accept or queue work, never wait for storage I/O, and return
a string reference or ``None``. Queue-acceptance failure is logged and the
Signal is still emitted without the reference. Once a dataframe has been
accepted, the model must not mutate it.

Advanced storage interfaces
---------------------------

Custom Portfolio recovery
~~~~~~~~~~~~~~~~~~~~~~~~~

``PortfolioStateMixin`` is optional. Its default ``load_state`` and
``save_state`` delegate to Book under a stable ``portfolio_key``. Loading is
explicit, usually during startup before new inputs; save a normalized mapping
of allocation state rather than Signal objects. With asynchronous Book saving,
return from ``save_state`` means ordered queue acceptance, not backend commit.

Override those methods to use an independent backend, or implement persistence
directly in your Portfolio without the mixin. Independently owned stores need
their own startup, cancellation and drain handling. Neither approach promises
an atomic transaction between Portfolio state and execution targets. Recovery
must reconcile the saved decision state with Book's latest targets and fills;
completion feedback can be repeated or missed across process boundaries.

.. autoclass:: haymaker.datastore.AsyncDataStore

.. autoclass:: haymaker.datastore.QueuedDataSink

.. autoclass:: haymaker.datastore.SignalFramePersistence
   :members: save

.. autoclass:: haymaker.datastore.QueuedSignalFramePersistence
   :members: save
