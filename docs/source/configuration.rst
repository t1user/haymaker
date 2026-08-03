*************
Configuration
*************

Haymaker configuration describes framework infrastructure and operating policy:
connections, logging, persistence, controller behavior, order defaults, and
historical downloads. Strategy-specific parameters remain ordinary Python data
in the user's strategy module; Haymaker does not parse or inject them.

Configuration is loaded once by the command-line entrypoint. Live and
dataloader configuration remain grouped into plain section mappings until each
target constructs itself. Importing a Haymaker module does not inspect
command-line arguments or environment values.

Precedence
==========

Values are merged in this order, from lowest to highest priority:

#. Bundled profile defaults.
#. A YAML file selected by the profile's environment variable.
#. A YAML file supplied with ``--file`` or ``-f``.
#. Repeatable dotted-path ``--set-option`` or ``-s`` overrides.
#. Dedicated command-line switches such as ``--reset`` or
   ``--gap-fill-mode``.

Mappings are merged recursively. Lists and scalar values replace the lower
priority value. A typed mapping such as ``blotter.saver`` is replaced as a
whole when its ``type`` changes, which prevents options for one saver type from
leaking into another.

Default Configuration Files
===========================

The bundled profiles are defined in:

* ``haymaker/config/live_base_config.yaml`` for live execution.
* ``haymaker/config/dataloader_base_config.yaml`` for historical downloads.

Both files are complete, commented reference profiles. They enumerate the
supported settings and pin the effective defaults used by their command. User
override files only need to contain values that differ from the bundled
profile. The loader validates configuration structure and typed storage
settings; the runtime targets that consume the other sections validate their
own fields and values.

Duplicate YAML keys and unknown top-level sections are rejected while loading.
Section keys and values are interpreted by the target that consumes them;
ordinary constructor errors therefore report invalid target arguments during
runtime construction.

Framework and Strategy Configuration
====================================

Keep framework overrides in YAML and strategy construction in Python. For
example:

.. code-block:: yaml
   :caption: live_config.yaml

   connection:
     port: 4002
   controller:
     sync_frequency: 60
   logging:
     directory: production_logs

.. code-block:: python
   :caption: strategy.py

   strategy = MyStrategy(fast_period=10, slow_period=30)

The strategy parameters are not part of the Haymaker configuration schema and
should not be placed in the framework YAML file.

YAML Override Files
===================

Pass a profile-specific YAML file on the command line:

.. code-block:: bash

   haymaker strategy.py --file live_config.yaml
   dataloader contracts.csv --file dataloader_config.yaml

Alternatively, select a deployment-specific file with one of the two supported
environment variables:

.. code-block:: bash

   export HAYMAKER_HAYMAKER_CONFIG_OVERRIDES=/path/to/live_config.yaml
   export HAYMAKER_DATALOADER_CONFIG_OVERRIDES=/path/to/dataloader_config.yaml

The live and dataloader selectors are independent. Direct setting overrides
such as ``HAYMAKER_LOGGING_PATH`` are not supported; use YAML or a dotted CLI
override instead. Environment variable names are case-sensitive.

Command-Line Overrides
======================

Use ``-s`` or ``--set-option`` with a dotted setting path and a YAML value.
The option may be repeated:

.. code-block:: bash

   haymaker strategy.py \
       --set-option controller.sync_frequency 60 \
       --set-option logging.log_broker true

   dataloader contracts.csv \
       --set-option download.bar_size '1 hour' \
       --set-option download.number_of_workers 4

Values are parsed as YAML, so booleans and numbers are typed rather than stored
as strings. Quote values containing spaces or values that the shell could
interpret. Use the bundled profile comments to find supported setting paths;
unknown options fail when their target is constructed.

Dedicated live switches are ``--cold-start``, ``--reset``, ``--zero``, and
``--nuke``. The dataloader provides ``--gap-fill-mode`` and accepts the contract
source CSV as its optional positional argument. Dedicated switches have the
highest precedence.

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_other_module
   :prog: haymaker

Run ``dataloader --help`` for the complete dataloader option list.

Live Configuration
==================

Live configuration is grouped by the service or policy it configures. Logging
and storage remain subsystem groups because their settings are composed across
closely related runtime objects.

``connection``
   IB endpoint, client ID, timeouts, connection probe, and recovery policy. See
   :doc:`supervisor`.

``logging``
   ``config_file``, output ``directory``, optional package ``level``, and raw
   broker logging. See :doc:`logging`.

``controller``
   One-run actions under ``controller.startup``, synchronization, health checks,
   execution verification, error filtering, unknown-trade policy, bracket
   policy, and futures-roll time.

``state_machine``
   Save delay, Mongo collection names, and rejected-order limit.

``storage``
   Base directory plus Mongo client arguments and the framework database name.
   Dataframe library names and save frequency belong to strategy composition
   and consumer constructors.

``blotter``
   Enablement and a safe built-in ``csv`` or ``mongo`` saver specification.

``orders``
   Default IB fields for open, close, stop, and take-profit orders, plus the
   default OCA type.

``timeout``
   Default streamer timeout in seconds and the ``restart`` or ``log`` action.

``futures``
   ``futures_roll_bdays`` controls when the selector advances ``ACTIVE`` and
   positions outside the allowed ``ACTIVE``/``NEXT`` set become eligible for
   rolling. ``futures_roll_margin_bdays`` advances ``NEXT`` earlier so strategy
   atoms may route new entries away from the expiring contract while market
   data and existing positions remain on ``ACTIVE``. Selector dates are
   evaluated as timezone-naive UTC and refreshed on supervised workload start.

Dataloader Configuration
========================

Dataloader configuration shares the ``connection`` and ``logging`` section
shapes with live execution. Its narrower ``storage`` section contains only
``base_directory`` and ``mongodb.client``. ``mongodb.database`` is live-only;
dataframe library and save-frequency settings are not framework configuration.
All of those unsupported paths are rejected.

``storage``
   Base directory and keyword arguments passed to ``pymongo.MongoClient``. The
   dataloader derives its Arctic library name from the requested data type and
   bar size, so no library name is configured here.

The remaining dataloader groups are:

``download``
   Source CSV, bar size, data type, lookback and gap-fill policy, RTH selection,
   save cadence, and worker count. This user-facing run group is composed into
   the dataloader manager and worker session.

``pacing``
   Optional pacing bypass and the fraction of IB request capacity assigned to
   the dataloader.

``futures``
   Contract selector, full-chain marker, and current-contract index.

Unknown dataloader ``storage`` keys fail during configuration loading; unknown
keys in ``download``, ``pacing``, or ``futures`` fail during runtime
construction. Target constructors also enforce value types and ranges,
including positive worker/save counts, an optional positive lookback, booleans
for boolean policies, a finite positive pacing allowance, and valid futures
selector values.

Safe Object Conversion
======================

YAML contains plain data only. ``ConnectionSettings.from_mapping()`` converts
the ``connection.probe_contract`` mapping to an
:class:`ib_insync.contract.Contract`, while ``OrderDefaults.from_mapping()``
converts order ``algoParams`` entries to
:class:`ib_insync.objects.TagValue` instances and verifies that each mapping can
construct an IB order. Arbitrary Python object constructors and executable YAML
tags are rejected.

Migration from the Legacy Schema
================================

The former flat configuration API and import-time ``CONFIG`` object have been
removed. Common migrations include:

.. list-table::
   :header-rows: 1

   * - Legacy key or command
     - Replacement
   * - ``app.host`` / top-level ``host``
     - ``connection.host``
   * - ``clientId``
     - ``connection.client_id``
   * - ``logging_config``
     - ``logging.config_file``
   * - ``logging_path``
     - ``logging.directory``
   * - ``data_folder``
     - ``storage.base_directory``
   * - ``storage.block_library``
     - Select the library explicitly in the strategy's
       ``FrameStoreProvider.queued_sink()`` call
   * - ``storage.market_data_library``
     - Select the library explicitly in the strategy's
       ``FrameStoreProvider.datastore()`` call
   * - ``storage.dataframe_save_frequency``
     - Pass ``save_frequency`` to ``DfAggregator``
   * - ``barSize``
     - ``download.bar_size``
   * - ``wts``
     - ``download.what_to_show``
   * - ``useRTH``
     - ``download.use_rth``
   * - ``pacer_allowance_fraction``
     - ``pacing.allowance_fraction``
   * - top-level ``startup.reset`` and related startup paths
     - ``controller.startup.reset`` and related startup paths
   * - ``-s key value`` with a flat key
     - ``-s section.key value``

Code that imported ``haymaker.config.CONFIG`` should instead receive a ready
runtime service. Framework composition uses ``load_live_config()`` and passes
each section to its owning target or subsystem composition boundary; user
strategy code normally does not need direct access to framework configuration.
