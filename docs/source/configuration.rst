*************
Configuration
*************

Haymaker configuration describes framework infrastructure and operating policy:
connections, logging, persistence, controller behavior, order defaults, and
historical downloads. Strategy-specific parameters remain ordinary Python data
in the user's strategy module; Haymaker does not parse or inject them.

Configuration is loaded once by the command-line entrypoint. The resulting
validated settings are passed to the live or dataloader runtime, so importing a
Haymaker module does not inspect command-line arguments or environment values.

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

The complete schemas and defaults are defined in:

* ``haymaker/config/live_base_config.yaml`` for live execution.
* ``haymaker/config/dataloader_base_config.yaml`` for historical downloads.

Override files only need to contain values that differ from these defaults.
Unknown sections, unknown keys, invalid types, and duplicate YAML keys are
reported as configuration errors before the runtime is created.

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
interpret. A path must already exist in the profile schema; command-line
overrides cannot invent new settings.

Dedicated live switches are ``--cold-start``, ``--reset``, ``--zero``, and
``--nuke``. The dataloader provides ``--gap-fill-mode`` and accepts the contract
source CSV as its optional positional argument. Dedicated switches have the
highest precedence.

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_other_module
   :prog: haymaker

Run ``dataloader --help`` for the complete dataloader option list.

Live Settings
=============

Live configuration is grouped into these sections:

``startup``
   One-run startup actions: ``cold_start``, ``reset``, ``zero``, and ``nuke``.

``connection``
   IB endpoint, client ID, timeouts, connection probe, and recovery policy. See
   :doc:`supervisor`.

``logging``
   ``config_file``, output ``directory``, and optional raw broker logging. See
   :doc:`logging`.

``controller``
   Synchronization, health checks, execution verification, error filtering,
   unknown-trade policy, bracket policy, and futures-roll time.

``state_machine``
   Save delay, Mongo collection names, and rejected-order limit.

``storage``
   Base directory, Mongo client arguments and database, Arctic libraries, and
   dataframe save frequency.

``blotter``
   Enablement and a safe built-in ``csv`` or ``mongo`` saver specification.

``orders``
   Default IB fields for open, close, stop, and take-profit orders, plus the
   default OCA type.

``timeout``
   Default streamer timeout in seconds and the ``restart`` or ``log`` action.

``futures``
   Business-day offsets used to select and roll live futures contracts.

Dataloader Settings
===================

Dataloader configuration shares the ``connection``, ``logging``, and
``storage`` sections and adds:

``download``
   Source CSV, bar size, data type, lookback and gap-fill policy, RTH selection,
   save cadence, and worker count.

``pacing``
   Optional pacing bypass and the fraction of IB request capacity assigned to
   the dataloader.

``futures``
   Contract selector, full-chain marker, and current-contract index.

Safe Object Conversion
======================

YAML contains plain data only. The loader converts the ``connection``
``probe_contract`` mapping to an :class:`ib_insync.contract.Contract`, and
converts order ``algoParams`` entries to :class:`ib_insync.objects.TagValue`
instances. Arbitrary Python object constructors and executable YAML tags are
rejected.

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
   * - ``barSize``
     - ``download.bar_size``
   * - ``wts``
     - ``download.what_to_show``
   * - ``useRTH``
     - ``download.use_rth``
   * - ``pacer_allowance_fraction``
     - ``pacing.allowance_fraction``
   * - ``-s key value`` with a flat key
     - ``-s section.key value``

Code that imported ``haymaker.config.CONFIG`` should instead receive the
specific typed settings or ready runtime service it needs. User strategy code
normally does not need direct access to framework settings.
