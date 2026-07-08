*************
Configuration
*************

Haymaker is highly configurable. Parameters can be set in the following order of priority:

#. Command-line options (highest priority).
#. User-provided configuration YAML file passed on the command line.
#. User-provided configuration YAML file selected by environment variable.
#. System environment variables.
#. Default configuration YAML file (fallback for undefined variables).

Default Configuration Files
===========================

Sensible starting configurations are defined in default configuration files located here:

- For the Execution Module: https://github.com/t1user/haymaker/blob/master/haymaker/config/base_config.yaml 

- For the Dataloader Module: https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml 

Connection Recovery
===================

Live execution and the current managed dataloader path use the same socket
connection supervisor package, with separate supervisor instances for separately
owned sockets. The supervisor reconnects to the configured TWS or IB Gateway API
endpoint but does not start, stop, or restart the gateway process.
The default ``state`` supervisor verifies broker usability with a short
historical-data probe before starting the supervised workload. The alternative
``onion`` supervisor starts the workload after the socket connects, arms the IB
idle timeout, and defers historical-data probes until a health signal such as
``timeoutEvent`` or broker recovery wait requires one.

``app.supervisor`` selects the live supervisor implementation. ``state`` is the
default state-machine implementation; ``onion`` selects the alternative layered
implementation. For the current managed dataloader configuration, the equivalent
setting is top-level ``supervisor`` because dataloader connection settings are
read from a flat mapping.

For live execution, ``timeoutEvent`` is the active health signal after the IB
client has received no traffic for ``app.appTimeout`` seconds. Haymaker probes
the connection and enters a delayed restart cycle if the probe fails. For the
default ``state`` supervisor, only broker connectivity-loss messages ``1100``
and ``2110`` move Haymaker into broker-connectivity recovery while connected.
``app.auto_recovery_grace_period`` controls how long Haymaker waits there
before probing the connection. Generic ``updateEvent`` traffic does not wake
that state, and weak data-farm messages such as ``2103``, ``2105``, ``2157``,
``10182``, ``2104``, ``2106``, and ``2158`` are logged as context only when
``app.log_datafarm_status`` is ``True``. Set it to ``False`` to ignore those
non-actionable messages silently while keeping the same recovery behavior.
Restart cycles after failed probes wait ``app.connection_lost_retry`` before
reconnecting; ``app.retryDelay`` controls the pause between failed connection
attempts.
When IB sends ``1102`` (connectivity restored, data maintained),
``app.restart_on_recovered_connection`` controls whether Haymaker requests an
immediate restart cycle anyway. The default is ``False``: Haymaker treats
``1102`` as a prompt to probe while broker connectivity is marked lost and
leaves streamers to detect stale subscriptions. Set it to ``True`` to rebuild
immediately after ``1102``.
``max_recoveries`` limits consecutive unexpected supervisor-cycle recoveries
before Haymaker stops trying to rebuild the same failing cycle. Planned restart
requests and normal broker/socket recovery cycles do not consume this budget.
For the current managed dataloader configuration, the same setting is top-level
``restart_on_recovered_connection`` because dataloader connection settings are
read from a flat mapping.

See :doc:`ib_message_codes` for the broker message codes most relevant to
connection recovery and logging.
See :doc:`supervisor` for the supervisor lifecycle, workload contract, and state
transition chart.

Contract details are refreshed during successful live startup. The live
controller also schedules a fixed daily futures roll at the UTC time configured
by ``controller.future_roll_time`` so normal socket recovery cannot leave
futures selectors unchanged indefinitely.

The top-level ``ignore_errors`` setting lists noisy broker message codes that
should be omitted from normal logs. It does not suppress supervisor actions or
the optional raw ``broker.log`` audit trail.


User-Provided Configuration File
================================

The easiest way to create an override YAML file is to copy the default file and modify the desired values. Haymaker must be directed to the location of the overridden config file in one of two ways:

* From the command line using the ``--file`` or ``-f`` option.
* Via environment variables: ``HAYMAKER_HAYMAKER_CONFIG_OVERRIDES`` for the execution module or ``HAYMAKER_DATALOADER_CONFIG_OVERRIDES`` for the dataloader module.

Environment Variables
=====================

All Haymaker-related environment variables are prefixed with ``HAYMAKER_``. This prefix is removed when reading the variables, signaling to the framework which variables to load into its configuration environment. This prevents Haymaker from importing unrelated variables.

.. note::
   Environment variable names are **case-insensitive**.

.. warning::
   For nested variables, only top-level settings can be overridden via CLI or environment variables. These are intended for quick overrides; the framework is primarily configured via YAML files.

Overriding Defaults - Examples
==============================

To override defaults, copy a configuration file, modify the desired parameters, and pass the new file’s location to Haymaker via:

* Environment variable:

  .. code-block:: bash
     :caption: Setting the config override via environment variable

     export HAYMAKER_HAYMAKER_CONFIG_OVERRIDES=config.yaml

* Command-line argument:

  .. code-block:: bash
     :caption: Passing the config file via CLI

     haymaker strategy.py --file config.yaml

.. note::
   Command-line arguments take precedence over environment variables.

Passing Key-Value Pairs from Command Line
=========================================

Use ``-s`` or ``--set-option`` to temporarily override parameters. For example,
to specify a data source:

.. code-block:: bash
   :caption: Overriding a parameter via CLI

   dataloader my_list.csv

Available command-line options can be listed with:

.. code-block:: bash
   :caption: Displaying help for CLI options

   haymaker --help

CLI Options
-----------

Assuming your strategy is in ``your_module.py``, the following options are
available from the command line:

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_other_module
   :prog: haymaker

Examples
^^^^^^^^

.. code-block:: bash
   :caption: Running a strategy with a config override

   haymaker my_strategy.py -f config_overrides.yaml

This runs the strategy defined in ``my_strategy.py`` with configuration
overrides from ``config_overrides.yaml`` in the current directory.

.. code-block:: bash
   :caption: Triggering the emergency circuit breaker

   haymaker my_strategy.py --nuke

This activates the emergency circuit breaker, closing all open positions, canceling resting orders, and preventing new positions.

.. code-block:: bash
   :caption: Changing the log location

   haymaker my_strategy.py -s logging_path /path/to/log

This changes the log location to ``/path/to/log``.

Dataloader CLI Options
----------------------

Dataloader options differ from those of other modules:

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_dataloader
   :prog: dataloader

Examples
^^^^^^^^

.. code-block:: bash
   :caption: Collecting historical data from a CSV

   dataloader my_list.csv

This runs the dataloader to collect historical data for contracts defined in ``my_list.csv``.

.. code-block:: bash
   :caption: Running dataloader with a custom settings file

   dataloader my_list.csv -f settings.yaml

This runs the dataloader for contracts in ``my_list.csv`` with settings from
``settings.yaml`` in the current directory. This file should be a modified copy of:
https://github/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml

specifying the source file with defined contracts, data type, frequency, etc.
