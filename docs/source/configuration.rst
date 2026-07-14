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

Live execution and managed dataloader runs automatically recover Haymaker-owned
Interactive Brokers API connections. The supervisor reconnects to the
configured TWS or IB Gateway endpoint but does not manage the gateway process.

Live connection settings belong under ``app``. Managed dataloader connection
settings use the same names at the top level. See :doc:`supervisor` for the
available settings and recovery policies, and :doc:`ib_message_codes` when
interpreting broker messages.

The ``controller.ignore_errors`` setting lists noisy controller-owned broker
message codes that should be omitted from normal logs. Connection and data-farm
status codes are owned by the supervisor and should not be listed here. This
setting does not suppress supervisor actions or the optional raw ``broker.log``
audit trail.

User-Provided Configuration File
================================

The easiest way to create an override YAML file is to copy the default file and
modify the desired values. Haymaker must be directed to the location of the
overridden config file in one of two ways:

* From the command line using the ``--file`` or ``-f`` option.
* Via environment variables: ``HAYMAKER_HAYMAKER_CONFIG_OVERRIDES`` for live
  execution or ``HAYMAKER_DATALOADER_CONFIG_OVERRIDES`` for the dataloader.

The live command always takes the strategy file path as its first positional
argument:

.. code-block:: bash
   :caption: Live execution with a YAML override

   haymaker strategy.py --file live_config.yaml

The dataloader command uses its optional positional argument as the source file
for contracts:

.. code-block:: bash
   :caption: Dataloader with a YAML override

   dataloader contracts.csv --file dataloader_config.yaml

Environment Variables
=====================

All Haymaker-related environment variables are prefixed with ``HAYMAKER_``. This prefix is removed when reading the variables, signaling to the framework which variables to load into its configuration environment. This prevents Haymaker from importing unrelated variables.

.. note::
   Environment variable names are **case-insensitive**.

.. warning::
   For nested variables, only top-level settings can be overridden via CLI or environment variables. These are intended for quick overrides; the framework is primarily configured via YAML files.

Use environment-selected YAML files when the same command should run with a
stable deployment-specific configuration:

.. code-block:: bash
   :caption: Selecting live configuration from the environment

   export HAYMAKER_HAYMAKER_CONFIG_OVERRIDES=/path/to/live_config.yaml
   haymaker strategy.py

.. code-block:: bash
   :caption: Selecting dataloader configuration from the environment

   export HAYMAKER_DATALOADER_CONFIG_OVERRIDES=/path/to/dataloader_config.yaml
   dataloader contracts.csv

Top-level scalar values can also be overridden directly through environment
variables. For example, ``HAYMAKER_LOGGING_PATH=/tmp/haymaker.log`` provides a
quick override for ``logging_path``.

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

Use ``-s`` or ``--set-option`` to temporarily override top-level parameters.
This is useful for short-lived operational overrides:

.. code-block:: bash
   :caption: Overriding a parameter via CLI

   haymaker strategy.py --set-option logging_path /tmp/haymaker.log

For dataloader, the source file is the positional argument:

.. code-block:: bash
   :caption: Running dataloader from a source file

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
https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml

specifying the source file with defined contracts, data type, frequency, etc.
