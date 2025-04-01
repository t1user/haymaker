*************
Configuration
*************

Haymaker is highly configurable. Parameters can be set in the following order of priority:

#. Command-line options (highest priority).
#. System environment variables.
#. User-provided configuration YAML file.
#. Default configuration YAML file (fallback for undefined variables).

Default Configuration Files
===========================

Sensible starting configurations are defined in default configuration files located here:

- For the Execution Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/base_config.yaml 

- For the Dataloader Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml 


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

     dataloader --file config.yaml

.. note::
   Command-line arguments take precedence over environment variables.

Passing Key-Value Pairs from Command Line
=========================================

Use ``-s`` or ``--set-option`` to temporarily override parameters. For example, to specify a data source:

.. code-block:: bash
   :caption: Overriding a parameter via CLI

   dataloader -s source my_list.csv

Available command-line options can be listed with:

.. code-block:: bash
   :caption: Displaying help for CLI options

   module_name --help

CLI Options
-----------

Assuming your strategy is in ``your_module.py`` and imports :class:`haymaker.app.App`, the following options are available from the command line:

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_other_module
   :prog: dataloader.py

Examples
^^^^^^^^

.. code-block:: bash
   :caption: Running a strategy with a config override

   python my_strategy.py -f config_overrides.yaml

This runs the strategy defined in ``my_strategy.py`` with configuration overrides from ``config_overrides.yaml`` in the current directory.

.. code-block:: bash
   :caption: Triggering the emergency circuit breaker

   python my_strategy.py --nuke

This activates the emergency circuit breaker, closing all open positions, canceling resting orders, and preventing new positions.

.. code-block:: bash
   :caption: Changing the log location

   python my_strategy.py -s logging_path /path/to/log

This changes the log location to ``/path/to/log``.

Dataloader CLI Options
----------------------

Dataloader options differ from those of other modules:

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_dataloader
   :prog: dataloader.py

Examples
^^^^^^^^

.. code-block:: bash
   :caption: Collecting historical data from a CSV

   dataloader my_list.csv

This runs the dataloader to collect historical data for contracts defined in ``my_list.csv``.

.. code-block:: bash
   :caption: Running dataloader with a custom settings file

   dataloader -f settings.yaml

This runs the dataloader with settings from ``settings.yaml`` in the current directory. This file should be a modified copy of:

https://github/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml

specifying the source file with defined contracts, data type, frequency, etc.