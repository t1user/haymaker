*************
Configuration
*************

Haymaker is highly configurable. Parameters can be set in order of priority:

#. Command-line options (highest priority).
#. System environment variables.
#. User-provided configuration YAML file.
#. Default configuration YAML file. (fall-back for all variables that have not been defined otherwise)


Default Configuration Files
===========================

Sensible starting configuration, is defined in default configuration files located here:

For Execution Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/base_config.yaml 

For Dataloader Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml 

User-provided Configuration File
================================

It's easiest to create an override YAML file by making a copy of the default file and overriding the values as desired. `Haymaker` has to be pointed to the location of overriden config file. It can be done in one of two ways:

* from command line by passing `--file` or `-f` option
* via environment variable `HAYMAKER_HAYMAKER_CONFIG_OVERRIDES` or `HAYMAKER_DATALOADER_CONFIG_OVERRIDES` for execution and dataloader modules respectively

Environment Variables
=====================

All `Haymaker`-related environment variables are prefixed with `HAYMAKER_`, this prefix will be chopped off, while reading env variables, it's a way to indicate to the framework, which variables should be read, this will prevent `Haymaker` from loading unrelated variables into its config environement.

Environment variable names are **case insensitive**.

.. note::
   For nested variables, only top level can be overriden from CLI or environment. These are meant as a quick override, but in priciple the framework is configured via YAML file.
   

Overriding Defaults - Examples
==============================

To override defaults, copy a configuration file and modify desired parameters. Pass the location of new file to ``Haymaker`` via:

* Environment variable:

.. code-block:: bash

    export HAYMAKER_HAYMAKER_CONFIG_OVERRIDES=config.yaml

* Command-line argument:

.. code-block:: bash

    dataloader --file config.yaml

.. note::
    
    Command-line arguments take priority over environment variables.


Passing Key-Value Pairs from Command Line 
=========================================

Use -s or --set-option to override parameters temporarily.For example, to specify a data source:

.. code-block:: bash

    dataloader -s source my_list.csv

Options availble through command line can be listed with:

.. code-block:: bash

    module_name --help


Cli Options
-----------
Assuming your strategy is in 'you_module.py', which imports :class:`App` following options are availbel from command line.

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_other_module
   :prog: dataloader.py

Examples:


.. code-block:: bash

    python my_strategy.py -f config_overrides.yaml


will run strategy defined in ``my_strategy.py`` with configuration overrides defined in file ``config_overrides.yaml`` in the current directory.

.. code-block:: bash

    python my_strategy.py --nuke


will run emergency circut breaker, which will close all open positions, cancel all resting orders and not open any new positions.

.. code-block:: bash

    python my_strategy.py -s logging_path /path/to/log


will change location of logs to ``/path/to/log``.


Dataloader Cli Options
----------------------
Dataloader options are different than for other modules. 

.. argparse::
   :module: haymaker.config.cli_options
   :func: get_parser_for_dataloader
   :prog: dataloader.py

Examples:


.. code-block:: bash

    dataloader my_list.csv

will run dataloader to collect historical data for contracts defined in ``my_list.csv``.

.. code-block:: bash

    dataloader -f settings.yaml

will run dataloader with settings defined in ``settings.yaml`` file in the current directory, this file should be a copy of `this <https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml>`_ with desired changes indicating source file with defined contracts, type of data, frequency, etc.
