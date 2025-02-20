*************
Configuration
*************

Haymaker is highly configurable. Parameters can be set in order of priority:

#. Default configuration YAML file.
#. User-provided configuration YAML file.
#. System environment variables.
#. Command-line options (highest priority).

Default Configuration Files
===========================

Sensible starting configuration, is defined in default configuration files located here:

For Execution Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/base_config.yaml 

For Dataloader Module: 
https://github.com/t1user/haymaker/blob/master/haymaker/config/dataloader_base_config.yaml 


Overriding Defaults
===================

To override defaults, copy a configuration file and modify desired parameters. Pass the location of new file to ``Haymaker`` via:

* Environment variable:

.. code-block:: bash

    export HAYMAKER_CONFIG_OVERRIDES=config.yaml

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