Installation
============

To install Haymaker, it’s recommended to use a virtual environment to isolate dependencies. Ensure you have Python (version 3.11 or later recommended) and ``git`` installed on your system.

Follow these steps to clone and install Haymaker:

.. code-block:: bash
   :caption: Cloning and installing Haymaker

   git clone https://github.com/t1user/haymaker.git
   cd haymaker
   python -m venv venv  # Create a virtual environment
   source venv/bin/activate  # Activate on Unix/macOS
   # venv\Scripts\activate  # Activate on Windows
   pip install .

Optional Dependencies
---------------------

Haymaker supports additional features through optional dependency groups. Install them as needed:

- **Jupyter Environment for Research**:

  .. code-block:: bash
     :caption: Installing with Jupyter support

     pip install .[notebooks]

  This includes tools like Jupyter notebooks for strategy research and analysis.

- **Development Tools**:

  .. code-block:: bash
     :caption: Installing with development tools

     pip install .[dev]

  This includes linters, type checkers, and testing frameworks for development.

- **All Features**:

  .. code-block:: bash
     :caption: Installing all optional dependencies

     pip install .[all]

  This installs all optional dependencies (notebooks and development tools).

.. note::
   Activate the virtual environment before running Haymaker commands or scripts. Deactivate it with ``deactivate`` when done.