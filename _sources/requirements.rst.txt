Requirements
============

* Linux 
* Python 3.11 or higher
* A running TWS or IB Gateway application

Dependencies
============

* `ib_insync <https://ib-insync.readthedocs.io/>`_ Handles all communication with IB.
* `Eventkit <https://github.com/erdewit/eventkit>`_ Enables event-driven communication between strategy components and with broker.
* `MongoDB <https://www.mongodb.com/>`_ Stores state and user data between restarts (voluntary or otherwise).
* `Arctic <https://github.com/man-group/arctic?tab=readme-ov-file>`_ Wraps MongoDB for storing timeseries data in a fast and space efficient manner, developed by `Man group <https://www.man.com/>`_, leading global asset manager; other databases can be easily added.
* `Pandas <https://pandas.pydata.org/>`_ Used for all tools in research module; strategies can but don't have to be coded using pandas.
