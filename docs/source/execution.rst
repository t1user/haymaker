****************
Execution Module
****************

.. toctree::
    :maxdepth: 3

Haymaker trading algorithms consist of a series of Atom components, each implementing a step in a trading algorithm. These components are piped together in an event-driven fashion.

Common trading steps include:

* Receiving price data.

* Processing/aggregating/filtering data.

* Generating trading signals.

* Portfolio management.

* Risk control.

* Execution management.

Each processing component (which is called: Atom) inherits from :class:`Atom` .

Atom Object 
===========

.. autoclass:: haymaker.base.Atom
    :members:


Auxiliary Objects
-----------------

.. autoclass:: haymaker.base.Pipe
    :members:


.. autoclass:: haymaker.base.Details
    :members:

Example Usage 
=============

.. literalinclude:: includes/example.py 


.. warning::
    **NOT AN INVESTMENT ADVICE**

    This example is only meant to illustrate how to use Haymaker framework. It is unlikely to produce favourable investment outcomes.
