

`Haymaker` is a Python framework designed for algorithmic trading via `Interactive Brokers <https://ibkrcampus.com/ibkr-api-page/twsapi-doc/>`_. Built on top of `ib_insync <https://ib-insync.readthedocs.io/readme.html>`_ library, it provides essential scafolling typically required for building live trading strategy components. 

.. note::
    `Haymaker` works as an extension to `ib_insync`, throughout this documentation it is assumed that users are familiar with this library.

Functionality
=============

* **Strategy Execution**: Run your stategy in production. Haymaker is designed to serve as a long-running process, robustly recovering from any faults regardless whether they are due to local crashes, broker issues or internet disconnections.
* **Historical Data Download**: Work within IB's limitations to download and store any data that is available regardless of how long it takes to get it.
* **Strategy Research**: Develop your strategy using Haymaker's set of vector based tools aiding typical research pipeline.
* **Backtesting (experimental extra)**: Replay dataloader bars through the event-driven strategy framework. This simulator is distinct from the research vector backtester and is not fully functional yet.

Why Haymaker?
=============

* **Modular**: Easily customize components or build your own to suit your trading needs.
* **Minimal**: Provide only essential components, allowing you to code your strategy in preferred style.
* **Tried and Tested**: Built on top of `ib_insync`, leveraging over a decade of development community feedback. `Interactive Brokers <https://www.interactivebrokers.com/>`__ is a pioneer in algoritmic trading and still remains the leading platform both for individual and institutional investors.
* **Event Driven**: Processes market data as soon as it becomes available.

Documentation
=============

For full documentation, visit:

👉 https://t1user.github.io/haymaker/
