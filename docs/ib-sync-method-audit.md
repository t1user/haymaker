# ib_insync synchronous method audit

This audit tracks synchronous `ib_insync` calls that were reviewed for async
replacement.  It excludes the experimental dashboard.

## Replaced

- `haymaker/handlers.py`: `IB.accountSummary()` now awaits
  `IB.accountSummaryAsync()` in the connected event handler.

## Not replaced

- `haymaker/trader.py`, `haymaker/controller/controller.py`, and
  `haymaker/controller/reset.py`: `IB.placeOrder()` has no async
  replacement; it returns a live `Trade` object that is updated by events.
- `haymaker/trader.py` and `haymaker/controller/reset.py`:
  `IB.cancelOrder()` and `IB.reqGlobalCancel()` have no async replacements.
  Emergency reset uses concrete Contracts from the broker position cache
  without an additional qualification request.
- `haymaker/streamers.py` and `haymaker/controller/future_roller.py`:
  `IB.reqMktData()`, `IB.reqRealTimeBars()`, and `IB.reqTickByTickData()` are
  streaming subscription APIs with no async replacements.
- `haymaker/handlers.py`: `IB.reqPnL()` starts a PnL subscription and has no
  async replacement.
- `haymaker/trader.py`, `haymaker/controller/sync_routines.py`,
  `haymaker/controller/sync_brackets.py`, `haymaker/controller/reset.py`,
  `haymaker/controller/controller.py`, `haymaker/controller/sync_coordinator.py`,
  `haymaker/handlers.py`, and `haymaker/manager.py`: `IB.openTrades()`,
  `IB.trades()`, `IB.fills()`, and `IB.positions()` read local wrapper state and
  do not have same-semantics async replacements.  Fresh broker verification
  should use request APIs such as `reqPositionsAsync()` where needed, as
  `sync_coordinator` already does.
