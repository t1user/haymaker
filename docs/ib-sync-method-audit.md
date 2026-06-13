# ib_insync synchronous method audit

This audit tracks synchronous `ib_insync` calls that were reviewed for async
replacement.  It excludes the experimental dashboard.

## Replaced

- `haymaker/handlers.py`: `IB.accountSummary()` now awaits
  `IB.accountSummaryAsync()` in the connected event handler.
- `haymaker/controller/sync_routines.py`: `IB.reqCompletedOrders()` now awaits
  `IB.reqCompletedOrdersAsync()` in `OrderSync.verify()`.
- `haymaker/controller/controller.py`: `IB.qualifyContracts()` now awaits
  `IB.qualifyContractsAsync()` in the nuke close path.

## Not replaced

- `haymaker/app.py` and `haymaker/dataloader/connect.py`: `IB.run()` starts the
  `ib_insync` event loop and has no async replacement.
- `haymaker/dataloader/connect.py`, `haymaker/dataloader/dataloader.py`, and
  `haymaker/timeout.py`: `IB.disconnect()` is a synchronous state-clearing API
  with no async replacement.
- `haymaker/dataloader/connect.py`: `IB.connect()` has `connectAsync()`, but the
  current reconnect helper is synchronous and is called directly from
  synchronous event handlers.  Converting it would require changing that
  dataloader connection mode rather than making a local call-site substitution.
- `haymaker/trader.py`, `haymaker/controller/controller.py`, and
  `haymaker/controller/terminator.py`: `IB.placeOrder()` has no async
  replacement; it returns a live `Trade` object that is updated by events.
- `haymaker/trader.py` and `haymaker/controller/terminator.py`:
  `IB.cancelOrder()` and `IB.reqGlobalCancel()` have no async replacements.
- `haymaker/streamers.py` and `haymaker/controller/future_roller.py`:
  `IB.reqMktData()`, `IB.reqRealTimeBars()`, and `IB.reqTickByTickData()` are
  streaming subscription APIs with no async replacements.
- `haymaker/handlers.py`: `IB.reqPnL()` starts a PnL subscription and has no
  async replacement.
- `haymaker/trader.py`, `haymaker/controller/sync_routines.py`,
  `haymaker/controller/sync_brackets.py`, `haymaker/controller/terminator.py`,
  `haymaker/controller/controller.py`, `haymaker/controller/sync_coordinator.py`,
  `haymaker/handlers.py`, and `haymaker/manager.py`: `IB.openTrades()`,
  `IB.trades()`, `IB.fills()`, and `IB.positions()` read local wrapper state and
  do not have same-semantics async replacements.  Fresh broker verification
  should use request APIs such as `reqPositionsAsync()` where needed, as
  `sync_coordinator` already does.
