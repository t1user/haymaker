********************
Composition Examples
********************

Strategy modules are imported after ``LiveRuntime`` has installed a ready
runtime context. Construct components at module scope and keep the resulting
pipeline referenced by the module.

One-to-one bracketed strategy
=============================

This flow turns a dataframe STATE signal into one independently attributed
position episode:

.. code-block:: python

   import ib_insync as ibi
   import pandas as pd

   from haymaker.base import Pipe
   from haymaker.components import (
       BracketExecutionModel,
       FuturesPandasAggregator,
       FixedSizeAllocator,
       FixedStop,
       HistoricalDataStreamer,
       BinarySignalProcessor,
       PandasSignalModel,
       PortfolioWrapper,
       SignalType,
   )


   CONTRACT = ibi.ContFuture("ES", exchange="CME")
   SOURCE = "trend.es.daily"


   class TrendModel(PandasSignalModel):
       """Emit long/flat STATE from a calculated dataframe."""

       def df(self, data: pd.DataFrame) -> pd.DataFrame:
           result = data.copy()
           result["fast"] = result["close"].rolling(20).mean()
           result["slow"] = result["close"].rolling(50).mean()
           result["atr"] = (
               result["high"] - result["low"]
           ).rolling(20).mean()
           result["signal"] = (result["fast"] > result["slow"]).astype(float)
           return result

   streamer = HistoricalDataStreamer(
       contract=CONTRACT,
       durationStr=100,
       barSizeSetting="1 day",
       whatToShow="TRADES",
       datastore=True,
   )
   frames = FuturesPandasAggregator()
   signals = TrendModel(
       SOURCE,
       CONTRACT,
       SignalType.STATE,
       persistence=True,
   )
   processor = BinarySignalProcessor(respect_blocked_direction=True)
   allocation = PortfolioWrapper(FixedSizeAllocator(1))
   execution = BracketExecutionModel(
       SOURCE,
       name="trend_es_brackets",
       stop=FixedStop(3),
   )

   strategy: Pipe = streamer.pipe(
       frames,
       signals,
       processor,
       allocation,
       execution,
   )

``datastore=True`` on the streamer and the aggregator's default constructor
resolve the same cached runtime store for this bar size, data type, and RTH
policy. The aggregator restores and periodically saves the complete DataFrame;
on a later process start, the streamer reads the persisted endpoint and
requests only the missing broker history. See
:ref:`advanced-storage-configuration` when the runtime defaults are
unsuitable.

``TrendModel`` places ``atr`` in Signal metadata. ``FixedSizeAllocator``
preserves it in the PositionTarget, and ``FixedStop`` consumes it after the
entry is completely filled. Book assigns a fresh ``position_id`` to the
episode, while Controller carries source and episode attribution to every
order and blotter row.

An EVENT model uses ``SignalType.EVENT``. Its zero values are ignored by the
processor instead of requesting a flat position. Repeating a non-zero EVENT
represents another event, although one-to-one processing still suppresses
actions that are already satisfied by effective quantity.

Account-wide direct Portfolio
=============================

A direct Portfolio receives raw Signals from several sources and may change
several execution targets after any one input:

.. code-block:: python

   from collections.abc import Iterable

   import ib_insync as ibi

   from haymaker.components import (
       ExecutionRouter,
       ExecutionRule,
       Portfolio,
       PositionTarget,
       SerialTargetExecutionModel,
       Signal,
       symbol_is,
   )


   class EqualWeightPortfolio(Portfolio):
       """Keep the latest STATE direction and emit all registered targets."""

       def __init__(
           self,
           contracts: dict[str, ibi.Contract],
           size: float,
       ) -> None:
           super().__init__(sources=contracts)
           self.contracts = contracts
           self.size = size
           self.latest: dict[str, float] = {}

       def process(self, signal: Signal) -> Iterable[PositionTarget]:
           self.latest[signal.source_key] = signal.value
           return (
               PositionTarget(
                   contract=self.contracts[source_key],
                   target_quantity=(
                       0
                       if value == 0
                       else self.size if value > 0 else -self.size
                   ),
               )
               for source_key, value in self.latest.items()
           )


   contracts = {
       "trend.es": ibi.Future(
           conId=123,
           symbol="ES",
           exchange="CME",
       ),
       "trend.nq": ibi.Future(
           conId=456,
           symbol="NQ",
           exchange="CME",
       ),
   }
   portfolio = EqualWeightPortfolio(contracts, size=2)
   index_model = SerialTargetExecutionModel(name="index_targets")
   router = ExecutionRouter(
       rules=[
           ExecutionRule(
               predicate=symbol_is("ES"),
               model=index_model,
           ),
           ExecutionRule(
               predicate=symbol_is("NQ"),
               model=index_model,
           ),
       ]
   )
   portfolio.connect(router)

   es_signal_model.connect(portfolio)
   nq_signal_model.connect(portfolio)

The example Portfolio implements its own immediate STATE recomputation policy.
A synchronized implementation could instead require ``as_of``, wait for all
registered sources, define duplicate/late input handling, and persist its
normalized mapping under a stable ``portfolio_key`` through Book.

Each target is an absolute setpoint for its concrete Contract. This example
uses distinct Contracts for its two sources; a Portfolio combining several
sources on the same Contract must aggregate their allocations before emitting
one target. Source allocation state is not inferred from net broker fills. If a newer target arrives while an
adjustment order works, it retains only the newer target and
re-evaluates after completion. Router rules are evaluated in declaration order;
current rules remain authoritative. On restart, an active TARGET_ADJUSTMENT
must still select the model that submitted it or the Router blocks locally;
idle recovered targets are reassigned to the model selected by current rules.
