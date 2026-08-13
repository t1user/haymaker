"""Coordinate deterministic replay through a live-style Haymaker graph."""

from __future__ import annotations

import asyncio
import inspect
import math
import sys
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from contextvars import Context, ContextVar
from datetime import date, datetime, time, timezone
from typing import Any, TypeAlias, cast

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import QueueRunner
from haymaker.base import Atom
from haymaker.components.bracket_execution import (
    BracketExecutionModel,
    FixedStop,
    FlexibleTakeProfitAsStopMultiple,
    TakeProfitAsStopMultiple,
)
from haymaker.components.streamers import HistoricalDataStreamer, Streamer
from haymaker.durationStr_converters import durationStr_to_offset
from haymaker.order_defaults import OrderDefaults

from .data import (
    BacktestDataRepository,
    BacktestDataStore,
    ReplayTimestamp,
    StoredSeries,
)
from .exceptions import (
    BacktestConfigurationError,
    BacktestStrategyError,
    UnsupportedBacktestFeatureError,
)
from .results import BacktestResult
from .runtime import SimulationRuntime

StrategyFactory: TypeAlias = Callable[[], object | Awaitable[object]]

_SETTLEMENT_TIMEOUT = 30.0
_MISSING = object()
_BACKTEST_TASK_SCOPE: ContextVar[object | None] = ContextVar(
    "haymaker_backtest_task_scope", default=None
)


class _EventErrorCollector:
    """Collect synchronous callback errors otherwise consumed by eventkit."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, Exception]] = []
        self._connections: list[ev.Event] = []
        self._watched: set[int] = set()

    def watch(self, event: ev.Event) -> None:
        """Attach to one event's error channel at most once."""

        error_event = event.error_event
        if error_event is None or id(error_event) in self._watched:
            return
        self._watched.add(id(error_event))
        error_event.connect(self.onEventError, keep_ref=True)
        self._connections.append(error_event)

    def onEventError(self, event: ev.Event, exception: Exception) -> None:
        """Retain one synchronous event callback failure."""

        self.errors.append((event.name(), exception))

    def raise_if_failed(self) -> None:
        """Raise the oldest callback failure with its original cause."""

        if not self.errors:
            return
        name, exception = self.errors[0]
        self.errors.clear()
        raise BacktestStrategyError(
            f"Strategy callback failed while emitting {name!r}"
        ) from exception

    def close(self) -> None:
        """Disconnect every collector callback installed for this run."""

        for event in self._connections:
            event.disconnect(self.onEventError)
        self._connections.clear()
        self._watched.clear()


class _TaskTracker:
    """Track eventkit-created tasks without awaiting ambient application work."""

    def __init__(self) -> None:
        self._tracked: set[asyncio.Future[Any]] = set()
        self._errors: list[BaseException] = []

    def observe(self, task: asyncio.Future[Any]) -> None:
        """Observe one simulation-owned task from the event-loop factory."""

        if task in self._tracked:
            return
        self._tracked.add(task)
        task.add_done_callback(self._on_done)

    def capture(self, runners: Iterable[QueueRunner[Any]]) -> None:
        """Observe queue workers created before the loop factory saw them."""

        for runner in runners:
            if runner._worker_task is not None:
                self.observe(runner._worker_task)

    def _on_done(self, task: asyncio.Future[Any]) -> None:
        """Consume and retain a task exception for deterministic propagation."""

        self._tracked.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            self._errors.append(exception)

    def pending(self, runners: Iterable[QueueRunner[Any]]) -> list[asyncio.Future[Any]]:
        """Return unfinished tracked tasks excluding queue workers."""

        workers = {
            runner._worker_task for runner in runners if runner._worker_task is not None
        }
        return [
            task for task in self._tracked if not task.done() and task not in workers
        ]

    async def cancel_pending(self, runners: Iterable[QueueRunner[Any]]) -> None:
        """Cancel unfinished non-worker callbacks during simulation cleanup."""

        pending = self.pending(runners)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def raise_if_failed(self) -> None:
        """Raise the oldest asynchronous callback failure."""

        if not self._errors:
            return
        exception = self._errors.pop(0)
        raise BacktestStrategyError(
            "Asynchronous strategy callback failed"
        ) from exception


class Backtester:
    """Replay one dataloader library through a newly constructed strategy graph.

    Args:
        store: Read-only async datastore configured for one dataloader library.
        start: Optional first replay point. Earlier rows remain available as
            streamer warmup history.
        end: Optional final replay point.
        initial_cash: Opening cash used by the final marked result.
        slippage_ticks: Adverse tick slippage applied to every simulated fill.
        order_defaults: Optional real framework order defaults.
        futures_roll_bdays: ACTIVE futures selection lead time.
        futures_roll_margin_bdays: NEXT futures selection lead time.

    The strategy factory is invoked only after a passive simulated
    :class:`~haymaker.runtime.RuntimeContext` has been installed. It must build
    a fresh graph rooted in one or more
    :class:`~haymaker.components.HistoricalDataStreamer` instances.
    """

    def __init__(
        self,
        store: BacktestDataStore,
        *,
        start: ReplayTimestamp | None = None,
        end: ReplayTimestamp | None = None,
        initial_cash: float = 100_000.0,
        slippage_ticks: float = 0.0,
        order_defaults: OrderDefaults | None = None,
        futures_roll_bdays: int = 3,
        futures_roll_margin_bdays: int = 3,
    ) -> None:
        if not isinstance(store, BacktestDataStore):
            raise TypeError("store must implement the async BacktestDataStore protocol")
        self.store = store
        self.start = start
        self.end = end
        self.initial_cash = self._finite(initial_cash, "initial_cash")
        self.slippage_ticks = self._finite(slippage_ticks, "slippage_ticks")
        if self.slippage_ticks < 0:
            raise ValueError("slippage_ticks must not be negative")
        if order_defaults is not None and not isinstance(order_defaults, OrderDefaults):
            raise TypeError("order_defaults must be an OrderDefaults or None")
        self.order_defaults = order_defaults
        self.futures_roll_bdays = self._nonnegative_int(
            futures_roll_bdays, "futures_roll_bdays"
        )
        self.futures_roll_margin_bdays = self._nonnegative_int(
            futures_roll_margin_bdays, "futures_roll_margin_bdays"
        )

    async def run(self, strategy_factory: StrategyFactory) -> BacktestResult:
        """Construct a fresh strategy and replay every observed timestamp.

        Args:
            strategy_factory: Callable that constructs and connects the strategy
                graph. An async factory is accepted, although graph construction
                itself is normally synchronous.

        Returns:
            Immutable fills, final positions, and marked PnL.

        Raises:
            BacktestConfigurationError: If the strategy or data cannot form one
                unambiguous replay.
            BacktestStrategyError: If a graph callback fails.
            UnsupportedBacktestFeatureError: If the graph requests a live-only
                behavior outside the experimental simulation subset.
        """

        if not callable(strategy_factory):
            raise TypeError("strategy_factory must be callable")

        previous_runtime = Atom.__dict__.get("runtime", _MISSING)
        existing_streamer_ids = {id(streamer) for streamer in Streamer.instances}
        existing_queue_ids = {id(runner) for runner in QueueRunner._instances}
        runtime: SimulationRuntime | None = None
        collector = _EventErrorCollector()
        tracker = _TaskTracker()
        loop = asyncio.get_running_loop()
        previous_task_factory = loop.get_task_factory()
        task_scope = object()
        task_scope_token = _BACKTEST_TASK_SCOPE.set(task_scope)

        def tracking_task_factory(
            event_loop: asyncio.AbstractEventLoop,
            coroutine: Coroutine[Any, Any, Any],
            context: Context | None = None,
        ) -> asyncio.Future[Any]:
            """Chain the loop factory and observe only this replay's tasks."""

            if previous_task_factory is None:
                task: asyncio.Future[Any] = asyncio.Task(
                    coroutine,
                    loop=event_loop,
                    context=context,
                )
            elif context is None:
                task = previous_task_factory(event_loop, coroutine)
            else:
                task = cast(Any, previous_task_factory)(
                    event_loop,
                    coroutine,
                    context=context,
                )
            if _BACKTEST_TASK_SCOPE.get() is task_scope:
                tracker.observe(task)
            return task

        loop.set_task_factory(
            cast(
                Callable[
                    [asyncio.AbstractEventLoop, Coroutine[Any, Any, Any]],
                    asyncio.Future[Any],
                ],
                tracking_task_factory,
            )
        )

        try:
            runtime = SimulationRuntime.create(
                initial_cash=self.initial_cash,
                slippage_ticks=self.slippage_ticks,
                order_defaults=self.order_defaults,
                futures_roll_bdays=self.futures_roll_bdays,
                futures_roll_margin_bdays=self.futures_roll_margin_bdays,
            )
            factory_result = strategy_factory()
            if inspect.isawaitable(factory_result):
                await factory_result

            streamers = self._new_historical_streamers(existing_streamer_ids)
            self._validate_request_identity(streamers)
            atoms = self._walk_graph(streamers)
            self._validate_supported_graph(atoms)
            self._watch_graph_events(collector, atoms, runtime)

            repository = BacktestDataRepository(
                self.store,
                start=self.start,
                end=self.end,
            )
            repository.register(*runtime.contract_registry.blueprints)
            await repository.load()
            if not repository.timestamps:
                raise BacktestConfigurationError(
                    "No stored bars fall within the requested replay range"
                )
            self._validate_unique_contracts(repository.series)
            self._validate_tick_metadata(repository, atoms)
            runtime.ib.configure_sources(repository.series)

            await self._replay(
                runtime,
                repository,
                streamers,
                atoms,
                existing_queue_ids,
                collector,
                tracker,
            )
            return runtime.ib.result()
        finally:
            active_exception = sys.exception()
            if loop.get_task_factory() is tracking_task_factory:
                loop.set_task_factory(previous_task_factory)
            _BACKTEST_TASK_SCOPE.reset(task_scope_token)
            await tracker.cancel_pending(self._simulation_queues(existing_queue_ids))
            collector.close()
            cleanup_errors = await self._cleanup_queues(existing_queue_ids)
            if runtime is not None:
                try:
                    await runtime.close()
                except Exception as exception:  # pragma: no cover - defensive boundary
                    cleanup_errors.append(exception)
            Streamer.instances[:] = [
                streamer
                for streamer in Streamer.instances
                if id(streamer) in existing_streamer_ids
            ]
            if previous_runtime is _MISSING:
                if "runtime" in Atom.__dict__:
                    delattr(Atom, "runtime")
            else:
                Atom.runtime = previous_runtime  # type: ignore[assignment]
            if cleanup_errors and active_exception is None:
                raise BacktestStrategyError(
                    "Backtest queue cleanup failed"
                ) from BaseExceptionGroup("Backtest cleanup failures", cleanup_errors)

    async def _replay(
        self,
        runtime: SimulationRuntime,
        repository: BacktestDataRepository,
        streamers: Sequence[HistoricalDataStreamer],
        atoms: Sequence[Atom],
        existing_queue_ids: set[int],
        collector: _EventErrorCollector,
        tracker: _TaskTracker,
    ) -> None:
        """Run registry, fill, publication, and settlement phases."""

        histories: dict[int, tuple[int, ibi.BarDataList]] = {}
        current_day: date | None = None
        initialized = False

        for timestamp in repository.timestamps:
            available = repository.series_at(timestamp)
            # Startup and Contract-change hooks run before this bar is filled,
            # but they still need the current observed-session envelope. Keep
            # the broker submission clock at the preceding point (or None for
            # the first point) so startup orders are eligible at this bar.
            runtime.ib.set_session_contracts(series.contract for series in available)
            replay_day = self._calendar_day(timestamp)
            if replay_day != current_day:
                before = self._resolved_contract_ids(atoms) if initialized else {}
                self._reset_registry(runtime, repository, timestamp)
                after = self._resolved_contract_ids(atoms)
                if initialized:
                    self._reject_open_futures_roll(runtime)
                if not initialized or before != after:
                    self._start_graph(streamers)
                    tracker.capture(self._simulation_queues(existing_queue_ids))
                    await self._settle(
                        existing_queue_ids,
                        collector,
                        tracker,
                    )
                initialized = True
                current_day = replay_day

            runtime.ib.set_current_time(timestamp)
            self._watch_trade_events(collector, runtime.ib.openTrades())
            for series in available:
                bar = series.bar_at(timestamp)
                assert bar is not None
                await runtime.ib.process_bar(
                    series.contract,
                    bar,
                    series.metadata,
                )
                tracker.capture(self._simulation_queues(existing_queue_ids))
            await self._settle(existing_queue_ids, collector, tracker)

            for streamer in streamers:
                contract = streamer.contract
                if contract is None:
                    raise BacktestConfigurationError(
                        f"{streamer!s} has no resolved contract"
                    )
                try:
                    series = repository.series_for(contract)
                except LookupError as exception:
                    raise BacktestConfigurationError(
                        f"No replay series resolves {streamer!s} contract {contract}"
                    ) from exception
                bar = series.bar_at(timestamp)
                if bar is None:
                    continue
                history = self._streamer_history(
                    histories,
                    streamer,
                    series,
                    timestamp,
                    bar,
                )
                streamer.on_new_bar(history)
                tracker.capture(self._simulation_queues(existing_queue_ids))
            await self._settle(existing_queue_ids, collector, tracker)

            runtime.ib.mark_to_market(
                {
                    series.contract: float(bar.close)
                    for series in available
                    if (bar := series.bar_at(timestamp)) is not None
                }
            )

    @staticmethod
    def _new_historical_streamers(
        existing_streamer_ids: set[int],
    ) -> tuple[HistoricalDataStreamer, ...]:
        """Return and validate streamers constructed by this strategy factory."""

        created = tuple(
            streamer
            for streamer in Streamer.instances
            if id(streamer) not in existing_streamer_ids
        )
        if not created:
            raise BacktestConfigurationError(
                "strategy_factory must construct a HistoricalDataStreamer graph"
            )
        unsupported = [
            type(streamer).__name__
            for streamer in created
            if not isinstance(streamer, HistoricalDataStreamer)
        ]
        if unsupported:
            raise UnsupportedBacktestFeatureError(
                "Only HistoricalDataStreamer sources are supported, not: "
                + ", ".join(unsupported)
            )
        return tuple(
            streamer
            for streamer in created
            if isinstance(streamer, HistoricalDataStreamer)
        )

    @staticmethod
    def _validate_request_identity(
        streamers: Sequence[HistoricalDataStreamer],
    ) -> None:
        """Require all sources to consume the supplied single store library."""

        identities = {
            (streamer.barSizeSetting, streamer.whatToShow, streamer.useRTH)
            for streamer in streamers
        }
        if len(identities) != 1:
            raise BacktestConfigurationError(
                "One Backtester store can serve only one bar size, data type, "
                "and RTH policy per run"
            )

    @staticmethod
    def _walk_graph(roots: Sequence[Atom]) -> tuple[Atom, ...]:
        """Return graph nodes reachable through normal Atom connections."""

        result: list[Atom] = []
        pending = list(reversed(roots))
        seen: set[int] = set()
        while pending:
            atom = pending.pop()
            if id(atom) in seen:
                continue
            seen.add(id(atom))
            result.append(atom)
            pending.extend(reversed(atom._downstream_targets))
        return tuple(result)

    @staticmethod
    def _validate_supported_graph(atoms: Sequence[Atom]) -> None:
        """Reject bracket legs whose generated IB orders cannot be simulated."""

        for atom in atoms:
            if not isinstance(atom, BracketExecutionModel):
                continue
            if type(atom.stop) is not FixedStop:
                raise UnsupportedBacktestFeatureError(
                    "Backtests currently support only FixedStop protective legs"
                )
            allowed_take_profit = {
                TakeProfitAsStopMultiple,
                FlexibleTakeProfitAsStopMultiple,
            }
            if (
                atom.take_profit is not None
                and type(atom.take_profit) not in allowed_take_profit
            ):
                raise UnsupportedBacktestFeatureError(
                    "Backtests currently support only fixed LMT take-profit legs"
                )

    @staticmethod
    def _watch_graph_events(
        collector: _EventErrorCollector,
        atoms: Sequence[Atom],
        runtime: SimulationRuntime,
    ) -> None:
        """Observe graph and broker event error channels."""

        for atom in atoms:
            for name in (*Atom.events, "_contractChangedEvent"):
                event = getattr(atom, name, None)
                if isinstance(event, ev.Event):
                    collector.watch(event)
        for name in runtime.ib.events:
            event = getattr(runtime.ib, name)
            if isinstance(event, ev.Event):
                collector.watch(event)

    @staticmethod
    def _watch_trade_events(
        collector: _EventErrorCollector,
        trades: Iterable[ibi.Trade],
    ) -> None:
        """Observe dynamic order event error channels before a fill phase."""

        for trade in trades:
            for name in trade.events:
                event = getattr(trade, name)
                if isinstance(event, ev.Event):
                    collector.watch(event)

    @staticmethod
    def _validate_unique_contracts(series: Sequence[StoredSeries]) -> None:
        """Reject duplicate persisted series for one concrete Contract."""

        seen: dict[int, str] = {}
        for item in series:
            previous = seen.setdefault(item.contract.conId, item.key)
            if previous != item.key:
                raise BacktestConfigurationError(
                    f"Collections {previous!r} and {item.key!r} share conId "
                    f"{item.contract.conId}"
                )

    def _validate_tick_metadata(
        self,
        repository: BacktestDataRepository,
        atoms: Sequence[Atom],
    ) -> None:
        """Require an explicit positive tick size when execution needs one."""

        if not self.slippage_ticks and not any(
            isinstance(atom, BracketExecutionModel) for atom in atoms
        ):
            return
        for series in repository.series:
            try:
                raw = repository.require_metadata(series.contract, "minTick")
                value = float(raw)
            except (KeyError, TypeError, ValueError) as exception:
                raise BacktestConfigurationError(
                    f"Contract {series.contract.localSymbol or series.key} requires "
                    "positive minTick or min_tick metadata"
                ) from exception
            if not math.isfinite(value) or value <= 0:
                raise BacktestConfigurationError(
                    f"Contract {series.contract.localSymbol or series.key} requires "
                    "positive minTick or min_tick metadata"
                )

    @staticmethod
    def _reset_registry(
        runtime: SimulationRuntime,
        repository: BacktestDataRepository,
        timestamp: ReplayTimestamp,
    ) -> None:
        """Rebuild framework selectors using datastore details and replay time."""

        runtime.contract_registry.today = Backtester._registry_datetime(timestamp)
        details = [
            repository.details_for_blueprint(blueprint)
            for blueprint in runtime.contract_registry.blueprints
        ]
        runtime.contract_registry.reset_data(details)

    @staticmethod
    def _start_graph(streamers: Sequence[HistoricalDataStreamer]) -> None:
        """Propagate framework startup and Contract changes from each source."""

        for streamer in streamers:
            streamer.onStart({})

    @staticmethod
    def _resolved_contract_ids(atoms: Sequence[Atom]) -> dict[int, int | None]:
        """Snapshot each Contract-owning node's current concrete identity."""

        result: dict[int, int | None] = {}
        for atom in atoms:
            if atom._contract_blueprint is None:
                continue
            contract = atom.contract
            result[id(atom)] = contract.conId if contract is not None else None
        return result

    @staticmethod
    def _reject_open_futures_roll(runtime: SimulationRuntime) -> None:
        """Fail before silently carrying positions or orders off the active chain."""

        allowed = {
            contract.conId for contract in runtime.contract_registry.current_contracts
        }
        held = [
            contract
            for contract in runtime.book.logical_positions()
            if contract.secType == "FUT" and contract.conId not in allowed
        ]
        working = [
            info.trade.contract
            for info in runtime.book.active_orders()
            if info.trade.contract.secType == "FUT"
            and info.trade.contract.conId not in allowed
        ]
        if held or working:
            names = sorted(
                {
                    contract.localSymbol or contract.symbol
                    for contract in (*held, *working)
                }
            )
            raise UnsupportedBacktestFeatureError(
                "Rolling an open futures position or working order is not "
                "supported: " + ", ".join(names)
            )

    @staticmethod
    def _streamer_history(
        histories: dict[int, tuple[int, ibi.BarDataList]],
        streamer: HistoricalDataStreamer,
        series: StoredSeries,
        timestamp: ReplayTimestamp,
        bar: ibi.BarData,
    ) -> ibi.BarDataList:
        """Return cumulative warmup/current bars for one source and Contract."""

        state = histories.get(id(streamer))
        if state is None or state[0] != series.contract.conId:
            history = series.bars_through(timestamp)
            history = Backtester._trim_initial_history(history, streamer, timestamp)
            history.contract = series.contract
            histories[id(streamer)] = (series.contract.conId, history)
            return history
        history = state[1]
        if not history or bar.date > history[-1].date:
            history.append(bar)
        return history

    @staticmethod
    def _trim_initial_history(
        history: ibi.BarDataList,
        streamer: HistoricalDataStreamer,
        timestamp: ReplayTimestamp,
    ) -> ibi.BarDataList:
        """Apply a streamer's requested warmup window to stored earlier rows."""

        duration = streamer.durationStr
        if isinstance(duration, int):
            if duration <= 0:
                raise BacktestConfigurationError(
                    "HistoricalDataStreamer durationStr count must be positive"
                )
            selected = history[-duration:]
        elif isinstance(duration, str):
            try:
                cutoff = pd.Timestamp(timestamp) - durationStr_to_offset(duration)
            except (KeyError, TypeError, ValueError) as exception:
                raise BacktestConfigurationError(
                    f"Invalid HistoricalDataStreamer durationStr {duration!r}"
                ) from exception
            cutoff_value: date | datetime
            if isinstance(timestamp, datetime):
                cutoff_value = cutoff.to_pydatetime()
            else:
                cutoff_value = cutoff.date()
            selected = [bar for bar in history if bar.date >= cutoff_value]
        else:
            raise BacktestConfigurationError(
                "HistoricalDataStreamer durationStr must be a string or integer"
            )
        trimmed = ibi.BarDataList(selected)
        trimmed.contract = history.contract
        return trimmed

    async def _settle(
        self,
        existing_queue_ids: set[int],
        collector: _EventErrorCollector,
        tracker: _TaskTracker,
    ) -> None:
        """Drain adapter-owned graph queues and scheduled event callbacks."""

        stable_passes = 0
        for _ in range(100):
            runners = self._simulation_queues(existing_queue_ids)
            tracker.capture(runners)
            await asyncio.sleep(0)
            tracker.capture(runners)

            joins = [runner._queue.join() for runner in runners]
            if joins:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*joins),
                        timeout=_SETTLEMENT_TIMEOUT,
                    )
                except TimeoutError as exception:
                    raise BacktestStrategyError(
                        "Strategy queues did not settle within 30 seconds"
                    ) from exception

            for runner in runners:
                if runner._processing_error is not None:
                    raise BacktestStrategyError(
                        f"{runner!s} failed while processing replay data"
                    ) from runner._processing_error

            tracker.capture(runners)
            pending = tracker.pending(runners)
            if pending:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=_SETTLEMENT_TIMEOUT,
                    )
                except TimeoutError as exception:
                    raise BacktestStrategyError(
                        "Asynchronous strategy callbacks did not settle within "
                        "30 seconds"
                    ) from exception

            await asyncio.sleep(0)
            runners = self._simulation_queues(existing_queue_ids)
            tracker.capture(runners)
            collector.raise_if_failed()
            tracker.raise_if_failed()
            if all(runner.qsize() == 0 for runner in runners) and not tracker.pending(
                runners
            ):
                stable_passes += 1
                if stable_passes == 2:
                    return
            else:
                stable_passes = 0
        raise BacktestStrategyError(
            "Strategy graph did not reach a stable replay state"
        )

    @staticmethod
    def _simulation_queues(existing_queue_ids: set[int]) -> list[QueueRunner[Any]]:
        """Return QueueRunners created within this backtest scope."""

        return [
            runner
            for runner in QueueRunner._instances
            if id(runner) not in existing_queue_ids
        ]

    @staticmethod
    async def _cleanup_queues(existing_queue_ids: set[int]) -> list[BaseException]:
        """Close only QueueRunners created by this strategy/runtime."""

        runners = Backtester._simulation_queues(existing_queue_ids)
        if not runners:
            return []
        results = await asyncio.gather(
            *(runner.close() for runner in runners),
            return_exceptions=True,
        )
        return [result for result in results if isinstance(result, BaseException)]

    @staticmethod
    def _calendar_day(timestamp: ReplayTimestamp) -> date:
        """Return the replay calendar day without consulting wall-clock time."""

        return timestamp.date() if isinstance(timestamp, datetime) else timestamp

    @staticmethod
    def _registry_datetime(timestamp: ReplayTimestamp) -> datetime:
        """Return the timezone-naive UTC timestamp expected by selectors."""

        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                aware = timestamp.replace(tzinfo=timezone.utc)
            else:
                aware = timestamp.astimezone(timezone.utc)
            return aware.replace(tzinfo=None)
        return datetime.combine(timestamp, time.min)

    @staticmethod
    def _finite(value: float, name: str) -> float:
        """Normalize one finite public numeric option."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        return normalized

    @staticmethod
    def _nonnegative_int(value: int, name: str) -> int:
        """Validate one non-negative integer selector option."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must not be negative")
        return value


__all__ = ["Backtester", "StrategyFactory"]
