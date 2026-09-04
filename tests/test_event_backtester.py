"""Focused tests for the experimental event-driven backtester."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.async_wrappers import QueueRunner
from haymaker.backtester import (
    BacktestDataRepository,
    BacktestStrategyError,
    Backtester,
    MissingBacktestDataError,
    MissingContractMetadataError,
    SimulatedIB,
    UnsupportedBacktestFeatureError,
    UnsupportedBacktestOrderError,
)
from haymaker.base import Atom, Pipe
from haymaker.components import (
    BarAggregator,
    BinarySignalProcessor,
    BracketExecutionModel,
    FixedSizeAllocator,
    FixedStop,
    HistoricalDataStreamer,
    NoFilter,
    Portfolio,
    PortfolioWrapper,
    PositionTarget,
    Signal,
    SignalCalculation,
    SignalModel,
    SignalType,
    SerialTargetExecutionModel,
    Streamer,
    TakeProfitAsStopMultiple,
    TrailingStop,
)


class MemoryBacktestStore:
    """Minimal read-only async store keyed by persisted collection name."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        metadata: dict[str, dict[str, Any]],
    ) -> None:
        self.frames = frames
        self.metadata = metadata
        self.read_keys: list[str] = []

    async def keys(self) -> list[str]:
        """Return all persisted collection names."""

        return list(self.metadata)

    async def read(
        self,
        symbol: str,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None:
        """Return a copied frame without applying backend-side bounds."""

        del start_date, end_date
        self.read_keys.append(symbol)
        frame = self.frames.get(symbol)
        return None if frame is None else frame.copy()

    async def read_metadata(self, symbol: str) -> dict[str, Any]:
        """Return copied top-level contract and execution metadata."""

        return dict(self.metadata[symbol])


class AlwaysLongSignalModel(SignalModel):
    """Emit a stateful long Signal with a fixed bracket distance."""

    def calculate_signal(self, data: ibi.BarDataList) -> SignalCalculation:
        """Use the latest completed bar as the Signal observation time."""

        observation = data[-1].date
        if not isinstance(observation, datetime):
            raise TypeError("AlwaysLongSignalModel requires intraday bars")
        return SignalCalculation(
            value=1.0,
            metadata={"atr": 1.0},
            as_of=observation,
        )


class DirectLongPortfolio(Portfolio):
    """Map each scalar Signal to one stable direct target identity."""

    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """Emit the source value as its absolute direct target."""

        if not isinstance(signal.value, float):
            raise TypeError("DirectLongPortfolio requires scalar Signals")
        return (
            PositionTarget(
                contract=signal.contract,
                target_quantity=signal.value,
                target_key=f"{signal.source_key}-target",
            ),
        )


class FailingAsyncAtom(Atom):
    """Represent an asynchronous strategy callback failure."""

    async def onData(self, data: object, *args: object) -> None:
        """Fail after eventkit schedules this callback as a Task."""

        del data, args
        raise RuntimeError("async strategy failure")


class CrossContractTarget(Atom):
    """Emit a direct target for a Contract different from the data source."""

    def __init__(self, contract: ibi.Contract) -> None:
        self.contract = contract
        super().__init__()

    def onData(self, data: object, *args: object) -> None:
        """Emit one-unit target for the currently resolved Contract."""

        del data, args
        contract = self.contract
        if contract is None:
            raise RuntimeError("CrossContractTarget has no resolved Contract")
        self.dataEvent.emit(
            PositionTarget(
                contract=contract,
                target_quantity=1,
                target_key="cross-contract-target",
            )
        )


class StartupTarget(Atom):
    """Submit one target before the first observed bar is processed."""

    def onStart(self, data: object, source: Atom | None = None) -> None:
        """Emit a one-unit target for the source's resolved Contract."""

        if source is None or source.contract is None:
            raise RuntimeError("StartupTarget requires a resolved source Contract")
        self.dataEvent.emit(
            PositionTarget(
                contract=source.contract,
                target_quantity=1,
                target_key="startup-target",
            )
        )
        super().onStart(data, source)

    def onData(self, data: object, *args: object) -> None:
        """Ignore later bars after the startup target has been submitted."""

        del data, args


def _frame(
    index: pd.DatetimeIndex,
    *,
    open_: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
) -> pd.DataFrame:
    """Return a valid dataloader-style OHLC frame."""

    size = len(index)
    open_values = open_ or [100.0] * size
    high_values = high or [value + 1.0 for value in open_values]
    low_values = low or [value - 1.0 for value in open_values]
    close_values = close or list(open_values)
    return pd.DataFrame(
        {
            "open": open_values,
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "average": close_values,
            "volume": [1.0] * size,
            "barCount": [1] * size,
        },
        index=index,
    )


def _stock(con_id: int, symbol: str) -> ibi.Stock:
    """Return one concrete qualified stock Contract."""

    return ibi.Stock(
        conId=con_id,
        symbol=symbol,
        exchange="SMART",
        currency="USD",
        localSymbol=symbol,
    )


def _future(
    con_id: int,
    local_symbol: str,
    expiry: str,
) -> ibi.Future:
    """Return one exact qualified future Contract."""

    return ibi.Future(
        conId=con_id,
        symbol="ES",
        lastTradeDateOrContractMonth=expiry,
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol=local_symbol,
        tradingClass="ES",
    )


def _metadata(contract: ibi.Contract, **extra: Any) -> dict[str, Any]:
    """Return the top-level metadata shape written by the dataloader."""

    result = ibi.util.dataclassNonDefaults(contract)
    result.update(extra)
    return result


async def test_repository_uses_union_clock_and_retains_sparse_warmup() -> None:
    """Missing contract rows define sessions while earlier rows remain warmup."""

    points = pd.date_range("2025-01-06 10:00", periods=3, freq="h", tz="UTC")
    first = _stock(101, "ONE")
    second = _stock(202, "TWO")
    store = MemoryBacktestStore(
        {
            "ONE_STK": _frame(points[[0, 2]]),
            "TWO_STK": _frame(points[[1]]),
        },
        {
            "ONE_STK": _metadata(first),
            "TWO_STK": _metadata(second),
        },
    )

    repository = await BacktestDataRepository(store, start=points[1]).load()

    assert repository.timestamps == (points[1], points[2])
    assert tuple(item.key for item in repository.series_at(points[1])) == ("TWO_STK",)
    assert repository.bar_at(first, points[1]) is None
    assert [bar.date for bar in repository.bars_through(first, points[1])] == [
        points[0]
    ]


async def test_repository_requires_exact_futures_expiry_metadata() -> None:
    """A futures month is insufficient for contract-specific replay."""

    point = pd.date_range("2025-01-06", periods=1, tz="UTC")
    future = ibi.Future(
        conId=303,
        symbol="ES",
        lastTradeDateOrContractMonth="202503",
        exchange="CME",
        currency="USD",
        localSymbol="ESH5",
    )
    store = MemoryBacktestStore(
        {"ESH5_FUT": _frame(point)},
        {"ESH5_FUT": _metadata(future)},
    )

    with pytest.raises(MissingContractMetadataError, match="exact YYYYMMDD"):
        await BacktestDataRepository(store).load()


async def test_repository_skips_clearly_unrelated_malformed_metadata() -> None:
    """An unrelated legacy collection cannot block a selected replay."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(323, "GOOD")
    store = MemoryBacktestStore(
        {"GOOD_STK": _frame(point)},
        {
            "GOOD_STK": _metadata(contract),
            "OLD_STK": {
                "secType": "STK",
                "symbol": "OLD",
                "exchange": "SMART",
                "currency": "USD",
                "localSymbol": "OLD",
            },
        },
    )
    repository = BacktestDataRepository(store)
    repository.register(ibi.Stock("GOOD", "SMART", "USD"))

    await repository.load()

    assert [series.key for series in repository.series] == ["GOOD_STK"]
    assert store.read_keys == ["GOOD_STK"]


async def test_repository_rejects_plausibly_matching_malformed_metadata() -> None:
    """Missing identity data still fails when the collection may be requested."""

    blueprint = _stock(324, "GOOD")
    store = MemoryBacktestStore(
        {},
        {
            "GOOD_STK": {
                "conId": 0,
                "secType": "STK",
                "symbol": "GOOD",
                "exchange": "SMART",
                "currency": "USD",
                "localSymbol": "GOOD",
            }
        },
    )
    repository = BacktestDataRepository(store)
    repository.register(blueprint)

    with pytest.raises(MissingContractMetadataError, match="nonzero contract conId"):
        await repository.load()


async def test_contract_blueprints_compare_derivative_identity() -> None:
    """Option strike and right cannot silently resolve a different series."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    stored = ibi.Option(
        conId=333,
        symbol="SPX",
        lastTradeDateOrContractMonth="20250321",
        strike=5100,
        right="P",
        exchange="SMART",
        currency="USD",
        localSymbol="SPXW  250321P05100000",
    )
    requested = ibi.Option(
        "SPX",
        "20250321",
        5000,
        "C",
        "SMART",
        currency="USD",
    )
    store = MemoryBacktestStore(
        {"SPXW_OPT": _frame(point)},
        {"SPXW_OPT": _metadata(stored)},
    )
    repository = BacktestDataRepository(store)
    repository.register(requested)

    with pytest.raises(MissingBacktestDataError, match="No persisted series"):
        await repository.load()

    broker = SimulatedIB()
    broker.configure_contract(stored, _metadata(stored))
    assert await broker.reqContractDetailsAsync(requested) == []
    assert await broker.qualifyContractsAsync(requested) == []


async def test_repository_preserves_missing_optional_bar_values_as_nan() -> None:
    """Missing saved values are not fabricated as valid numeric zeroes."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(353, "NAN")
    frame = _frame(point)
    frame.loc[point[0], "average"] = float("nan")
    store = MemoryBacktestStore(
        {"NAN_STK": frame},
        {"NAN_STK": _metadata(contract)},
    )

    repository = await BacktestDataRepository(store).load()
    bar = repository.bar_at(contract, point[0])

    assert bar is not None
    assert pd.isna(bar.average)


async def test_contract_details_normalize_string_tick_metadata() -> None:
    """Dataloader-compatible numeric strings reach real components as floats."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(363, "TICK")
    store = MemoryBacktestStore(
        {"TICK_STK": _frame(point)},
        {"TICK_STK": _metadata(contract, minTick="0.25")},
    )

    repository = await BacktestDataRepository(store).load()

    assert repository.contract_details(contract).minTick == 0.25
    assert isinstance(repository.contract_details(contract).minTick, float)


async def test_simulated_broker_waits_for_next_actual_contract_bar() -> None:
    """Submission-bar data and missing timestamps cannot fill an order."""

    contract = _stock(404, "GAP")
    broker = SimulatedIB(initial_cash=10_000, slippage_ticks=1)
    broker.configure_contract(
        contract,
        _metadata(
            contract,
            minTick="0.25",
            multiplier="50",
            commission="0.5",
        ),
    )
    submitted_at = datetime(2025, 1, 6, 10, tzinfo=timezone.utc)
    next_actual_bar = datetime(2025, 1, 6, 13, tzinfo=timezone.utc)
    broker.set_current_time(submitted_at)
    trade = broker.placeOrder(
        contract,
        ibi.MarketOrder("BUY", 1),
    )

    await broker.process_bar(
        contract,
        ibi.BarData(
            date=submitted_at,
            open=100,
            high=101,
            low=99,
            close=100,
        ),
    )
    assert not trade.fills

    await broker.process_bar(
        contract,
        ibi.BarData(
            date=next_actual_bar,
            open=102,
            high=103,
            low=101,
            close=102,
        ),
    )

    assert trade.isDone()
    assert trade.fills[0].time == next_actual_bar
    assert trade.fills[0].execution.price == 102.25
    result = broker.result()
    assert result.commission == 0.5
    assert result.positions == {contract.conId: 1.0}
    assert result.ending_cash == 4_887.0


async def test_stock_fills_update_cash_without_changing_pnl_baseline() -> None:
    """Cash products exchange notional while marked equity remains PnL-based."""

    contract = _stock(434, "CASH")
    broker = SimulatedIB(initial_cash=1_000)
    broker.configure_contract(contract, _metadata(contract))
    first = datetime(2025, 1, 6, 10, tzinfo=timezone.utc)
    second = datetime(2025, 1, 6, 11, tzinfo=timezone.utc)
    broker.set_current_time(first)
    broker.placeOrder(contract, ibi.MarketOrder("BUY", 2))

    await broker.process_bar(
        contract,
        ibi.BarData(date=second, open=100, high=101, low=99, close=100),
    )
    result = broker.result()

    assert result.ending_cash == 800
    assert result.equity == 1_000
    assert result.contracts[0].cash_change == -200


async def test_limit_slippage_never_violates_limit_price() -> None:
    """Adverse tick costs are capped by BUY and SELL limit guarantees."""

    contract = _stock(454, "LIMIT")
    broker = SimulatedIB(slippage_ticks=2)
    broker.configure_contract(contract, _metadata(contract, minTick=0.25))
    first = datetime(2025, 1, 6, 10, tzinfo=timezone.utc)
    second = datetime(2025, 1, 6, 11, tzinfo=timezone.utc)
    third = datetime(2025, 1, 6, 12, tzinfo=timezone.utc)
    broker.set_current_time(first)
    buy = broker.placeOrder(contract, ibi.LimitOrder("BUY", 1, 100))
    await broker.process_bar(
        contract,
        ibi.BarData(date=second, open=100, high=101, low=99, close=100),
    )
    sell = broker.placeOrder(contract, ibi.LimitOrder("SELL", 1, 100))
    await broker.process_bar(
        contract,
        ibi.BarData(date=third, open=100, high=101, low=99, close=100),
    )

    assert buy.fills[0].execution.price == 100
    assert sell.fills[0].execution.price == 100


async def test_market_close_precedes_intrabar_oca_stop() -> None:
    """The bar open occurs before a later protective stop touch."""

    contract = _stock(464, "OCA")
    broker = SimulatedIB()
    broker.configure_contract(contract, _metadata(contract, minTick=0.25))
    first = datetime(2025, 1, 6, 10, tzinfo=timezone.utc)
    second = datetime(2025, 1, 6, 11, tzinfo=timezone.utc)
    third = datetime(2025, 1, 6, 12, tzinfo=timezone.utc)
    broker.set_current_time(first)
    broker.placeOrder(contract, ibi.MarketOrder("BUY", 1))
    await broker.process_bar(
        contract,
        ibi.BarData(date=second, open=100, high=101, low=99, close=100),
    )
    stop = broker.placeOrder(
        contract,
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType="STP",
            auxPrice=99,
            ocaGroup="exit",
            ocaType=1,
            tif="GTC",
        ),
    )
    market = broker.placeOrder(
        contract,
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType="MKT",
            ocaGroup="exit",
            ocaType=1,
            tif="GTC",
        ),
    )

    await broker.process_bar(
        contract,
        ibi.BarData(date=third, open=101, high=102, low=98, close=100),
    )

    assert market.orderStatus.status == ibi.OrderStatus.Filled
    assert market.fills[0].execution.price == 101
    assert stop.orderStatus.status == ibi.OrderStatus.Cancelled
    assert not stop.fills


async def test_market_phase_precedes_older_oca_intrabar_orders() -> None:
    """All market fills precede intrabar OCA stop-versus-limit ambiguity."""

    contract = _stock(469, "PHASE")
    broker = SimulatedIB()
    broker.configure_contract(contract, _metadata(contract, minTick=0.25))
    first = datetime(2025, 1, 6, 10, tzinfo=timezone.utc)
    second = datetime(2025, 1, 6, 11, tzinfo=timezone.utc)
    broker.set_current_time(first)
    limit = broker.placeOrder(
        contract,
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType="LMT",
            lmtPrice=103,
            ocaGroup="protective",
            ocaType=1,
            tif="GTC",
        ),
    )
    stop = broker.placeOrder(
        contract,
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType="STP",
            auxPrice=99,
            ocaGroup="protective",
            ocaType=1,
            tif="GTC",
        ),
    )
    market = broker.placeOrder(contract, ibi.MarketOrder("BUY", 1))

    executed = await broker.process_bar(
        contract,
        ibi.BarData(date=second, open=101, high=104, low=98, close=100),
    )

    assert executed == (market, stop)
    assert [fill.execution.orderId for fill in broker.fills()] == [
        market.order.orderId,
        stop.order.orderId,
    ]
    assert limit.orderStatus.status == ibi.OrderStatus.Cancelled
    assert not limit.fills


@pytest.mark.parametrize(
    "order",
    [
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            algoStrategy="Adaptive",
            algoParams=[ibi.TagValue("adaptivePriority", "Normal")],
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            tif="DAY",
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            whatIf=True,
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            parentId=123,
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            parentPermId=456,
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            autoCancelParent=True,
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            transmit=False,
        ),
        ibi.Order(
            action="BUY",
            totalQuantity=1,
            orderType="MKT",
            discretionaryAmt=1,
        ),
    ],
)
def test_simulated_broker_rejects_unmodeled_order_modifiers(
    order: ibi.Order,
) -> None:
    """Live algorithm and lifetime semantics are never silently discarded."""

    contract = _stock(474, "MOD")
    broker = SimulatedIB()
    broker.configure_contract(contract, _metadata(contract))

    with pytest.raises(UnsupportedBacktestOrderError):
        broker.placeOrder(contract, order)


async def test_backtester_runs_real_bracket_controller_and_book_path() -> None:
    """A real graph opens next-bar and resolves ambiguous OCA stop-first."""

    points = pd.date_range("2025-01-06 10:00", periods=4, freq="h", tz="UTC")
    contract = _stock(505, "TEST")
    store = MemoryBacktestStore(
        {
            "TEST_STK": _frame(
                points,
                open_=[100, 101, 101, 100],
                high=[101, 102, 105, 101],
                low=[99, 100, 98, 99],
                close=[100, 101, 100, 100],
            )
        },
        {
            "TEST_STK": _metadata(
                contract,
                minTick=0.25,
                commission=0.5,
            )
        },
    )
    blueprint = ibi.Stock("TEST", "SMART", "USD")
    previous_runtime = Atom.__dict__.get("runtime")
    previous_streamers = {id(streamer) for streamer in Streamer.instances}
    previous_queues = {id(runner) for runner in QueueRunner._instances}
    previous_task_factory = asyncio.get_running_loop().get_task_factory()

    def strategy_factory() -> None:
        """Construct a normal one-to-one Haymaker bracket graph."""

        source = HistoricalDataStreamer(
            blueprint,
            2,
            "1 hour",
            "TRADES",
            timeout=False,
        )
        Pipe(
            source,
            BarAggregator(NoFilter(), future_adjust_type=None),
            AlwaysLongSignalModel("alpha", blueprint, SignalType.STATE),
            BinarySignalProcessor(respect_blocked_direction=True),
            PortfolioWrapper(FixedSizeAllocator(1)),
            BracketExecutionModel(
                "alpha",
                name="alpha_backtest",
                stop=FixedStop(1),
                take_profit=TakeProfitAsStopMultiple(1, 2),
            ),
        )

    result = await Backtester(store, initial_cash=10_000).run(strategy_factory)

    assert [
        (fill.time, fill.action, fill.order_type, fill.price) for fill in result.fills
    ] == [
        (points[1], "BUY", "MKT", 101.0),
        (points[2], "SELL", "STP", 100.0),
    ]
    assert result.realized_pnl == -1.0
    assert result.commission == 1.0
    assert result.net_pnl == -2.0
    assert result.equity == 9_998.0
    assert result.positions == {}
    assert {id(streamer) for streamer in Streamer.instances} == previous_streamers
    assert {id(runner) for runner in QueueRunner._instances} == previous_queues
    assert Atom.__dict__.get("runtime") is previous_runtime
    assert asyncio.get_running_loop().get_task_factory() is previous_task_factory


async def test_startup_order_uses_first_observed_contract_session() -> None:
    """An order emitted by onStart is eligible on the first stored bar."""

    points = pd.date_range("2025-01-06 10:00", periods=2, freq="h", tz="UTC")
    contract = _stock(506, "START")
    store = MemoryBacktestStore(
        {"START_STK": _frame(points, open_=[100, 101])},
        {"START_STK": _metadata(contract)},
    )
    blueprint = ibi.Stock("START", "SMART", "USD")

    def strategy_factory() -> None:
        """Build a graph whose target is emitted only during startup."""

        Pipe(
            HistoricalDataStreamer(
                blueprint,
                1,
                "1 hour",
                "TRADES",
                timeout=False,
            ),
            StartupTarget(),
            SerialTargetExecutionModel(name="startup_serial"),
        )

    result = await Backtester(store).run(strategy_factory)

    assert len(result.fills) == 1
    assert result.fills[0].time == points[0]
    assert result.fills[0].price == 100


async def test_result_exposes_order_working_at_end_of_range() -> None:
    """A last-bar order is visible even though it has no eligible next bar."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(507, "PENDING")
    store = MemoryBacktestStore(
        {"PENDING_STK": _frame(point)},
        {"PENDING_STK": _metadata(contract)},
    )
    blueprint = ibi.Stock("PENDING", "SMART", "USD")

    def strategy_factory() -> None:
        """Submit a normal market order from the final observation."""

        Pipe(
            HistoricalDataStreamer(
                blueprint,
                1,
                "1 hour",
                "TRADES",
                timeout=False,
            ),
            BarAggregator(NoFilter(), future_adjust_type=None),
            AlwaysLongSignalModel("pending", blueprint, SignalType.STATE),
            DirectLongPortfolio(),
            SerialTargetExecutionModel(name="pending_serial"),
        )

    result = await Backtester(store).run(strategy_factory)

    assert result.fills == ()
    assert len(result.orders) == 1
    assert result.orders[0].status == ibi.OrderStatus.Submitted
    assert result.orders[0].submitted_at == point[0]
    assert result.working_orders == result.orders


async def test_new_order_requires_a_bar_for_its_own_contract() -> None:
    """A cross-Contract target waits for a later observed target session."""

    points = pd.date_range("2025-01-06 10:00", periods=3, freq="h", tz="UTC")
    source_contract = _stock(507, "SOURCE")
    target_contract = _stock(508, "TARGET")
    store = MemoryBacktestStore(
        {
            "SOURCE_STK": _frame(points[:2]),
            "TARGET_STK": _frame(points[1:], open_=[200, 202]),
        },
        {
            "SOURCE_STK": _metadata(source_contract),
            "TARGET_STK": _metadata(target_contract),
        },
    )
    source_blueprint = ibi.Stock("SOURCE", "SMART", "USD")
    target_blueprint = ibi.Stock("TARGET", "SMART", "USD")

    def strategy_factory() -> None:
        """Drive TARGET orders from SOURCE observations."""

        source = HistoricalDataStreamer(
            source_blueprint,
            1,
            "1 hour",
            "TRADES",
            timeout=False,
        )
        HistoricalDataStreamer(
            target_blueprint,
            1,
            "1 hour",
            "TRADES",
            timeout=False,
        )
        Pipe(
            source,
            BarAggregator(NoFilter(), future_adjust_type=None),
            CrossContractTarget(target_blueprint),
            SerialTargetExecutionModel(name="cross_contract_serial"),
        )

    result = await Backtester(store).run(strategy_factory)

    assert len(result.fills) == 1
    assert result.fills[0].contract.conId == target_contract.conId
    assert result.fills[0].time == points[2]
    assert result.fills[0].price == 202


async def test_flat_futures_roll_restarts_graph_on_new_active_contract() -> None:
    """Framework selectors drive a flat ACTIVE roll using metadata only."""

    points = pd.DatetimeIndex(
        [
            datetime(2025, 1, 7, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 8, 10, tzinfo=timezone.utc),
        ]
    )
    expiring = _future(515, "ESF5", "20250108")
    successor = _future(516, "ESG5", "20250208")
    store = MemoryBacktestStore(
        {
            "ESF5_FUT": _frame(points[[0]]),
            "ESG5_FUT": _frame(points),
        },
        {
            "ESF5_FUT": _metadata(expiring, minTick=0.25),
            "ESG5_FUT": _metadata(successor, minTick=0.25),
        },
    )
    blueprint = ibi.ContFuture("ES", "CME", "USD", tradingClass="ES")
    signals: list[Any] = []

    def strategy_factory() -> None:
        """Record Signals while leaving the simulated account flat."""

        source = HistoricalDataStreamer(
            blueprint,
            2,
            "1 hour",
            "TRADES",
            timeout=False,
        )
        model = AlwaysLongSignalModel("flat_roll", blueprint, SignalType.STATE)
        model.dataEvent.connect(signals.append, keep_ref=True)
        Pipe(
            source,
            BarAggregator(NoFilter(), future_adjust_type=None),
            model,
        )

    result = await Backtester(
        store,
        futures_roll_bdays=0,
        futures_roll_margin_bdays=0,
    ).run(strategy_factory)

    assert [signal.contract.conId for signal in signals] == [
        expiring.conId,
        successor.conId,
    ]
    assert result.fills == ()


def _held_future_store() -> tuple[
    MemoryBacktestStore,
    ibi.ContFuture,
    ibi.Future,
]:
    """Return an overlapping chain that rolls ACTIVE on 2025-01-10."""

    points = pd.date_range("2025-01-06 10:00", periods=5, freq="D", tz="UTC")
    expiring = _future(525, "ESF5", "20250110")
    successor = _future(526, "ESG5", "20250210")
    store = MemoryBacktestStore(
        {
            "ESF5_FUT": _frame(points),
            "ESG5_FUT": _frame(points),
        },
        {
            "ESF5_FUT": _metadata(expiring, minTick=0.25),
            "ESG5_FUT": _metadata(successor, minTick=0.25),
        },
    )
    blueprint = ibi.ContFuture("ES", "CME", "USD", tradingClass="ES")
    return store, blueprint, expiring


def _held_future_strategy(blueprint: ibi.ContFuture) -> None:
    """Construct a normal serial strategy that remains long its first future."""

    Pipe(
        HistoricalDataStreamer(
            blueprint,
            1,
            "1 day",
            "TRADES",
            timeout=False,
        ),
        BarAggregator(NoFilter(), future_adjust_type=None),
        AlwaysLongSignalModel("held_future", blueprint, SignalType.STATE),
        DirectLongPortfolio(),
        SerialTargetExecutionModel(name="held_future_serial"),
    )


async def test_next_only_futures_change_keeps_held_active_contract() -> None:
    """A held ACTIVE remains valid when only NEXT advances early."""

    store, blueprint, expiring = _held_future_store()

    result = await Backtester(
        store,
        end=datetime(2025, 1, 8, 10, tzinfo=timezone.utc),
        futures_roll_bdays=0,
        futures_roll_margin_bdays=2,
    ).run(lambda: _held_future_strategy(blueprint))

    assert result.positions == {expiring.conId: 1.0}


async def test_active_futures_roll_rejects_held_expiring_contract() -> None:
    """The documented unsupported open-position roll fails before new bars."""

    store, blueprint, _expiring = _held_future_store()

    with pytest.raises(
        UnsupportedBacktestFeatureError,
        match="Rolling an open futures position",
    ):
        await Backtester(
            store,
            futures_roll_bdays=0,
            futures_roll_margin_bdays=2,
        ).run(lambda: _held_future_strategy(blueprint))


async def test_backtester_rejects_trailing_bracket_before_reading_bars() -> None:
    """Unsupported protection fails explicitly instead of producing fake fills."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(606, "TRAIL")
    store = MemoryBacktestStore(
        {"TRAIL_STK": _frame(point)},
        {"TRAIL_STK": _metadata(contract, minTick=0.25)},
    )
    blueprint = ibi.Stock("TRAIL", "SMART", "USD")

    def strategy_factory() -> None:
        """Construct a graph whose trailing protection is out of scope."""

        source = HistoricalDataStreamer(
            blueprint,
            1,
            "1 hour",
            "TRADES",
            timeout=False,
        )
        Pipe(
            source,
            BarAggregator(NoFilter(), future_adjust_type=None),
            AlwaysLongSignalModel("trail", blueprint, SignalType.STATE),
            BinarySignalProcessor(),
            PortfolioWrapper(FixedSizeAllocator(1)),
            BracketExecutionModel(
                "trail",
                stop=TrailingStop(1),
            ),
        )

    with pytest.raises(
        UnsupportedBacktestFeatureError,
        match="only FixedStop",
    ):
        await Backtester(store).run(strategy_factory)

    assert store.read_keys == []


async def test_backtester_propagates_async_event_callback_failures() -> None:
    """Fast task failures are surfaced instead of logged and discarded."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(707, "FAIL")
    store = MemoryBacktestStore(
        {"FAIL_STK": _frame(point)},
        {"FAIL_STK": _metadata(contract)},
    )
    blueprint = ibi.Stock("FAIL", "SMART", "USD")

    def strategy_factory() -> None:
        """Connect a scheduled failing callback to a historical source."""

        Pipe(
            HistoricalDataStreamer(
                blueprint,
                1,
                "1 hour",
                "TRADES",
                timeout=False,
            ),
            BarAggregator(NoFilter(), future_adjust_type=None),
            FailingAsyncAtom(),
        )

    with pytest.raises(
        BacktestStrategyError,
        match="Asynchronous strategy callback failed",
    ) as exc_info:
        await Backtester(store).run(strategy_factory)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_backtester_captures_fast_broker_callback_failures() -> None:
    """A broker event that fails before process_bar returns still fails the run."""

    points = pd.date_range("2025-01-06 10:00", periods=2, freq="h", tz="UTC")
    contract = _stock(717, "BROKERFAIL")
    store = MemoryBacktestStore(
        {"BROKERFAIL_STK": _frame(points)},
        {"BROKERFAIL_STK": _metadata(contract)},
    )
    blueprint = ibi.Stock("BROKERFAIL", "SMART", "USD")

    def strategy_factory() -> None:
        """Attach a failing callback to the simulated IB execution event."""

        async def fail_on_execution(trade: ibi.Trade, fill: ibi.Fill) -> None:
            """Fail immediately when the broker reports an execution."""

            del trade, fill
            raise RuntimeError("broker callback failure")

        Atom.runtime.ib.execDetailsEvent.connect(
            fail_on_execution,
            keep_ref=True,
        )
        Pipe(
            HistoricalDataStreamer(
                blueprint,
                1,
                "1 hour",
                "TRADES",
                timeout=False,
            ),
            BarAggregator(NoFilter(), future_adjust_type=None),
            AlwaysLongSignalModel("broker_failure", blueprint, SignalType.STATE),
            DirectLongPortfolio(),
            SerialTargetExecutionModel(name="broker_failure_serial"),
        )

    with pytest.raises(
        BacktestStrategyError,
        match="Asynchronous strategy callback failed",
    ) as exc_info:
        await Backtester(store).run(strategy_factory)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "broker callback failure"


async def test_backtester_does_not_await_unrelated_concurrent_tasks() -> None:
    """The temporary loop task factory tracks only the strategy context."""

    point = pd.date_range("2025-01-06 10:00", periods=1, tz="UTC")
    contract = _stock(727, "ISOLATED")
    store = MemoryBacktestStore(
        {"ISOLATED_STK": _frame(point)},
        {"ISOLATED_STK": _metadata(contract)},
    )
    blueprint = ibi.Stock("ISOLATED", "SMART", "USD")
    create_child = asyncio.Event()
    child_created = asyncio.Event()
    release_child = asyncio.Event()

    async def unrelated_owner() -> asyncio.Task[bool]:
        """Create a child after the simulation has installed its task factory."""

        await create_child.wait()
        child = asyncio.create_task(release_child.wait())
        child_created.set()
        return child

    owner = asyncio.create_task(unrelated_owner())

    async def strategy_factory() -> None:
        """Ensure the unrelated task is alive while constructing the graph."""

        create_child.set()
        await child_created.wait()
        HistoricalDataStreamer(
            blueprint,
            1,
            "1 hour",
            "TRADES",
            timeout=False,
        )

    await Backtester(store).run(strategy_factory)
    child = await owner
    assert not child.done()
    release_child.set()
    assert await child
