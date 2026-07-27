import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import ib_insync as ibi
import pytest

from haymaker.book import PositionState
from haymaker.components import PositionTarget, StandardOrderRole
from haymaker.controller import Controller
from haymaker.controller.controller import SyncOutcome
from haymaker.controller.sync_brackets import BracketSync
from haymaker.controller.sync_coordinator import verify_broker_position_source
from haymaker.controller.terminator import Terminator


class FakeTrader:
    def __init__(self):
        self.trades = []
        self.cancelled = []
        self.broker_positions = {}

    def trade(self, contract, order):
        if not order.orderId:
            order.orderId = len(self.trades) + 1
        order.permId = order.permId or 100 + abs(order.orderId)
        trade = ibi.Trade(
            contract=contract,
            order=order,
            orderStatus=ibi.OrderStatus(
                orderId=order.orderId,
                status=ibi.OrderStatus.Submitted,
                remaining=order.totalQuantity,
            ),
        )
        self.trades.append(trade)
        return trade

    def cancel(self, trade):
        self.cancelled.append(trade)
        return trade

    def position_for_contract(self, contract):
        return self.broker_positions.get(contract, 0)

    def positions(self):
        return dict(self.broker_positions)


def contract():
    return ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )


@pytest.fixture
def controller_runtime(atom_runtime):
    trader = FakeTrader()
    controller = Controller(trader=trader, missing_brackets="ignore")
    atom_runtime.bind_controller(controller)
    return atom_runtime, controller, trader


def fill(trade, exec_id="exec-1", quantity=1):
    return ibi.Fill(
        contract=trade.contract,
        execution=ibi.Execution(
            execId=exec_id,
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            side="BOT" if trade.order.action == "BUY" else "SLD",
            shares=quantity,
            price=100,
            time=datetime.now(timezone.utc),
        ),
        commissionReport=ibi.CommissionReport(execId=exec_id),
        time=datetime.now(timezone.utc),
    )


def test_from_mapping_constructs_nested_startup_config(atom_runtime):
    controller = Controller.from_mapping(
        {
            "startup": {"cold_start": False, "zero": True},
            "sync_frequency": 10,
        },
        trader=FakeTrader(),
    )

    assert controller.cold_start is False
    assert controller.zero is True
    assert controller.sync_frequency == 10


def test_trade_registers_complete_attribution_immediately(controller_runtime):
    runtime, controller, trader = controller_runtime

    result = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 2),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
        params={"atr": 5},
    )

    assert result is trader.trades[0]
    info = runtime.book.order_by_id(result.order.orderId)
    assert info.execution_model_name == "brackets"
    assert info.source_key == "alpha"
    assert info.position_id == "episode"
    assert info.role == StandardOrderRole.OPEN


def test_disabled_trading_does_not_submit_or_register(controller_runtime):
    runtime, controller, trader = controller_runtime
    controller.disable_trading("test")

    result = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
    )

    assert result is None
    assert trader.trades == []
    assert runtime.book.active_orders() == ()


@pytest.mark.asyncio
async def test_exec_details_applies_fill_once(controller_runtime):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    execution = fill(trade)

    await controller.onExecDetailsEvent(trade, execution)
    await controller.onExecDetailsEvent(trade, execution)

    assert runtime.book.position_state("alpha").quantity == 1
    assert len(runtime.book.order_by_id(trade.order.orderId).fills) == 1


@pytest.mark.asyncio
async def test_offline_zero_order_id_fill_rebinds_by_perm_id(controller_runtime):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    original = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
    )
    rebound = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=0,
            permId=original.order.permId,
            action="BUY",
            totalQuantity=1,
        ),
    )

    await controller.onExecDetailsEvent(rebound, fill(rebound))

    assert rebound.order.orderId == original.order.orderId
    assert runtime.book.position_state("alpha").quantity == 1


@pytest.mark.asyncio
async def test_manual_trade_has_explicit_role_and_source_attribution(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=-1,
            permId=999,
            action="SELL",
            totalQuantity=1,
        ),
    )

    await controller.onExecDetailsEvent(trade, fill(trade))

    info = runtime.book.order_by_id(-1)
    assert info.role == StandardOrderRole.MANUAL
    assert info.source_key == "alpha"
    assert info.position_id == "episode"


@pytest.mark.asyncio
async def test_commission_report_uses_book_owned_blotter(controller_runtime):
    runtime, controller, _ = controller_runtime

    class Blotter:
        def __init__(self):
            self.calls = []

        def log_commission(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    blotter = Blotter()
    runtime.book.blotter = blotter
    controller.release_hold()
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    report = ibi.CommissionReport(execId="exec-1", commission=1)
    execution = fill(trade)

    await controller.onCommissionReport(trade, execution, report)

    assert blotter.calls[0][1]["source_key"] == "alpha"
    assert blotter.calls[0][1]["position_id"] == "episode"
    assert blotter.calls[0][1]["role"] == StandardOrderRole.OPEN


@pytest.mark.asyncio
async def test_commission_report_persists_without_blotter(controller_runtime):
    """Commission evidence remains durable when reporting is disabled."""

    runtime, controller, _ = controller_runtime
    assert runtime.book.blotter is None
    assert len(controller.ib.commissionReportEvent) == 1
    controller.release_hold()
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    execution = fill(trade)
    await controller.onExecDetailsEvent(trade, execution)
    report = ibi.CommissionReport(execId="exec-1", commission=1.25)

    await controller.onCommissionReport(trade, execution, report)

    info = runtime.book.order_by_id(trade.order.orderId)
    assert info.fills[0].commission_report == report


@pytest.mark.parametrize(
    ("role", "missing"),
    [
        (StandardOrderRole.STOP_LOSS, False),
        (StandardOrderRole.TAKE_PROFIT, True),
    ],
)
def test_bracket_sync_requires_stop_but_not_take_profit(
    controller_runtime, role, missing
):
    """Only an absent stop is a critical local bracket-record issue."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    controller.trade(
        contract(),
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType=(
                "STP" if role == StandardOrderRole.STOP_LOSS else "LMT"
            ),
        ),
        role=role,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )

    sync = BracketSync(controller)

    assert bool(sync.missing_brackets) is missing


@pytest.mark.asyncio
async def test_terminator_waits_for_cancellation_before_logical_close(
    controller_runtime,
):
    """Reset never overlaps an attributed close with a working exit order."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    protective = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=77,
            action="SELL",
            totalQuantity=1,
            orderType="STP",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.openTrades = Mock(return_value=[protective])
    controller.ib.positions = Mock(return_value=[])
    loop = asyncio.get_running_loop()

    def cancel(_trade):
        """Complete cancellation after more than one event-loop turn."""

        loop.call_soon(
            lambda: loop.call_soon(
                setattr,
                protective.orderStatus,
                "status",
                ibi.OrderStatus.Cancelled,
            )
        )
        return protective

    def close(*args, **kwargs):
        """Record that the close was submitted only after cancellation."""

        assert protective.isDone()
        return ibi.Trade(
            contract=args[0],
            order=ibi.Order(
                orderId=78,
                action="SELL",
                totalQuantity=1,
                orderType="MKT",
            ),
            orderStatus=ibi.OrderStatus(
                orderId=78,
                status=ibi.OrderStatus.Filled,
                filled=1,
                remaining=0,
            ),
        )

    controller.cancel = Mock(side_effect=cancel)
    controller.trade = Mock(side_effect=close)

    completed = await Terminator(controller).run()

    assert completed is True
    controller.trade.assert_called_once()


@pytest.mark.asyncio
async def test_terminator_stops_when_cancellation_does_not_complete(
    controller_runtime,
):
    """Reset fails closed instead of overlapping a close with a live order."""

    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    protective = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=77,
            action="SELL",
            totalQuantity=1,
            orderType="STP",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.openTrades = Mock(return_value=[protective])
    terminator = Terminator(controller)
    terminator.cancellation_timeout = 0

    completed = await terminator.run()

    assert completed is False
    assert trader.trades == []


@pytest.mark.asyncio
async def test_terminator_does_not_hide_residual_behind_flat_state(
    controller_runtime,
):
    """A stale flat PositionState does not suppress broker liquidation."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=0,
        )
    )
    controller.ib.openTrades = Mock(return_value=[])
    controller.ib.positions = Mock(
        return_value=[
            ibi.Position(
                account="DU123",
                contract=contract(),
                position=2,
                avgCost=100,
            )
        ]
    )
    done_trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=79,
            action="SELL",
            totalQuantity=2,
            orderType="MKT",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=79,
            status=ibi.OrderStatus.Filled,
            filled=2,
            remaining=0,
        ),
    )
    controller.trade = Mock(return_value=done_trade)

    completed = await Terminator(controller).run()

    assert completed is True
    assert controller.trade.call_args.kwargs["role"] == (StandardOrderRole.LIQUIDATION)


@pytest.mark.asyncio
async def test_run_keeps_book_when_explicit_reset_fails(controller_runtime):
    """Failed reset disables trading without discarding recovery state."""

    runtime, controller, _ = controller_runtime
    controller.cold_start = True
    controller.reset = True
    controller.sync = AsyncMock(return_value=SyncOutcome.OK)
    controller.execute_stops_and_close_positions = AsyncMock(
        return_value=False
    )
    runtime.book.clear_state = Mock()

    completed = await controller.run()

    assert completed is False
    assert controller._trading_disabled is True
    assert controller.reset is True
    runtime.book.clear_state.assert_not_called()


@pytest.mark.asyncio
async def test_target_verification_uses_absolute_quantity(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
        )
    )
    target = PositionTarget(
        contract=contract(),
        target_quantity=3,
        source_key="alpha",
    )

    await controller.verify_target_integrity(target, "brackets")

    assert "target=3.0 actual=2" in caplog.text


@pytest.mark.asyncio
async def test_nuke_liquidation_is_registered_with_episode_attribution(
    controller_runtime, monkeypatch
):
    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            position_id="episode",
        )
    )
    position = ibi.Position("account", contract(), 2, 100)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [position])
    monkeypatch.setattr(
        runtime.ib,
        "qualifyContractsAsync",
        AsyncMock(return_value=[position.contract]),
    )

    await controller.close_positions()

    trade = trader.trades[0]
    info = runtime.book.order_by_id(trade.order.orderId)
    assert trade.order.action == "SELL"
    assert info.role == StandardOrderRole.LIQUIDATION
    assert info.execution_model_name == "brackets"
    assert info.source_key == "alpha"
    assert info.position_id == "episode"


@pytest.mark.asyncio
async def test_broker_position_request_timeout_is_unavailable(monkeypatch):
    ib = ibi.IB()
    monkeypatch.setattr(ib, "positions", lambda: [])
    monkeypatch.setattr(
        ib,
        "reqPositionsAsync",
        AsyncMock(side_effect=asyncio.TimeoutError),
    )

    assert not await verify_broker_position_source(ib, 0.01)


@pytest.mark.asyncio
async def test_clean_sync_releases_hold(controller_runtime, monkeypatch):
    runtime, controller, _ = controller_runtime
    monkeypatch.setattr(runtime.ib, "isConnected", lambda: True)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [])
    monkeypatch.setattr(runtime.ib, "openTrades", lambda: [])
    monkeypatch.setattr(runtime.ib, "trades", lambda: [])
    monkeypatch.setattr(runtime.ib, "fills", lambda: [])
    monkeypatch.setattr(
        runtime.ib, "reqPositionsAsync", AsyncMock(return_value=[])
    )

    outcome = await controller.sync()

    assert outcome is SyncOutcome.OK
    assert controller._hold is False


@pytest.mark.asyncio
async def test_run_disables_trading_when_book_restore_fails(
    controller_runtime, monkeypatch
):
    runtime, controller, _ = controller_runtime
    controller.cold_start = False
    monkeypatch.setattr(
        runtime.book,
        "read_from_store",
        AsyncMock(side_effect=RuntimeError("store failed")),
    )
    sync = AsyncMock()
    monkeypatch.setattr(controller, "sync", sync)

    assert not await controller.run()
    assert controller._trading_disabled
    sync.assert_not_awaited()


def test_rejections_are_scoped_by_execution_model(controller_runtime):
    runtime, controller, _ = controller_runtime
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
    )

    controller.onErrEvent(
        trade.order.orderId,
        201,
        "rejected",
        contract(),
    )

    assert runtime.book._rejected_orders["brackets"] == 1
