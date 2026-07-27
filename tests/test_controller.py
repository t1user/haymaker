import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import ib_insync as ibi
import pytest

from haymaker.book import PositionState
from haymaker.components import PositionTarget, StandardOrderRole
from haymaker.controller import Controller
from haymaker.controller.controller import SyncOutcome
from haymaker.controller.sync_coordinator import verify_broker_position_source


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
