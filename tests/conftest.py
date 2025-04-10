import datetime
import logging
from typing import Any

import ib_insync as ibi
import pytest

from haymaker.base import Atom as BaseAtom
from haymaker.controller import Controller
from haymaker.saver import AbstractBaseSaver, SyncSaveManager
from haymaker.state_machine import StateMachine
from haymaker.trader import Trader

log = logging.getLogger(__name__)


class FakeMongoSaver(AbstractBaseSaver):
    """
    It's a mock saver for testing that doesn't save to external media.

    :attr:`.store` has all the data that would have been saved.
    """

    store: dict[str, dict | list[dict]] = {}

    def __init__(
        self, collection: str, query_key: str | None = None, timestamp: bool = False
    ) -> None:
        host = "fakehostname"
        port = 9999
        db = "test"
        self.client = f"MongoClient({host}, {port})"
        self.db = f"{self.client}[{db}]"
        self.collection = collection
        self.query_key = query_key

        if self.query_key:
            self.store[self.collection] = {}
        else:
            self.store[self.collection] = []

        super().__init__("", timestamp)

    def save(self, data: dict[str, Any], *args: str) -> None:
        try:
            if self.query_key:
                key = data.get(self.query_key)
                self.store[self.collection][key] = data  # type: ignore
            elif not all(data.keys()):
                log.error(f"Attempt to save with wrong keys: {list(data.keys())}")
            else:
                self.store[self.collection].append(data)  # type: ignore
        except Exception:
            log.exception(Exception)
            log.debug(f"Data that caused error: {data}")
            raise

    def save_many(self, data: list[dict[str, Any]]):
        for d in data:
            if self.query_key:
                self.store[self.collection][d.get(self.query_key)]  # type: ignore
            else:
                self.store[self.collection].append(d)  # type: ignore

    def read(self, key: dict | None = None) -> list:
        s = self.store[self.collection]
        if key:
            try:
                return [s.get(key)]  # type: ignore
            except AttributeError:
                return s  # type: ignore
        else:
            return s  # type: ignore

    def read_latest(self):
        s = self.store[self.collection]
        try:
            return s[s.keys()[-1]]  # type: ignore
        except AttributeError:
            return s[-1]


@pytest.fixture
def order_saver():
    return FakeMongoSaver("orders", query_key="orderId")


@pytest.fixture
def strategy_saver():
    return FakeMongoSaver("models")


@pytest.fixture
def state_machine(order_saver, strategy_saver):
    sm = StateMachine(
        order_saver=order_saver, strategy_saver=strategy_saver, save_async=False
    )
    yield sm
    # ensure any existing singleton is destroyed
    StateMachine._instance = None


@pytest.fixture
def contract():
    # it's a different contract than in details below
    return ibi.Contract(
        secType="FUT",
        conId=620730920,
        symbol="NQ",
        lastTradeDateOrContractMonth="20240621",
        multiplier="20",
        exchange="CME",
        currency="USD",
        localSymbol="NQM4",
        tradingClass="NQ",
    )


@pytest.fixture
def details():
    # Note: contains also ibi.Contract
    return ibi.ContractDetails(
        contract=ibi.Contract(
            secType="CONTFUT",
            conId=603558814,
            symbol="NQ",
            lastTradeDateOrContractMonth="20240315",
            multiplier="20",
            exchange="CME",
            currency="USD",
            localSymbol="NQH4",
            tradingClass="NQ",
        ),
        marketName="NQ",
        minTick=0.25,
        orderTypes="ACTIVETIM,AD,ADJUST,ALERT,ALGO,ALLOC,AVGCOST,BASKET,BENCHPX,COND,CONDORDER,DAY,DEACT,DEACTDIS,DEACTEOD,GAT,GTC,GTD,GTT,HID,ICE,IOC,LIT,LMT,LTH,MIT,MKT,MTL,NGCOMB,NONALGO,OCA,PEGBENCH,SCALE,SCALERST,SNAPMID,SNAPMKT,SNAPREL,STP,STPLMT,TRAIL,TRAILLIT,TRAILLMT,TRAILMIT,WHATIF",
        validExchanges="CME,QBALGO",
        priceMagnifier=1,
        underConId=11004958,
        longName="E-mini NASDAQ 100 ",
        contractMonth="202403",
        industry="",
        category="",
        subcategory="",
        timeZoneId="US/Central",
        tradingHours="20240303:1700-20240304:1600;20240304:1700-20240305:1600;20240305:1700-20240306:1600;20240306:1700-20240307:1600;20240307:1700-20240308:1600",
        liquidHours="20240304:0830-20240304:1600;20240305:0830-20240305:1600;20240306:0830-20240306:1600;20240307:0830-20240307:1600;20240308:0830-20240308:1600",
        evRule="",
        evMultiplier=0,
        mdSizeMultiplier=1,
        aggGroup=2147483647,
        underSymbol="NQ",
        underSecType="IND",
        marketRuleIds="67,67",
        secIdList=[],
        realExpirationDate="20240315",
        lastTradeTime="08:30:00",
        stockType="",
        minSize=1.0,
        sizeIncrement=1.0,
        suggestedSizeIncrement=1.0,
        cusip="",
        ratings="",
        descAppend="",
        bondType="",
        couponType="",
        callable=False,
        putable=False,
        coupon=0,
        convertible=False,
        maturity="",
        issueDate="",
        nextOptionDate="",
        nextOptionType="",
        nextOptionPartial=False,
        notes="",
    )


@pytest.fixture
def Atom(state_machine, details):
    sm = state_machine
    BaseAtom.set_init_data(ibi.IB(), sm)
    BaseAtom.contract_details[details.contract] = details
    return BaseAtom


@pytest.fixture
def controller(Atom):
    return Controller(Trader(Atom.ib))


@pytest.fixture
def trade():
    """
    This is an example trade as actually produced by ib_insync, it can
    be passed to objects that require a trade.
    """
    return ibi.Trade(
        contract=ibi.Future(
            conId=620731015,
            symbol="ES",
            lastTradeDateOrContractMonth="20250620",
            multiplier="50",
            exchange="CME",
            currency="USD",
            localSymbol="ESM5",
            tradingClass="ES",
        ),
        order=ibi.Order(
            orderId=55308,
            permId=1761343997,
            action="BUY",
            totalQuantity=1.0,
            orderType="MKT",
            lmtPrice=5740.75,
            auxPrice=0.0,
            tif="Day",
            algoStrategy="Adaptive",
            algoParams=[ibi.TagValue(tag="adaptivePriority", value="Normal")],
        ),
        orderStatus=ibi.OrderStatus(
            orderId=55308,
            status="Filled",
            filled=1.0,
            remaining=0.0,
            avgFillPrice=5740.5,
            permId=1761343997,
            parentId=0,
            lastFillPrice=5740.5,
            clientId=0,
            whyHeld="",
            mktCapPrice=0.0,
        ),
        fills=[
            ibi.Fill(
                contract=ibi.Future(
                    conId=620731015,
                    symbol="ES",
                    lastTradeDateOrContractMonth="20250620",
                    multiplier="50",
                    exchange="CME",
                    currency="USD",
                    localSymbol="ESM5",
                    tradingClass="ES",
                ),
                execution=ibi.Execution(
                    execId="0000e1a7.67dbf24e.01.01",
                    time=datetime.datetime(
                        2025, 3, 20, 16, 6, 5, tzinfo=datetime.timezone.utc
                    ),
                    acctNumber="DU3598515",
                    exchange="CME",
                    side="BOT",
                    shares=1.0,
                    price=5740.5,
                    permId=1761343997,
                    clientId=0,
                    orderId=55308,
                    liquidation=0,
                    cumQty=1.0,
                    avgPrice=5740.5,
                    orderRef="",
                    evRule="",
                    evMultiplier=0.0,
                    modelCode="",
                    lastLiquidity=1,
                ),
                commissionReport=ibi.CommissionReport(
                    execId="0000e1a7.67dbf24e.01.01",
                    commission=2.25,
                    currency="USD",
                    realizedPNL=0.0,
                    yield_=0.0,
                    yieldRedemptionDate=0,
                ),
                time=datetime.datetime(
                    2025, 3, 20, 16, 6, 6, 104085, tzinfo=datetime.timezone.utc
                ),
            )
        ],
        log=[
            ibi.TradeLogEntry(
                time=datetime.datetime(
                    2025, 3, 20, 16, 6, 5, 387455, tzinfo=datetime.timezone.utc
                ),
                status="PendingSubmit",
                message="",
                errorCode=0,
            ),
            ibi.TradeLogEntry(
                time=datetime.datetime(
                    2025, 3, 20, 16, 6, 6, 104085, tzinfo=datetime.timezone.utc
                ),
                status="Submitted",
                message="",
                errorCode=0,
            ),
            ibi.TradeLogEntry(
                time=datetime.datetime(
                    2025, 3, 20, 16, 6, 6, 104085, tzinfo=datetime.timezone.utc
                ),
                status="Submitted",
                message="Fill 1.0@5740.5",
                errorCode=0,
            ),
            ibi.TradeLogEntry(
                time=datetime.datetime(
                    2025, 3, 20, 16, 6, 6, 104085, tzinfo=datetime.timezone.utc
                ),
                status="Filled",
                message="",
                errorCode=0,
            ),
        ],
        advancedError="",
    )
