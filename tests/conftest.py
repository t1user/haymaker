import ib_insync as ibi
import pytest

from ib_tools.base import Atom as BaseAtom
from ib_tools.controller import Controller as C
from ib_tools.saver import FakeMongoSaver, SyncSaveManager
from ib_tools.state_machine import StateMachine

order_saver = FakeMongoSaver("orders", query_key="orderId")
model_saver = FakeMongoSaver("models")


@pytest.fixture
def state_machine():
    # ensure any existing singleton is destroyed
    # mere module imports will create an instance
    # so using yield and subsequent tear-down
    # will not work
    if StateMachine._instance:
        StateMachine._instance = None
    sm = StateMachine()
    sm._save_order = SyncSaveManager(order_saver)
    sm._save_model = SyncSaveManager(model_saver)
    return sm


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
def Controller():
    class TestController(C):
        config = {}

    return TestController
