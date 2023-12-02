import pytest

from ib_tools.base import Atom
from ib_tools.portfolio import AbstractBasePortfolio, FixedPortfolio, wrap_portfolio


def test_abstract_portoflio_is_abstract():
    with pytest.raises(TypeError):
        AbstractBasePortfolio()


def test_portfolio_is_a_singleton():
    class Portfolio(AbstractBasePortfolio):
        def allocate(self):
            return 1

    p1 = Portfolio()
    p2 = Portfolio()

    assert p1 is p2


@pytest.fixture
def portfolio():
    @wrap_portfolio
    class Portfolio(AbstractBasePortfolio):
        def allocate(self, data):
            print(id(self))
            return self.other_method()

        def other_method(self):
            return id(self)

    portfolio_class = Portfolio
    yield portfolio_class

    # ensure any existing singleton is destroyed
    portfolio_class._instance = None
    del portfolio_class


def test_every_PortfolioWrapper_instance_unique(portfolio):
    p1 = portfolio()
    p2 = portfolio()
    p3 = portfolio()
    p4 = portfolio()

    assert p1 is not p2
    assert p2 is not p3
    assert p3 is not p4


# instances are immediately garbage collected and tests fail
# TODO: investigate further if this is an issue anywhere
# def test_every_PortfolioWrapper_instance_unique_1(portfolio):
#     p1 = id(portfolio())
#     p2 = id(portfolio())
#     p3 = id(portfolio())
#     p4 = id(portfolio())

#     assert p1 != p2
#     assert p2 != p3
#     assert p3 != p4


def test_onData_separate_for_each_instance(portfolio):
    class NewAtom(Atom):
        data = None

        def onData(self, data, *args):
            super().onData(data)
            self.data = data
            self.dataEvent.emit(data)

    source = NewAtom()
    target = NewAtom()

    portfolio_ = portfolio()

    source.connect(portfolio_)
    portfolio_.connect(target)

    source.dataEvent.emit({"test_data": "test_data"})
    print(target.data)
    assert target.data["test_data"] == "test_data"


def test_onStart_separate_for_each_instance(portfolio):
    class NewAtom(Atom):
        data = None

        def onStart(self, data, *args):
            super().onStart(data)
            self.data = data
            self.startEvent.emit(data)

    source = NewAtom()
    target = NewAtom()

    portfolio_ = portfolio()

    source.connect(portfolio_)
    portfolio_.connect(target)

    source.startEvent.emit({"test_data": "test_data"})
    assert target.data["test_data"] == "test_data"


def test_allocate_shared_among_instances(portfolio):
    portfolio_1_id = portfolio().allocate("test_data")
    portfolio_2_id = portfolio().allocate("test_data")
    portfolio_3_id = portfolio().allocate("test_data")

    assert portfolio_1_id == portfolio_2_id
    assert portfolio_2_id == portfolio_3_id


def test_allocate_shared_among_instances_1(portfolio):
    class NewAtom(Atom):
        data = None

        def onData(self, data, *args):
            super().onData(data)
            self.data = data
            self.dataEvent.emit(data)

    source = NewAtom()
    target = NewAtom()
    portfolio_ = portfolio()
    source.connect(portfolio_)
    portfolio_.connect(target)

    source_1 = NewAtom()
    target_1 = NewAtom()
    portfolio_1 = portfolio()
    source_1.connect(portfolio_1)
    portfolio_1.connect(target_1)

    source.dataEvent.emit({"test_data": "test_data"})
    source_1.dataEvent.emit({"test_data": "test_data_1"})
    print(target.data)
    print(target_1.data)
    assert target.data["amount"] == target_1.data["amount"]


def test_fixed_portfolio():
    portfolio = FixedPortfolio(1)
    assert portfolio.allocate({"signal": "OPEN"}) == 1
