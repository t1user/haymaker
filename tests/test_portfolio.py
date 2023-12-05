import pytest

from ib_tools.base import Atom
from ib_tools.portfolio import AbstractBasePortfolio, FixedPortfolio, PortfolioWrapper


@pytest.fixture
def portfolio_1():
    class Portfolio(AbstractBasePortfolio):
        def allocate(self, data):
            return 1

    yield Portfolio
    AbstractBasePortfolio._instance = None


def test_abstract_portoflio_is_abstract():
    with pytest.raises(TypeError):
        AbstractBasePortfolio()


def test_portfolio_instantiates_and_returns_a_wrapper(portfolio_1):
    p = portfolio_1()
    assert isinstance(p, PortfolioWrapper)


def test_portfolio_is_a_singleton(portfolio_1):
    p1 = portfolio_1()
    p2 = portfolio_1()

    assert p1._portfolio is p2._portfolio


@pytest.fixture
def atom_portfolio():
    class Portfolio(AbstractBasePortfolio, Atom):
        def allocate(self, data):
            return 1

    portfolio = Portfolio
    yield portfolio
    AbstractBasePortfolio._instance = None
    del portfolio


def test_portfolio_is_not_Atom(atom_portfolio):
    with pytest.raises(TypeError):
        atom_portfolio()


@pytest.mark.parametrize("attribute", ["onStart", "onData", "startEven", "dataEvent"])
def test_portfolio_will_not_implement_Atom_attributes(attribute, portfolio_1):
    p = portfolio_1()
    with pytest.raises(AttributeError):
        setattr(p._instance, attribute, "x")


@pytest.fixture
def portfolio():
    class Portfolio(AbstractBasePortfolio):
        def allocate(self, data):
            return self.other_method()

        def other_method(self):
            return id(self)

    portfolio_class = Portfolio
    yield portfolio_class

    # ensure any existing singleton is destroyed
    del portfolio_class
    AbstractBasePortfolio._instance = None


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
    assert target.data["amount"] == target_1.data["amount"]


def test_fixed_portfolio():
    portfolio = FixedPortfolio(1)
    assert portfolio.allocate({"signal": "OPEN"}) == 1
