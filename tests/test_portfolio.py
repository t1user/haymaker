import pytest

from haymaker.base import Atom
from haymaker.portfolio import AbstractBasePortfolio, FixedPortfolio, PortfolioWrapper

# it's important to destroy portfolio after every test, because instance is saved
# on the class


@pytest.fixture
def portfolio_1():
    class Portfolio(AbstractBasePortfolio):
        def allocate(self, data):
            return 1

    yield Portfolio
    AbstractBasePortfolio.instance = None


def test_abstract_portoflio_is_abstract():
    with pytest.raises(TypeError):
        AbstractBasePortfolio()


def test_portfolio_is_a_singleton(portfolio_1):
    p1 = portfolio_1()
    p2 = portfolio_1()

    assert p1 is p2


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
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def allocate(self, data):
            return self.other_method()

        def other_method(self):
            return id(self)

    portfolio_class = Portfolio
    yield portfolio_class

    # ensure any existing singleton is destroyed
    del portfolio_class
    AbstractBasePortfolio.instance = None


def test_wrapper_will_not_instantiate_without_portoflio_created_first():
    with pytest.raises(TypeError):
        PortfolioWrapper()


def test_wrapper_will_instantiate_with_portfolio_created_first(portfolio):
    portfolio(1, 2)
    assert isinstance(PortfolioWrapper(), PortfolioWrapper)


def test_init_called_while_portfolio_created(portfolio):
    p = portfolio(1, 2)
    assert p.x == 1
    assert p.y == 2


def test_every_PortfolioWrapper_instance_unique(portfolio):
    portfolio(1, 2)
    p1 = PortfolioWrapper()
    p2 = PortfolioWrapper()

    assert p1 is not p2


def test_every_PortfolioWrapper_instance_refers_to_the_same_portfolio(portfolio):
    portfolio(1, 2)
    p1 = PortfolioWrapper()
    p2 = PortfolioWrapper()

    assert p1._portfolio is p2._portfolio


def test_allocate_shared_among_wrapper_instances(portfolio):
    portfolio(1, 2)
    portfolio_1_id = PortfolioWrapper().allocate("test_data")
    portfolio_2_id = PortfolioWrapper().allocate("test_data")
    portfolio_3_id = PortfolioWrapper().allocate("test_data")

    assert portfolio_1_id == portfolio_2_id
    assert portfolio_2_id == portfolio_3_id


def test_fixed_portfolio():
    portfolio = FixedPortfolio(1)
    assert portfolio.allocate({"signal": "OPEN"}) == 1


def test_wrapper_with_fixed_portfolio():
    FixedPortfolio(1)
    assert PortfolioWrapper().allocate({"signal": "OPEN"}) == 1
