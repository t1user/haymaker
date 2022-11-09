import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, TypeVar, Type, Optional
from research.utils import gap_tracer

import pandas as pd

from datastore import AbstractBaseStore, ArcticStore


@dataclass
class Contract:
    secType: str
    conId: int
    symbol: str
    lastTradeDateOrContractMonth: str
    multiplier: float
    exchange: str
    currency: str
    localSymbol: str
    tradingClass: str
    repr: str
    object: bytes
    name: str
    min_tick: float
    commission: float
    df: pd.DataFrame

    def __post_init__(self):
        self.lastTradeDateOrContractMonth = pd.to_datetime(
            self.lastTradeDateOrContractMonth
        )
        # two years before expiry
        self._start = self.lastTradeDateOrContractMonth - pd.Timedelta(days=729)
        self.object = pickle.loads(self.object)

    def __repr__(self):
        return (
            f"Contract({self.localSymbol}, {self.lastTradeDateOrContractMonth.date()})"
        )

    def __gt__(self, other):
        if not isinstance(other, Contract):
            raise TypeError
        return self.lastTradeDateOrContractMonth > other.lastTradeDateOrContractMonth

    @property
    def start(self):
        return self.data.index[0]

    @property
    def end(self):
        return self.data.index[-1]

    @property
    def data(self):
        return self.df.loc[self._start :]


def read_contracts(store: AbstractBaseStore, symbol: str, ending: str = "CONTFUT"):
    keys = [
        key for key in store.keys() if (key.startswith(symbol) and key.endswith(ending))
    ]

    contracts = []
    for key in keys:
        data = store.read(key)
        if data is None:
            continue
        meta = store.read_metadata(key)
        try:
            contract = Contract(**meta, df=data)
        except TypeError:
            continue
        contracts.append(contract)
    return sorted(contracts)


def concat_futures(contracts: List[Contract]) -> pd.DataFrame:
    dfs = [contract.data for contract in contracts]
    df = pd.concat(dfs)
    return df[~df.index.duplicated(keep="last")].sort_index()


def concat_from_store(
    store: AbstractBaseStore, symbol: str, ending: str = "CONTFUT"
) -> pd.DataFrame:
    return concat_futures(read_contracts(store, symbol, ending))


# metadata keys to be attached to the new SynteticContFuture objects
meta_keys = ["name", "min_tick", "commission", "exchange", "multiplier", "currency"]


T = TypeVar("T", bound="SyntheticContFuture")


@dataclass
class SyntheticContFuture:
    symbol: str
    df: pd.DataFrame = field(repr=False)
    meta: Dict[Any, Any] = field(repr=False)
    name: Optional[str] = field(init=False)
    start_date: pd.Timestamp = field(init=False)
    end_date: pd.Timestamp = field(init=False)
    contracts: List[int] = field(default_factory=list, repr=False)

    @classmethod
    def create(cls: Type[T], store: AbstractBaseStore, symbol: str) -> T:
        contracts = read_contracts(store, symbol)
        df = concat_futures(contracts)
        df["volume"] = df["volume"].astype(int)
        meta = {k: getattr(contracts[-1], k) for k in meta_keys}
        return cls(symbol, df, meta, contracts)

    def __post_init__(self):
        self.start_date = self.df.index[0]
        self.end_date = self.df.index[-1]
        self.meta.update({"start_date": self.start_date, "end_date": self.end_date})
        try:
            self.name = self.meta["name"]
        except KeyError:
            self.name = None

    def gap_tracer(self, *args, **kwargs) -> pd.DataFrame:
        return gap_tracer(self.df, *args, **kwargs)


S = TypeVar("S", bound="SaveData")


@dataclass
class SaveData:
    """
    Put together synthetic continuous futures to be written to data store.

    Methods:
    -------

    'create' : use existing data in the passed datastore to create
    synthetic continuous futures

    `review`, `meta`, `df` : use these methods to check data consistency

     `get` : return SyntheticContFuture; e.g. to run `gap_tracer`

    `save` : save newly created data to datastore; the target datastore can be
    different than the source datastore used in `create`

    """

    symbols: List[str]
    data: List[SyntheticContFuture] = field(default_factory=list, repr=False)
    _dict: Optional[Dict[str, int]] = field(init=False, repr=False, default=None)
    _review: Optional[pd.DataFrame] = field(init=False, repr=False, default=None)

    @classmethod
    def create(cls: Type[S], store: AbstractBaseStore, symbols: List[str]) -> S:
        data = [SyntheticContFuture.create(store, s) for s in symbols]
        return cls(symbols, data)

    @property
    def dict(self):
        self._dict = self._dict or {c.symbol: i for i, c in enumerate(self.data)}
        return self._dict

    def get(self, symbol: str) -> SyntheticContFuture:
        try:
            return self.data[self.dict[symbol]]
        except KeyError:
            raise KeyError(
                f"Unknown symbol: {symbol}, available symbols: {self.symbols}"
            )

    def meta(self, symbol: str) -> Dict[Any, Any]:
        return self.get(symbol).meta

    def df(self, symbol: str) -> pd.DataFrame:
        return self.get(symbol).df

    def _make_review(self):
        return pd.DataFrame.from_dict(
            {c.symbol: [c.name, c.start_date, c.end_date] for c in self.data},
            columns=["name", "start", "end"],
            orient="index",
        )

    def review(self):
        # this is to prevent re-generation of data on recurring calls
        self._review = self._review if self._review is not None else self._make_review()
        return self._review

    def __len__(self):
        return len(self.data)

    def save(self, store: AbstractBaseStore) -> None:
        for contract in self.data:
            store.write(contract.symbol, contract.df, contract.meta)


class ContData(ArcticStore):
    """
    Thin wrapper to extract synthetic continues contracts.

    Can be passed to `SaveData.save`.

    Usage:
    -----

    >>>store = ContData()
    >>>store.read("NQ")
    >>>store.read_metadata("NQ")

    Here, create SaveData object and use ContData to save.

    >>>save_data = SaveData.create(ArcticStore("TRADES_30_secs", ["NQ", "ES", "YM"])
    >>>save_data.review()
    >>>save_data.save(store)

    """

    def __init__(self):
        super().__init__("Synthetic_TRADES_30_secs")

    def review(self):
        return super().review("commission")[
            ["name", "from", "to", "min_tick", "commission"]
        ]


class ContCont:
    """
    Return df with data for single synthetic continuous contract.

    Usage:
    -----

    >>>reader = ContCont()
    >>>df = reader("NQ")

    """

    store = ContData()

    def __call__(self, symbol):
        return self.store.read(symbol)
