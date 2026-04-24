from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, partial, singledispatchmethod
from typing import Any, ClassVar

import ib_insync as ibi
import pandas as pd

from .base import Atom
from .config import CONFIG
from .databases import get_mongo_client
from .datastore import (
    AsyncAbstractBaseStore,
    AsyncArcticStore,
    CollectionNamerStrategySymbol,
)

log = logging.getLogger(__name__)

DATA_LIB = CONFIG.get("brick_data_library", None)


@dataclass
class AbstractBaseBrick(Atom, ABC):
    strategy: str
    contract: ibi.Contract

    def __post_init__(self):
        Atom.__init__(self)

    def onStart(self, data, *args):
        if isinstance(data, dict):
            data["strategy"] = self.strategy
            log.log(5, f"Updated dict on start: {data}")
        super().onStart(data, *args)

    def onData(self, data, *args) -> None:
        d = self._params(**self._signal(data))
        super().onData(d)  # timestamp on departure
        self._log.debug(d)
        self.dataEvent.emit(d)

    @abstractmethod
    def _signal(self, data) -> dict[str, Any]:
        """
        Must return a dict with any params required by the strategy.
        May also contain logging information.  Must have all keys
        required by atoms down the chain.
        """
        return {}

    def _params(self, **kwargs) -> dict[str, Any]:
        params = {
            "strategy": self.strategy,
            "contract": self.contract,
        }
        params.update(**kwargs)
        return params


@dataclass
class AbstractDfBrick(AbstractBaseBrick):
    strategy: str
    contract: ibi.Contract

    datastore: ClassVar[AsyncAbstractBaseStore | None] = None

    @classmethod
    def set_datastore(cls, datastore: AsyncAbstractBaseStore) -> type[AbstractDfBrick]:
        AbstractDfBrick.datastore = datastore
        return cls

    @cached_property
    def _datastore(self) -> AsyncAbstractBaseStore:
        datastore = self.datastore or AsyncArcticStore(
            DATA_LIB, host=get_mongo_client()
        )
        datastore.override_collection_namer(
            CollectionNamerStrategySymbol(self.strategy)
        )
        return datastore

    @cached_property
    def store(self) -> None | AsyncAbstractBaseStore:
        if DATA_LIB:
            return self._datastore
        else:
            return None

    def _signal(self, data) -> dict:
        return self.df_row(data).to_dict()

    def df_row(self, data) -> pd.Series:
        df = self._create_df(data)
        if self.store:
            self.store.append(self.contract, df)
        return df.reset_index().iloc[-1]

    @singledispatchmethod
    def _create_df(self, data) -> pd.DataFrame:
        try:
            d = pd.DataFrame(data).set_index("date")
        except KeyError:
            d = pd.DataFrame(data)
        except ValueError:
            log.error(
                f"{self} received data in wrong format: {data}, type: {type(data)}"
            )
            d = pd.DataFrame()
        return self.df(d)

    @_create_df.register(pd.DataFrame)
    def _(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.df(data).set_index("date")
        except KeyError:
            return self.df(data)

    # this is most likely redundand as dataframe constructor can handle these types
    @_create_df.register(ibi.BarDataList)
    def _(self, data: ibi.BarDataList) -> pd.DataFrame:
        df_ = ibi.util.df(data)
        if df_ is not None:
            return self.df(df_.set_index("date"))
        else:
            return pd.DataFrame()

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame: ...
