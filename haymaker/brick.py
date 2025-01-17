from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any

import ib_insync as ibi
import pandas as pd

from .base import Atom
from .processors import BarList

log = logging.getLogger(__name__)


def register_key(key: str):
    """
    Decorator to be used on :meth:`onData` to wrap the data in a
    dictionary with given `key`.  Used to signal to the next atom down
    the chain that this data needs specific interpreation.
    """

    def add_key(func):
        def wrapper(func, data, *args):
            return {key: func(data, *args)}

        return wrapper

    return add_key


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
        startup = self.__dict__.get("startup")
        if not startup:
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

    def _signal(self, data) -> dict:
        return self.df_row(data).to_dict()

    def df_row(self, data) -> pd.Series:
        return self._create_df(data).reset_index().iloc[-1]

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
        return self.df(d)

    @_create_df.register(pd.DataFrame)
    def _(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.df(data).set_index("date")
        except KeyError:
            return self.df(data)

    # this is most likely redundand as dataframe constructor can handle these types
    @_create_df.register(BarList)
    @_create_df.register(ibi.BarDataList)
    def _(self, data: BarList | ibi.BarDataList) -> pd.DataFrame:
        return self.df(ibi.util.df(data).set_index("date"))

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame: ...
