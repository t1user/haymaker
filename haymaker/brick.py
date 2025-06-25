from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any

import ib_insync as ibi
import pandas as pd

from .aggregators import BarList
from .base import ActiveNext, Atom

log = logging.getLogger(__name__)


@dataclass
class AbstractBaseBrick(Atom, ABC):
    strategy: str
    contract: ibi.Contract
    which_contract: ActiveNext = ActiveNext.NEXT

    def __post_init__(self):
        Atom.__init__(self, which_contract=self.which_contract)

    def onStart(self, data, *args):
        if isinstance(data, dict):
            data["strategy"] = self.strategy
            log.log(5, f"Updated dict on start: {data}")
        self.startup = data.get("startup", False)
        super().onStart(data, *args)

    def onData(self, data, *args) -> None:
        if not self.startup:
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
    which_contract: ActiveNext = ActiveNext.NEXT

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
            d = pd.DataFrame()
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
        df_ = ibi.util.df(data)
        if df_ is not None:
            return self.df(df_.set_index("date"))
        else:
            return pd.DataFrame()

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame: ...
