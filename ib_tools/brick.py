from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Optional, Union

import ib_insync as ibi
import pandas as pd

from ib_tools.base import Atom
from ib_tools.processors import BarList

log = logging.getLogger(__name__)


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
            # log.log(5, d)
            self.dataEvent.emit(d)

    @abstractmethod
    def _signal(self, data) -> dict:
        """
        Must return a dict with any params required by the strategy.
        May also contain logging information.  Must have key
        ``signal`` interpretable by subsequent bricks.
        """
        ...

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
    signal_column: str
    df_columns: Optional[list[str]]

    def _signal(self, data) -> dict:
        d = self.df_row(data).to_dict()
        if self.df_columns:
            d = {k: v for k, v in d.items() if k in self.df_columns}
        d["signal"] = int(d[self.signal_column])
        return d

    def df_row(self, data) -> pd.Series:
        return self._create_df(data).reset_index().iloc[-1]

    @singledispatchmethod
    def _create_df(self, data) -> pd.DataFrame:
        try:
            d = pd.DataFrame(data).set_index("date")
        except KeyError:
            d = pd.DataFrame(data)
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
    def _(self, data: Union[BarList, ibi.BarDataList]) -> pd.DataFrame:
        return self.df(ibi.util.df(data).set_index("date"))

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        ...
