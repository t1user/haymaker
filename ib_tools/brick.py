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
        self._log = logging.getLogger(f"{self.strategy}.{self.__class__.__name__}")

    def onStart(self, data, *args):
        self.startEvent.emit({"strategy": self.strategy}, self)

    def onData(self, data, *args) -> None:
        self.dataEvent.emit(self._params(**self._signal(data)))

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
        d["signal"] = d[self.signal_column]
        return d

    @singledispatchmethod
    def df_row(self, data) -> pd.Series:
        return self.df(pd.DataFrame(data)).iloc[-1]

    @df_row.register
    def _(self, data: pd.DataFrame) -> pd.Series:
        return self.df(data).iloc[-1]

    @df_row.register(BarList)
    @df_row.register(ibi.BarDataList)
    def _(self, data: Union[BarList, ibi.BarDataList]) -> pd.Series:
        return self.df(ibi.util.df(data)).iloc[-1]

    @abstractmethod
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        ...
