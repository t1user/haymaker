from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import ib_insync as ibi
import pandas as pd

from ib_tools.base import Atom


@dataclass
class AbstractBaseBrick(Atom, ABC):
    key: str
    contract: ibi.Contract

    def __post_init__(self):
        Atom.__init__(self)

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
            "key": self.key,
            "contract": self.contract,
        }
        params.update(**kwargs)
        return params


@dataclass
class AbstractDfBrick(AbstractBaseBrick):
    signal_column: str = "signal"
    df_columns: Optional[list[str]] = None

    def _signal(self, data) -> dict:
        d = self.df_row(data).to_dict()
        if self.df_columns:
            d = {k: v for k, v in d.items() if k in self.df_columns}
        d["signal"] = d[self.signal_column]
        return d

    def df_row(self, data) -> pd.Series:
        return self.df(data).iloc[-1]

    @abstractmethod
    def df(self, data) -> pd.DataFrame:
        ...
