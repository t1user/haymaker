from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import ib_insync as ibi
import pandas as pd

from ib_tools import misc
from ib_tools.base import Atom
from ib_tools.execution_models import BaseExecModel
from ib_tools.signals import SignalProcessor


@dataclass
class AbstractBaseBrick(Atom, ABC):
    key: tuple[str, str]
    contract: ibi.Contract
    exec_model: BaseExecModel

    def __post_init__(self):
        Atom.__init__(self)

    def onData(self, data, *args) -> None:
        signal, kwargs = self._signal(data)
        if signal:
            self.dataEvent.emit(self._params(signal, **kwargs))

    @abstractmethod
    def _signal(self, data) -> tuple[misc.PS, dict]:
        ...

    def _params(self, signal, **kwargs) -> dict:
        """
        Must return a dict with all params required by `Portfolio` and
        `Execution Model`.  The dict may also be used to pass logging
        information.
        """

        params = {
            "signal": signal,
            "contract": self.contract,
            "key": self.key,
            "exec_model": self.exec_model,
        }
        params.update(kwargs)
        return params


@dataclass
class AbstractDfBrick(AbstractBaseBrick):
    lockable: bool  # TODO: get rid of this
    always_on: bool  # TODO: get rid of this
    signal_processor: SignalProcessor
    signal_column: str = "signal"

    def _signal(self, data) -> tuple[misc.PS, dict]:
        df_row = self.df_row(data).to_dict()
        signal = df_row[self.signal_column]
        return self.signal_processor.process_signal(signal, self), df_row

    def df_row(self, data) -> pd.Series:
        return self.df(data).iloc[-1]

    @abstractmethod
    def df(self, data) -> pd.DataFrame:
        ...
