from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any

import ib_insync as ibi
import pandas as pd

from .base import Atom
from .datastore import QueuedDataSink

log = logging.getLogger(__name__)


@dataclass
class AbstractBaseBlock(Atom, ABC):
    """Base strategy block that emits strategy-labelled signal data.

    Args:
        strategy: Unique strategy name used in runtime and persisted state.
        contract: Broker contract processed by this strategy.
        auto_roll_futures: Whether positions for this strategy participate in
            automatic futures rolling. This option is keyword-only.
    """

    strategy: str
    contract: ibi.Contract
    auto_roll_futures: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Initialize Atom services and register this strategy's roll policy."""

        Atom.__init__(self)
        self._register_future_roll_policy()

    def _register_future_roll_policy(self) -> None:
        """Register one consistent automatic futures-roll policy."""

        policies = self.runtime.future_roll_policies
        if (
            self.strategy in policies
            and policies[self.strategy] != self.auto_roll_futures
        ):
            raise ValueError(
                "Conflicting auto_roll_futures values for strategy "
                f"{self.strategy!r}."
            )
        policies[self.strategy] = self.auto_roll_futures

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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"<{self.strategy} "
            f"{self.contract.localSymbol or self.contract.symbol}>"
        )


@dataclass
class AbstractDfBlock(AbstractBaseBlock):
    """Base dataframe strategy block with optional frame persistence.

    Args:
        datastore: Fully configured queued sink for this block. When omitted,
            dataframe persistence is disabled.
    """

    strategy: str
    contract: ibi.Contract
    datastore: QueuedDataSink | None = field(default=None, kw_only=True, repr=False)

    def _signal(self, data) -> dict:
        return self.df_row(data).to_dict()

    def df_row(self, data) -> pd.Series:
        df = self._create_df(data)
        if self.datastore is not None:
            self.datastore.enqueue_append(self.contract, df)
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
