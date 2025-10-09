import logging
import operator as op
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from itertools import accumulate
from typing import Any, Awaitable, Literal

import ib_insync as ibi
import pandas as pd

from . import misc
from .base import ActiveNext, Atom
from .contract_selector import FutureSelector
from .saver import AbstractBaseSaver

log = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


class NonOverLappingDfsError(Exception):
    pass


class MissingContract(Exception):
    pass


@dataclass
class Sticher(Atom):
    # this object totally re-writes the way Atom looks up contracts
    # because it needs very specific access to particular contract on
    # a futures chain
    saver: AbstractBaseSaver
    _contract: ibi.Contract | None = field(init=False)

    def __post_init__(self):
        self._contract = None
        super().__init__()

    def onStart(self, data: Any, *args: Any) -> None:
        super().onStart(data)
        contract = data.get("contract")
        assert contract, f"{self} received no contract onStart"
        # don't write to `self.contract` as that will be  directed to descriptor
        self._contract = contract

        self.pull_data()

    def _contract_getter(self, tag: ActiveNext) -> ibi.Contract:
        assert self._contract is not None, f"Contract has not been set on {self}"
        contract = self.contract_dict.get((misc.hash_contract(self._contract), tag))
        if contract is None:
            raise ValueError(f"Missing {tag.name} contract on {self}")
        else:
            return contract

    @property
    def previous_contract(self) -> ibi.Contract:
        return self._contract_getter(ActiveNext.PREVIOUS)

    @property
    def active_contract(self) -> ibi.Contract:
        return self._contract_getter(ActiveNext.ACTIVE)

    @property
    def next_contract(self) -> ibi.Contract:
        return self._contract_getter(ActiveNext.NEXT)

    def pull_data(self):
        pass

    def onData(self, data: Any, *args: Any) -> Awaitable[None] | None:
        self.save_data(data)
        processed_data = self.process_data(data)
        super().onData(data)
        self.dataEvent.emit(processed_data)
        return None

    def save_data(self, data):
        pass

    def process_data(self, data: dict):
        pass


@dataclass
class FuturesSticher:
    source: dict[ibi.Future, pd.DataFrame]
    adjust_type: Literal["add", "mul", None] = "add"
    selector: FutureSelector | None = None
    roll_bdays: int | None = None
    roll_margin_bdays: int | None = None

    def __post_init__(self):
        assert self.adjust_type is None or self.adjust_type in [
            "add",
            "mul",
        ], f"Unknown adjust type operator: {self.adjust_type}"

    @cached_property
    def operator(self) -> Callable | None:
        return {"add": op.add, "mul": op.mul, None: None}[self.adjust_type]

    @cached_property
    def reverse_operator(self) -> Callable:
        return {"add": op.sub, "mul": op.truediv, None: lambda x, y: 0}[
            self.adjust_type
        ]

    @cached_property
    def _selector(self) -> FutureSelector:
        if self.selector is not None:
            return self.selector
        else:
            params = {
                param: p
                for param in ("roll_bdays", "roll_margin_bdays")
                if (p := getattr(self, param)) is not None
            }
        return FutureSelector.from_contracts(list(self.source.keys()), **params)

    @cached_property
    def _date_ranges(self) -> dict[ibi.Future, tuple[datetime, datetime]]:

        return {
            contract: (start, stop)
            for (contract, start, stop) in self._selector.date_ranges
        }

    @staticmethod
    def _tz(date: datetime, df: pd.DataFrame) -> pd.Timestamp:
        # just copy timezone from corresponding df even if I get it
        # wrong, stiching point might get shifted a couple hours but
        # it will not matter as long as the day is roughly correct
        return pd.Timestamp(date, tz=df.index.tzinfo)  # type: ignore

    @cached_property
    def _dfs(self) -> list[pd.DataFrame]:
        output: list[pd.DataFrame] = []
        # it's ok to rely on dfs being sorted (`self._date_ranges`
        # responsible for that)
        for contract, (start_date, stop_date) in self._date_ranges.items():
            try:
                df = self.source[contract]
            except KeyError:
                # impossible to occur if code working
                raise MissingContract("No df supplied for {contract}")

            df_slice = df.loc[
                self._tz(start_date, df) : self._tz(stop_date, df)  # type: ignore
            ]
            if df_slice.empty or not misc.is_timezone_aware(df):
                # one unusable df -> start-over
                output = []
            else:
                output.append(df_slice)
        # No verification at this point if the whole series is continuous
        # handled in `self._offsets` -> syncing indices
        return output

    @cached_property
    def _offsets(self) -> list[float]:
        offsets: list[float] = []
        for df0, df1 in zip(self._dfs[1:], self._dfs[:-1]):
            # find sync row
            sync_index = df0.index[-1]

            if sync_index in df1.index:
                old_row = df0.loc[sync_index]
                new_row = df1.loc[sync_index]
            else:
                try:
                    new_sync_index = self._find_common_index(df0, df1, sync_index)
                    old_row = df0.loc[new_sync_index]
                    new_row = df1.loc[new_sync_index]
                except NonOverLappingDfsError:
                    # start over; discard all previous values because
                    # there's a gap between dfs impossible to reconcile
                    offsets = []
                    continue

            offsets.append(self.offset(old_row.close, new_row.close))
        if offsets:
            return offsets
        else:
            raise NoDataError("No suitable data to create a continuous df.")

    @cached_property
    def data(self) -> pd.DataFrame:
        assert len(self._offsets) == len(self._dfs) - 1
        dfs = list(
            # we're going back to front
            reversed(
                [
                    # latest df doesn't need adjusting
                    self._dfs[-1],
                    *[
                        self.adjust(df, offset)
                        for df, offset in zip(
                            # going from end to start
                            # skipping last df, which needs no adjstment
                            reversed(self._dfs[:-1]),
                            # no offset for last df
                            accumulate(reversed(self._offsets)),
                        )
                    ],
                ]
            )
        )
        df = pd.concat(dfs)
        # duplicates on df joining points
        return df[~df.index.duplicated()]

    def offset(self, old_price: float, new_price: float) -> float:
        return self.reverse_operator(new_price, old_price)

    def adjust(self, df: pd.DataFrame, offset: float) -> pd.DataFrame:

        if self.operator is None:
            return df

        non_adjustable = {"volume", "WAP", "barCount"}
        adjustable_columns = list(set(df.columns) - non_adjustable)
        non_adjustable_columns = list(set(df.columns) - set(adjustable_columns))

        new_df = self.operator(pd.DataFrame(df[adjustable_columns]), offset)
        new_df[non_adjustable_columns] = df[non_adjustable_columns]
        return new_df[df.columns]

    @staticmethod
    def _find_common_index(
        old_df: pd.DataFrame, new_df: pd.DataFrame, sync_index: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Find the last point before `sync_index` that both dfs have.
        """
        inner = old_df.join(new_df, how="inner", rsuffix="_new")
        if inner.empty:
            raise NonOverLappingDfsError(
                f"Previous last point: {old_df.index[1]}, "
                f"next first point: {new_df.index[0]}"
            )
        inner_index_cut = inner.loc[:sync_index].index  # type: ignore
        return inner_index_cut[-1]
