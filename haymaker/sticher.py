import operator as op
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Awaitable, Literal

import ib_insync as ibi
import pandas as pd

from . import misc
from .base import Atom
from .contract_selector import FutureSelector
from .datastore import AbstractBaseStore
from .saver import AbstractBaseSaver


class NoDataError(Exception):
    pass


class NonOverLappingDfsError(Exception):
    pass


@dataclass
class Sticher(Atom):
    contract: ibi.Contract
    saver: AbstractBaseSaver

    def __post_init__(self):
        super().__init__()

    def onStart(self, data: Any, *args: Any) -> None:
        super().onStart(data)
        self.pull_data()

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
    symbol: str
    store: AbstractBaseStore
    adjust_type: Literal["add", "mul", None] = "add"
    _contract_fields: list[str] = field(
        default_factory=lambda: [field.name for field in fields(ibi.Contract)],
        repr=False,
    )

    def __post_init__(self):
        assert self.adjust_type is None or self.adjust_type in ["add", "mul"]

    @cached_property
    def operator(self) -> Callable | None:
        return {"add": op.add, "mul": op.mul, None: None}[self.adjust_type]

    @cached_property
    def reverse_operator(self) -> Callable:
        return {"add": op.sub, "mul": op.truediv, None: lambda x, y: 0}[
            self.adjust_type
        ]

    @cached_property
    def _target_symbols(self) -> list[str]:
        return [
            symbol
            for symbol in self._all_symbols
            if symbol.startswith(self.symbol) and symbol.endswith("_FUT")
        ]

    @cached_property
    def _all_symbols(self) -> list[str]:
        return self.store.keys()

    @cached_property
    def _contracts(self) -> dict[ibi.Future, str]:
        # contracts are not sorted, `_selector` will sort them
        return {
            future: symbol
            for future, symbol in {
                ibi.Contract.create(
                    **{
                        k: v
                        for k, v in self.store.read_metadata(symbol).items()
                        if k in self._contract_fields
                    }
                ): symbol
                for symbol in self._target_symbols
            }.items()
            if isinstance(future, ibi.Future)
        }

    @cached_property
    def _selector(self) -> FutureSelector:
        return FutureSelector.from_contracts(
            list(self._contracts.keys()),
        )

    @cached_property
    def _timedelta(self) -> timedelta:
        roll_dates = list(self._selector.back_roll_days.values())
        if len(roll_dates) < 2:
            return timedelta(days=90)
        else:
            return min([b - a for a, b in zip(roll_dates[:-1], roll_dates[1:])])

    @cached_property
    def _df_list(self) -> list[tuple[pd.DataFrame, datetime]]:
        # This method reads data from database
        assert len(self._target_symbols) == len(self._selector.back_roll_days)
        return [
            (
                # df is not None because `_target_symbols` has only symbols
                # that exist in database
                (df := self.store.read(self._contracts[contract])),
                pd.Timestamp(roll_date, tz=df.index.tzinfo),  # type: ignore
            )
            for contract, roll_date in self._selector.back_roll_days.items()
        ]

    @cached_property
    def _filtered_df_list(self) -> list[tuple[pd.DataFrame, datetime]]:
        dfs: list[tuple[pd.DataFrame, datetime]] = []
        for df, roll_date in self._df_list:
            if df is None or df.empty or not misc.is_timezone_aware(df):
                dfs = []
            else:
                dfs.append((df, roll_date))
        if dfs:
            return dfs
        else:
            raise NoDataError("No non-empty dataframes returned from database.")

    @cached_property
    def df(self) -> pd.DataFrame:
        full_df: pd.DataFrame | None = None
        for df, roll_date in self._filtered_df_list:
            if full_df is None:
                # this is first df only
                start = max(roll_date - self._timedelta, df.index[0])
                full_df = df.loc[start:roll_date]  # type: ignore
            else:
                # process df by chosing only relevant rows
                start_index = full_df.index[-1]
                end_index = min(roll_date, df.index[-1])
                new_df = df.loc[start_index:end_index]

                # find sync row
                sync_index = full_df.index[-1]

                if sync_index in df.index:
                    old_row = full_df.loc[sync_index]
                    new_row = df.loc[sync_index]
                else:
                    try:
                        new_sync_index = self._find_common_index(
                            full_df, df, sync_index
                        )
                        old_row = full_df.loc[new_sync_index]
                        new_row = df.loc[new_sync_index]
                    except NonOverLappingDfsError:
                        full_df = df.loc[roll_date:]  # type: ignore
                        continue

                offset = self.offset(old_row.close, new_row.close)

                new_full_df = self.adjust(full_df, offset)
                full_df = pd.concat([new_full_df, new_df.iloc[1:]])

        if full_df is None or full_df.empty:
            raise NoDataError()
        else:
            return full_df

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
