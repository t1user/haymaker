import operator as op
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime
from functools import cached_property
from itertools import accumulate
from typing import Any, AsyncGenerator, Awaitable, Literal

import ib_insync as ibi
import pandas as pd

from . import misc
from .base import Atom
from .contract_selector import FutureSelector
from .datastore import AbstractBaseStore, AsyncAbstractBaseStore
from .saver import AbstractBaseSaver


class NoDataError(Exception):
    pass


class NonOverLappingDfsError(Exception):
    pass


class MissingContract(Exception):
    pass


@dataclass
class Sticher(Atom):
    saver: AbstractBaseSaver

    def __post_init__(self):
        super().__init__()

    def onStart(self, data: Any, *args: Any) -> None:
        super().onStart(data)
        contract = data.get("contract")
        assert contract, f"{self} received no contract onStart"
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


_contract_fields: list[str] = [f.name for f in fields(ibi.Contract)]


@dataclass
class FuturesSticher:
    source: dict[ibi.Future, pd.DataFrame]
    adjust_type: Literal["add", "mul", None] = "add"
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


@dataclass
class FuturesReader:
    symbol: str
    store: AbstractBaseStore

    @cached_property
    def _all_symbols(self) -> list[str]:
        return self.store.keys()

    @cached_property
    def _target_symbols(self) -> list[str]:
        return [
            symbol
            for symbol in self._all_symbols
            if symbol.startswith(self.symbol) and symbol.endswith("_FUT")
        ]

    @cached_property
    def _contracts(self) -> dict[ibi.Future, str]:
        # contracts are not sorted in any way
        return {
            future: symbol
            for future, symbol in {
                ibi.Contract.create(
                    **{
                        k: v
                        for k, v in self.store.read_metadata(symbol).items()
                        if k in _contract_fields
                    }
                ): symbol
                for symbol in self._target_symbols
            }.items()
            if isinstance(future, ibi.Future)
        }

    @cached_property
    def data(self) -> dict[ibi.Future, pd.DataFrame]:
        # df is not None because `_contracts` has only symbols that
        # exist in database -> make type checker shut-up
        return {
            contract: self.store.read(symbol)  # type: ignore
            for contract, symbol in self._contracts.items()
        }


@dataclass
class AsyncFuturesReader:
    symbol: str
    store: AsyncAbstractBaseStore

    async def _all_symbols(self) -> AsyncGenerator[str, None]:
        for symbol in await self.store.keys():
            yield symbol

    async def _target_symbols(self) -> AsyncGenerator[str, None]:
        async for symbol in self._all_symbols():
            if symbol.startswith(self.symbol) and symbol.endswith("_FUT"):
                yield symbol

    async def _contracts(self) -> AsyncGenerator[tuple[ibi.Future, str], None]:
        # contracts are not sorted in any way
        async for symbol in self._target_symbols():
            metadata = await self.store.read_metadata(symbol)
            if isinstance(
                (
                    future := ibi.Contract.create(
                        **{k: v for k, v in metadata.items() if k in _contract_fields}
                    )
                ),
                ibi.Future,
            ):
                yield (future, symbol)

    async def _data(self) -> AsyncGenerator[tuple[ibi.Future, pd.DataFrame], None]:
        # df is not None because `_contracts` has only symbols that
        # exist in database -> make type checker shut-up
        async for contract, symbol in self._contracts():
            df = await self.store.read(symbol)
            assert df is not None
            yield contract, df

    async def data(self) -> dict[ibi.Future, pd.DataFrame]:
        return {contract: df async for contract, df in self._data()}
