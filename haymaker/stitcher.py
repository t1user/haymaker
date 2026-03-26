import logging
import operator as op
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from itertools import zip_longest
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo

import ib_insync as ibi
import pandas as pd

from . import misc
from .contract_selector import FutureSelector, custom_bday

log = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


class NonOverLappingDfsError(Exception):
    pass


class MissingContract(Exception):
    pass


class CorruptData(Exception):
    pass


class InfoRecord(TypedDict):
    roll_day: datetime
    sync_point: datetime
    old_contract: str
    new_contract: str
    old_end: datetime
    new_beginning: datetime
    old_close: float
    new_close: float
    offset: float


@dataclass
class Info:
    records: list[InfoRecord] = field(repr=False, default_factory=list)

    def save(self, record: InfoRecord) -> None:
        self.records.append(record)

    def inspect(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.records).sort_values("sync_point").reset_index(drop=True)
        )


@dataclass
class FuturesStitcher:
    source: dict[ibi.Future, pd.DataFrame]
    adjust_type: Literal["add", "mul", None] = "add"
    selector: FutureSelector | None = None
    roll_bdays: int | None = None
    roll_margin_bdays: int | None = None
    tz_info: ZoneInfo = ZoneInfo("US/Eastern")
    roll_hour: int = 10  # what time roll will happen
    buffer_bdays: int = 3
    debug: bool = False
    _debug_object: Info | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        assert self.adjust_type is None or self.adjust_type in [
            "add",
            "mul",
        ], f"Unknown adjust type operator: {self.adjust_type}"
        if self.debug:
            self._debug_object = Info()

    @cached_property
    def operator(self) -> Callable:
        return {
            "add": op.add,
            "mul": op.mul,
            None: lambda x, *args: x,  # type: ignore
        }[self.adjust_type]

    @cached_property
    def reverse_operator(self) -> Callable:
        return {
            "add": op.sub,
            "mul": op.truediv,
            None: lambda x, *args: 0,  # type: ignore
        }[self.adjust_type]

    @property
    def _buffer_offset(self):
        return self.buffer_bdays * custom_bday

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
    def _dfs(self) -> list[tuple[str, pd.DataFrame, datetime]]:
        # it's ok to rely on dfs being sorted
        # (`self._selector.date_ranges` responsible for that)
        # accept dfs that are either tz-aware or tz-naive, whichever
        # is later, but not both
        output: list[tuple[str, pd.DataFrame, datetime]] = []
        tz_aware: bool | None = None

        for contract, (start_date, roll_day) in self._selector.date_ranges.items():
            try:
                df = self.source[contract]
            except KeyError:
                raise MissingContract(f"No df supplied for {contract.localSymbol}")
            if (aware := misc.is_timezone_aware(df)) is not tz_aware:
                output = []
                tz_aware = aware
            if not df.is_monotonic_increasing:
                raise CorruptData(
                    f"dataframe for {contract.localSymbol} is not monotonic increasing"
                )
            output.append(
                (
                    contract.localSymbol,
                    df.loc[self._tz(start_date, df) - self._buffer_offset :],
                    roll_day,
                )
            )
        if not output:
            raise NoDataError("No suitable data in provided source.")
        return output

    @cached_property
    def data(self) -> pd.DataFrame:
        offset = 0 if self.adjust_type == "add" else 1
        segments = []
        syncs = []
        dfs = list(zip(self._dfs[:-1], self._dfs[1:]))
        for i, ((old_contract, df0, roll_day), (new_contract, df1, _)) in enumerate(
            reversed(dfs)
        ):
            try:
                sync = self._sync_point(df0, df1, roll_day)
            except NonOverLappingDfsError as e:
                w = f"{type(e).__name__}: {e}"
                warnings.warn(w)
                log.warning(w)
                break

            # self.operator creates a cummulative version of offset
            offset = self.operator(
                (
                    current_offset := self.offset(
                        (old_close := df0.loc[sync].close),  # type: ignore
                        (new_close := df1.loc[sync].close),  # type: ignore
                    )
                ),
                offset,
            )
            adjusted = self.adjust(df0.loc[:sync], offset)
            if i == 0:
                segments.append(df1.loc[sync:])
            syncs.append(sync)
            segments.append(adjusted)

            if self.debug:
                assert self._debug_object
                self._debug_object.save(
                    InfoRecord(
                        {
                            "roll_day": roll_day,
                            "sync_point": sync,
                            "old_contract": old_contract,
                            "new_contract": new_contract,
                            "old_end": df0.index[-1],
                            "new_beginning": df1.index[0],
                            "old_close": old_close,  # type: ignore
                            "new_close": new_close,  # type: ignore
                            "offset": current_offset,
                        }
                    )
                )
        if segments:
            # clear out any extra data except what goes into the final
            # df `sync` has one less element than `segments` the last
            # `seg` will have no corresponding `sync` because it
            # doesn't need trimming
            cleaned_segments = [
                segments[0],
                *[
                    seg.loc[sync if sync is not None else seg.index[0] :]
                    for sync, seg in zip_longest(syncs[1:], segments[1:])
                ],
            ]

            df = pd.concat(list(reversed(cleaned_segments)))
            return df[~df.index.duplicated()]
        else:
            raise NoDataError("No compliant data to stich.")

    def offset(self, old_price: float, new_price: float) -> float:
        return self.reverse_operator(new_price, old_price)

    def adjust(self, df: pd.DataFrame, offset: float) -> pd.DataFrame:

        if self.adjust_type is None:
            return df

        non_adjustable = {"volume", "WAP", "barCount"}
        adjustable_columns = list(set(df.columns) - non_adjustable)
        non_adjustable_columns = list(set(df.columns) - set(adjustable_columns))

        new_df = self.operator(pd.DataFrame(df[adjustable_columns]), offset)
        new_df[non_adjustable_columns] = df[non_adjustable_columns]
        return new_df[df.columns]

    def _tz(self, date: datetime, df: pd.DataFrame) -> pd.Timestamp:
        if df.index.tz is None:  # type: ignore
            return pd.Timestamp(date)
        else:
            return pd.Timestamp(date, tz=self.tz_info)

    def _make_roll_timestamp(
        self, roll_day: datetime, df: pd.DataFrame
    ) -> pd.Timestamp:
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty or diffs.mode()[0] < timedelta(days=1):  # type: ignore
            roll_ts = self._tz(roll_day, df).replace(hour=self.roll_hour) - custom_bday
        else:
            roll_ts = self._tz(roll_day, df)
        return roll_ts

    def _sync_point(
        self, df0: pd.DataFrame, df1: pd.DataFrame, roll_day: datetime
    ) -> pd.Timestamp:
        roll_ts = self._make_roll_timestamp(roll_day, df0)
        if roll_ts in df0.index and roll_ts in df1.index:
            return roll_ts
        else:
            return self._find_common_index(df0, df1, roll_ts)

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
                f"Previous last point: {old_df.index[-1]}, "
                f"next first point: {new_df.index[0]} "
                f"sync index: {sync_index}"
            )
        inner_index_cut = inner.loc[:sync_index].index  # type: ignore
        return inner_index_cut[-1]

    def inspect(self) -> pd.DataFrame | None:
        if self._debug_object:
            return self._debug_object.inspect()
