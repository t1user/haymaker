from dataclasses import dataclass, fields
from functools import cached_property
from typing import AsyncGenerator

import ib_insync as ibi
import pandas as pd

from .async_datastore import AsyncAbstractBaseStore
from .datastore import AbstractBaseStore

_contract_fields: list[str] = [f.name for f in fields(ibi.Contract)]


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
