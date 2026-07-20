from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

import ib_insync as ibi

SymbolNamer: TypeAlias = Callable[[ibi.Contract], str]


def simple_collection_namer(contract: ibi.Contract) -> str:
    assert isinstance(contract, ibi.Contract)
    return f'{"_".join(contract.localSymbol.split())}_{contract.secType}'


@dataclass(frozen=True)
class CollectionNamerBarsizeSetting:
    barSizeSetting: str
    _barSizeSetting: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize the bar-size component used in persisted symbols."""

        _barSizeSetting = self.barSizeSetting.replace(" ", "_")
        object.__setattr__(
            self,
            "_barSizeSetting",
            _barSizeSetting[:-1] if _barSizeSetting.endswith("s") else _barSizeSetting,
        )

    def __call__(self, contract: ibi.Contract) -> str:
        assert isinstance(contract, ibi.Contract)
        return (
            f'{"_".join(contract.localSymbol.split())}_{contract.secType}_'
            f"{self._barSizeSetting}"
        )


@dataclass(frozen=True)
class CollectionNamerStrategySymbol:
    strategy: str
    _timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    def __call__(self, contract: ibi.Contract) -> str:
        return f"{self.strategy}_{contract.localSymbol}_{self._timestamp}"
