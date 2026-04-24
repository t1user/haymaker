from dataclasses import dataclass, field
from datetime import datetime, timezone

import ib_insync as ibi


def simple_collection_namer(contract: ibi.Contract):
    assert isinstance(contract, ibi.Contract)
    return f'{"_".join(contract.localSymbol.split())}_{contract.secType}'


@dataclass
class CollectionNamerBarsizeSetting:
    barSizeSetting: str
    _barSizeSetting: str = field(init=False, repr=False)

    def __post_init__(self):
        _barSizeSetting = self.barSizeSetting.replace(" ", "_")
        self._barSizeSetting = (
            _barSizeSetting[:-1] if _barSizeSetting.endswith("s") else _barSizeSetting
        )

    def __call__(self, contract: ibi.Contract) -> str:
        assert isinstance(contract, ibi.Contract)
        return (
            f'{"_".join(contract.localSymbol.split())}_{contract.secType}_'
            f"{self._barSizeSetting}"
        )


@dataclass
class CollectionNamerStrategySymbol:
    strategy: str
    _timestamp: datetime = datetime.now(timezone.utc)

    def __call__(self, contract: ibi.Contract) -> str:
        return f"{self.strategy}_{contract.localSymbol}_{self._timestamp}"
