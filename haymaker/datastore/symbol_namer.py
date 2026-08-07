from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

import ib_insync as ibi

SymbolNamer: TypeAlias = Callable[[ibi.Contract], str]


def simple_symbol_namer(contract: ibi.Contract) -> str:
    assert isinstance(contract, ibi.Contract)
    return f'{"_".join(contract.localSymbol.split())}_{contract.secType}'


@dataclass(frozen=True)
class BarSizeSymbolNamer:
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
class MarketDataSymbolNamer:
    """Name persisted broker-bar series by their material request identity.

    Args:
        barSizeSetting: Interactive Brokers bar-size value.
        whatToShow: Interactive Brokers market-data type, such as ``TRADES``.
        useRTH: Whether the series contains regular-trading-hours bars only.

    The Contract, bar size, data type, and trading-hours policy all affect the
    stored series. Including each field prevents a runtime-default datastore
    from mixing bars produced by incompatible historical requests.
    """

    barSizeSetting: str
    whatToShow: str
    useRTH: bool
    _barSizeSetting: str = field(init=False, repr=False)
    _whatToShow: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the request fields used in symbols."""

        if not isinstance(self.barSizeSetting, str):
            raise TypeError("barSizeSetting must be a string")
        if not self.barSizeSetting:
            raise ValueError("barSizeSetting must not be empty")
        if not isinstance(self.whatToShow, str):
            raise TypeError("whatToShow must be a string")
        if not self.whatToShow:
            raise ValueError("whatToShow must not be empty")
        if not isinstance(self.useRTH, bool):
            raise TypeError("useRTH must be a bool")

        bar_size = self.barSizeSetting.replace(" ", "_")
        object.__setattr__(
            self,
            "_barSizeSetting",
            bar_size[:-1] if bar_size.endswith("s") else bar_size,
        )
        object.__setattr__(
            self,
            "_whatToShow",
            "_".join(self.whatToShow.upper().split()),
        )

    @property
    def identity(self) -> tuple[str, str, bool]:
        """Return the normalized identity used to cache one datastore."""

        return self._barSizeSetting, self._whatToShow, self.useRTH

    def __call__(self, contract: ibi.Contract) -> str:
        """Return the collision-safe persisted symbol for ``contract``."""

        session = "RTH" if self.useRTH else "ALL"
        return (
            f"{simple_symbol_namer(contract)}_{self._barSizeSetting}_"
            f"{self._whatToShow}_{session}"
        )


@dataclass(frozen=True)
class StrategySymbolNamer:
    """Build one run-scoped collection name per strategy and root symbol."""

    strategy: str
    _timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    def __call__(self, contract: ibi.Contract) -> str:
        return f"{self.strategy}_{contract.symbol}_{self._timestamp}"
