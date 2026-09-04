from collections import UserDict
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from typing import TypeAlias

import ib_insync as ibi

from .contract_selector import (
    AbstractBaseContractSelector,
    selector_factory,
    utc_now_naive,
)
from .details_processor import Details
from .enums import ActiveNext
from .misc import general_to_specific_contract_class

log = getLogger(__name__)


ContractKey: TypeAlias = str
# this is an unqualified contract (no conId) that can't be hashed
Blueprint: TypeAlias = ibi.Contract


class DetailsContainer(UserDict):
    def __setitem__(self, key: ibi.Contract, value: ibi.ContractDetails) -> None:
        super().__setitem__(key, Details(value))


@dataclass
class ContractRegistry:
    futures_roll_bdays: int = 3
    futures_roll_margin_bdays: int = 3
    # mapping of contract object saved on Atom instance to hash used by ContractRegistry
    _blueprints: dict[ContractKey, ibi.Contract] = field(
        default_factory=dict, init=False, repr=False
    )
    _selectors: dict[ContractKey, AbstractBaseContractSelector] = field(
        default_factory=dict, init=False, repr=False
    )
    _series_by_con_id: dict[int, ContractKey] = field(
        default_factory=dict, init=False, repr=False
    )
    details: DetailsContainer = field(default_factory=DetailsContainer, repr=False)
    today: datetime | None = None  # for testing only

    @staticmethod
    def hash_contract(contract: Blueprint) -> ContractKey:
        # hashing method can be changed, potential alternatives:
        # return tuple(ibi.util.dataclassNonDefaults(contract).items())
        # return hash(contractAsTuple(contract))
        return str(contract)

    @property
    def _today(self) -> datetime:
        return self.today or utc_now_naive()

    def register_blueprint(self, blueprint: Blueprint) -> None:
        self._blueprints[self.hash_contract(blueprint)] = blueprint

    def get_contract(
        self, blueprint: Blueprint, which: ActiveNext = ActiveNext.ACTIVE
    ) -> ibi.Contract | None:
        selector = self.get_selector(blueprint)
        if selector:
            return general_to_specific_contract_class(
                getattr(selector, f"{which.name.lower()}_contract")
            )
        else:
            return self._blueprints.get(self.hash_contract(blueprint))

    def get_selector(self, blueprint: Blueprint) -> AbstractBaseContractSelector | None:
        return self._selectors.get(self.hash_contract(blueprint))

    def get_details(self, contract: ibi.Contract | None) -> Details | None:
        if contract is None:
            return None
        else:
            try:
                return self.details.get(contract)
            except ValueError:
                return None

    def reset_data(self, input_details: list[list[ibi.ContractDetails]]) -> None:
        today = self._today
        self._series_by_con_id.clear()

        for blueprint, details_list in zip(self._blueprints, input_details):
            self._selectors[blueprint] = selector_factory(
                details_list,
                self.futures_roll_bdays,
                self.futures_roll_margin_bdays,
                today=today,
            )

            for details in details_list:
                if details.contract:
                    self.details[details.contract] = details
                    con_id = details.contract.conId
                    if con_id:
                        existing = self._series_by_con_id.get(con_id)
                        if existing is not None and existing != blueprint:
                            raise ValueError(
                                f"conId={con_id} belongs to multiple "
                                "registered futures series"
                            )
                        self._series_by_con_id[con_id] = blueprint

    def series_key(self, contract: ibi.Future) -> ContractKey:
        """Return the registered blueprint identity for a qualified Future.

        Args:
            contract: Qualified Future from any expiry in the registered chain.

        Raises:
            TypeError: If contract is not an IB Future.
            KeyError: If its conId is not owned by exactly one registered series.
        """

        if not isinstance(contract, ibi.Future):
            raise TypeError("contract must be an ib_insync.Future")
        if not contract.conId:
            raise ValueError("contract must have a non-zero conId")
        try:
            return self._series_by_con_id[contract.conId]
        except KeyError as exc:
            raise KeyError(
                f"No registered futures series owns conId={contract.conId}"
            ) from exc

    def selector_for_series(
        self, series_key: ContractKey
    ) -> AbstractBaseContractSelector:
        """Return the initialized selector for one registry series key."""

        try:
            return self._selectors[series_key]
        except KeyError as exc:
            raise KeyError(f"Unknown futures series key: {series_key!r}") from exc

    def active_for_series(self, series_key: ContractKey) -> ibi.Future:
        """Return the current ACTIVE Future for one registered series."""

        contract = self.selector_for_series(series_key).active_contract
        if not isinstance(contract, ibi.Future):
            raise TypeError(f"Series {series_key!r} does not resolve to a Future")
        return contract

    def current_for_series(
        self, series_key: ContractKey
    ) -> tuple[ibi.Future, ibi.Future]:
        """Return the acceptable ACTIVE and NEXT Futures for one series."""

        selector = self.selector_for_series(series_key)
        active = selector.active_contract
        next_contract = selector.next_contract
        if not isinstance(active, ibi.Future) or not isinstance(
            next_contract, ibi.Future
        ):
            raise TypeError(f"Series {series_key!r} does not resolve to Futures")
        return active, next_contract

    def active_contracts_for_logs(self) -> list[str]:
        return [
            selector.active_contract.localSymbol
            for selector in self._selectors.values()
        ]

    @property
    def current_contracts(self) -> set[ibi.Contract]:
        return {
            contract
            for selector in self._selectors.values()
            for contract in (selector.active_contract, selector.next_contract)
        }

    @property
    def all_contracts(self) -> set[ibi.Contract]:
        return {
            contract
            for selector in self._selectors.values()
            for contract in (
                selector.active_contract,
                selector.next_contract,
                selector.previous_contract,
            )
        }

    @property
    def blueprints(self) -> list[ibi.Contract]:
        return list(self._blueprints.values())

    @property
    def selectors(self) -> list[AbstractBaseContractSelector]:
        return list(self._selectors.values())
