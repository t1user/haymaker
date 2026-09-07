from collections import UserDict
from copy import deepcopy
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
        """Capture an unchanging declaration; identical declarations share it."""
        self._blueprints.setdefault(self.hash_contract(blueprint), deepcopy(blueprint))

    def blueprint_key(self, contract: ibi.Contract) -> ContractKey:
        """Identify a registered declaration or one of its qualified members.

        Raises:
            KeyError: If no registered blueprint owns the Contract. Membership
                comes from qualification, never from guessed symbol matching.
        """
        key = self.hash_contract(contract)
        if key in self._blueprints:
            return key
        try:
            return self._series_by_con_id[contract.conId]
        except KeyError as exc:
            raise KeyError(f"No registered blueprint owns {contract!r}") from exc

    def blueprint_for(self, contract: ibi.Contract) -> Blueprint:
        """Return a copy of the declaration owning a blueprint or qualified member."""
        return deepcopy(self._blueprints[self.blueprint_key(contract)])

    def contracts_for(self, contract: ibi.Contract) -> tuple[ibi.Contract, ...]:
        """Return all qualified members, including past and later futures expiries.

        This is a membership query, not a trading-eligibility decision.
        """
        key = self.blueprint_key(contract)
        return tuple(
            member
            for member in self.details
            if self._series_by_con_id.get(member.conId) == key
        )

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
        """Look up the initialized selector from a declaration or qualified member."""
        try:
            return self._selectors.get(self.blueprint_key(blueprint))
        except KeyError:
            return None

    def get_details(self, contract: ibi.Contract | None) -> Details | None:
        if contract is None:
            return None
        else:
            try:
                return self.details.get(contract)
            except ValueError:
                return None

    def reset_data(self, input_details: list[list[ibi.ContractDetails]]) -> None:
        """Atomically rebuild selectors and reject overlapping declarations."""
        today = self._today
        selectors: dict[ContractKey, AbstractBaseContractSelector] = {}
        members: dict[int, ContractKey] = {}
        details = DetailsContainer()

        for blueprint, details_list in zip(
            self._blueprints, input_details, strict=True
        ):
            selectors[blueprint] = selector_factory(
                details_list,
                self.futures_roll_bdays,
                self.futures_roll_margin_bdays,
                today=today,
            )

            for item in details_list:
                if item.contract:
                    details[item.contract] = item
                    con_id = item.contract.conId
                    if con_id:
                        existing = members.get(con_id)
                        if existing is not None and existing != blueprint:
                            raise ValueError(
                                f"conId={con_id} belongs to multiple "
                                f"registered blueprints: {existing!r} and {blueprint!r}"
                            )
                        members[con_id] = blueprint
        self._selectors = selectors
        self._series_by_con_id = members
        self.details = details

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
        """Return every qualified registered member, not only adjacent expiries."""
        return set(self.details)

    @property
    def blueprints(self) -> list[ibi.Contract]:
        """Return declaration copies safe to pass to broker qualification."""
        return deepcopy(list(self._blueprints.values()))

    @property
    def selectors(self) -> list[AbstractBaseContractSelector]:
        return list(self._selectors.values())
