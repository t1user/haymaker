from collections import UserDict
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from typing import TypeAlias

import ib_insync as ibi

from .config import CONFIG
from .contract_selector import AbstractBaseContractSelector, selector_factory
from .details_processor import Details
from .enums import ActiveNext
from .misc import general_to_specific_contract_class

log = getLogger(__name__)


FUTURES_ROLL_BDAYS = CONFIG["futures_roll_bdays"]
FUTURES_ROLL_MARGIN_BDAYS = CONFIG["futures_roll_margin_bdays"]


ContractKey: TypeAlias = str
# this is an unqualified contract (no conId) that can't be hashed
Blueprint: TypeAlias = ibi.Contract


class DetailsContainer(UserDict):
    def __setitem__(self, key: ibi.Contract, value: ibi.ContractDetails) -> None:
        super().__setitem__(key, Details(value))


@dataclass
class ContractRegistry:
    # mapping of contract object saved on Atom instance to hash used by ContractRegistry
    __blueprints: dict[ContractKey, ibi.Contract] = field(default_factory=dict)
    __selectors: dict[ContractKey, AbstractBaseContractSelector] = field(
        default_factory=dict
    )
    details: DetailsContainer = field(default_factory=DetailsContainer, repr=False)
    today: datetime = datetime.now()  # for testing only

    @staticmethod
    def hash_contract(contract: Blueprint) -> ContractKey:
        # hashing method can be changed, potential alternatives:
        # return tuple(ibi.util.dataclassNonDefaults(contract).items())
        # return hash(contractAsTuple(contract))
        return str(contract)

    def register_blueprint(self, blueprint: Blueprint) -> None:
        self.__blueprints[self.hash_contract(blueprint)] = blueprint

    def get_contract(
        self, blueprint: Blueprint, which: ActiveNext = ActiveNext.ACTIVE
    ) -> ibi.Contract | None:
        selector = self.get_selector(blueprint)
        if selector:
            return general_to_specific_contract_class(
                getattr(selector, f"{which.name.lower()}_contract")
            )
        else:
            return self.__blueprints.get(self.hash_contract(blueprint))

    def get_selector(self, blueprint: Blueprint) -> AbstractBaseContractSelector | None:
        return self.__selectors.get(self.hash_contract(blueprint))

    def get_details(self, contract: ibi.Contract | None) -> Details | None:
        if contract is None:
            return None
        else:
            return self.details.get(contract)

    def reset_data(self, input_details: list[list[ibi.ContractDetails]]) -> None:

        for blueprint, details_list in zip(self.__blueprints, input_details):
            self.__selectors[blueprint] = selector_factory(
                details_list,
                FUTURES_ROLL_BDAYS,
                FUTURES_ROLL_MARGIN_BDAYS,
                today=self.today,
            )

            for details in details_list:
                if details.contract:
                    self.details[details.contract] = details

    @property
    def current_contracts(self) -> set[ibi.Contract]:
        return {
            contract
            for selector in self.__selectors.values()
            for contract in (selector.active_contract, selector.next_contract)
        }

    @property
    def all_contracts(self) -> set[ibi.Contract]:
        return {
            contract
            for selector in self.__selectors.values()
            for contract in (
                selector.active_contract,
                selector.next_contract,
                selector.previous_contract,
            )
        }

    @property
    def blueprints(self) -> list[ibi.Contract]:
        return list(self.__blueprints.values())

    @property
    def selectors(self) -> list[AbstractBaseContractSelector]:
        return list(self.__selectors.values())
