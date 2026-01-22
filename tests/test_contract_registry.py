from datetime import datetime
from itertools import chain

import ib_insync as ibi
import pytest
from contract_details_for_registry import blueprints, details

from haymaker.base import ActiveNext
from haymaker.contract_registry import ContractRegistry
from haymaker.contract_selector import FutureSelector
from haymaker.details_processor import Details

# blueprints are for [es, dax, gc]
# details are for corresponding contracts


def test_hash_contract():
    # test if result hashable
    hash(ContractRegistry.hash_contract(ibi.ContFuture("NQ", "CME")))


@pytest.fixture
def registry():
    return ContractRegistry()


@pytest.fixture(scope="module")
def registry_with_data():
    registry = ContractRegistry(today=datetime(2025, 12, 16))
    for blueprint in blueprints:
        registry.register_blueprint(blueprint)

    registry.reset_data(details)
    return registry


def test_bluprints_registered(registry):
    for blueprint in blueprints:
        registry.register_blueprint(blueprint)

    assert len(registry.blueprints) == len(blueprints)
    assert blueprints[0] in registry.blueprints
    assert blueprints[1] in registry.blueprints
    assert blueprints[2] in registry.blueprints


def test_blueprint_returned_if_no_data_set(registry):
    contract = ibi.Future("NQ", "CME")
    registry.register_blueprint(contract)
    assert registry.get_contract(contract) == contract


def test_reset_data(registry):
    for blueprint in blueprints:
        registry.register_blueprint(blueprint)

    registry.reset_data(details)
    assert len(registry.selectors) == len(blueprints)


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_selector_accessible(registry_with_data, contract_index):
    assert registry_with_data.get_selector(blueprints[contract_index])


def test_there_is_a_blueprint_for_each_contract(registry_with_data):
    assert len(registry_with_data.blueprints) == len(blueprints)


def test_number_of_current_contracts(registry_with_data):
    # ACTIVE and NEXT for each blueprint, but they are the same and this is set
    print(list(registry_with_data.current_contracts))
    assert len(registry_with_data.current_contracts) == len(blueprints)


def test_number_of_all_contracts(registry_with_data):
    # ACTIVE, NEXT (same) and PREVIOUS (different) for each blueprint
    assert len(registry_with_data.all_contracts) == len(blueprints) * 2


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_correct_selector_accessed(registry_with_data, contract_index):
    blueprint = blueprints[contract_index]
    contract_on_selector = registry_with_data.get_selector(
        blueprint
    ).active_contract.symbol
    contract_on_blueprint = blueprint.symbol
    assert contract_on_blueprint == contract_on_selector


def test_details_wrapped(registry_with_data):
    object_pulled_from_details = list(registry_with_data.details.values())[0]
    assert isinstance(object_pulled_from_details, Details)


def test_all_input_details_made_it_to_registry(registry_with_data):
    assert len(list(chain.from_iterable(details))) == len(registry_with_data.details)


def test_details_indices_qualified_contracts(registry_with_data):
    blueprint = blueprints[0]
    contract_on_selector = registry_with_data.get_selector(blueprint).active_contract
    assert registry_with_data.details.get(contract_on_selector)


def test_get_contract(registry_with_data):
    blueprint = blueprints[0]
    # details for one contract (corresponding to blueprint)
    det = details[0]
    # manually created selector
    selector = FutureSelector.from_details(det)

    previous_contract = selector.previous_contract
    active_contract = selector.active_contract
    next_contract = selector.next_contract

    assert (
        registry_with_data.get_contract(blueprint, ActiveNext.PREVIOUS)
        == previous_contract
    )
    assert (
        registry_with_data.get_contract(blueprint, ActiveNext.ACTIVE) == active_contract
    )
    assert registry_with_data.get_contract(blueprint, ActiveNext.NEXT) == next_contract


def test_get_details(registry_with_data):
    # details for one contract (corresponding to blueprint)
    det = details[0]
    contract = det[0].contract
    assert registry_with_data.get_details(contract).details == det[0]
