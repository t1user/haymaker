from datetime import datetime
from itertools import chain

import ib_insync as ibi
import pytest
from contract_details_for_registry import contract_specs, details

import haymaker.contract_registry as contract_registry_module
from haymaker.base import ActiveNext
from haymaker.contract_registry import ContractRegistry
from haymaker.contract_selector import FutureSelector
from haymaker.details_processor import Details
from haymaker.misc import general_to_specific_contract_class

# contract_specs are for [es, dax, gc]
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
    for contract_spec in contract_specs:
        registry.register_contract_spec(contract_spec)

    registry.reset_data(details)
    return registry


def test_bluprints_registered(registry):
    for contract_spec in contract_specs:
        registry.register_contract_spec(contract_spec)

    assert len(registry.contract_specs) == len(contract_specs)
    assert contract_specs[0] in registry.contract_specs
    assert contract_specs[1] in registry.contract_specs
    assert contract_specs[2] in registry.contract_specs


def test_contract_spec_returned_if_no_data_set(registry):
    contract = ibi.Future("NQ", "CME")
    registry.register_contract_spec(contract)
    assert registry.get_contract(contract) == contract


def test_reset_data(registry):
    for contract_spec in contract_specs:
        registry.register_contract_spec(contract_spec)

    registry.reset_data(details)
    assert len(registry.selectors) == len(contract_specs)


def test_reset_data_refreshes_selector_date_from_utc_clock(monkeypatch):
    dates = iter((datetime(2026, 7, 20), datetime(2026, 7, 21)))
    monkeypatch.setattr(
        contract_registry_module,
        "utc_now_naive",
        lambda: next(dates),
    )
    registry = ContractRegistry()
    for contract_spec in contract_specs:
        registry.register_contract_spec(contract_spec)

    registry.reset_data(details)
    first_dates = {selector.today for selector in registry.selectors}
    registry.reset_data(details)
    second_dates = {selector.today for selector in registry.selectors}

    assert first_dates == {datetime(2026, 7, 20)}
    assert second_dates == {datetime(2026, 7, 21)}


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_selector_accessible(registry_with_data, contract_index):
    assert registry_with_data.get_selector(contract_specs[contract_index])


def test_there_is_a_contract_spec_for_each_contract(registry_with_data):
    assert len(registry_with_data.contract_specs) == len(contract_specs)


def test_number_of_current_contracts(registry_with_data):
    # ACTIVE and NEXT for each contract_spec, but they are the same and this is set
    print(list(registry_with_data.current_contracts))
    assert len(registry_with_data.current_contracts) == len(contract_specs)


def test_number_of_all_contracts(registry_with_data):
    assert registry_with_data.all_contracts == {
        item.contract for chain in details for item in chain
    }


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_correct_selector_accessed(registry_with_data, contract_index):
    contract_spec = contract_specs[contract_index]
    contract_on_selector = registry_with_data.get_selector(
        contract_spec
    ).active_contract.symbol
    contract_on_contract_spec = contract_spec.symbol
    assert contract_on_contract_spec == contract_on_selector


def test_details_wrapped(registry_with_data):
    object_pulled_from_details = list(registry_with_data.details.values())[0]
    assert isinstance(object_pulled_from_details, Details)


def test_all_input_details_made_it_to_registry(registry_with_data):
    assert len(list(chain.from_iterable(details))) == len(registry_with_data.details)


def test_details_indices_qualified_contracts(registry_with_data):
    contract_spec = contract_specs[0]
    contract_on_selector = registry_with_data.get_selector(
        contract_spec
    ).active_contract
    assert registry_with_data.details.get(contract_on_selector)


def test_get_contract(registry_with_data):
    contract_spec = contract_specs[0]
    # details for one contract (corresponding to contract_spec)
    det = details[0]
    # manually created selector, same today as in `registry_with_data`
    selector = FutureSelector.from_details(det, today=datetime(2025, 12, 16))

    previous_contract = selector.previous_contract
    active_contract = selector.active_contract
    next_contract = selector.next_contract

    assert (
        registry_with_data.get_contract(contract_spec, ActiveNext.PREVIOUS)
        == previous_contract
    )
    assert (
        registry_with_data.get_contract(contract_spec, ActiveNext.ACTIVE)
        == active_contract
    )
    assert (
        registry_with_data.get_contract(contract_spec, ActiveNext.NEXT) == next_contract
    )


def test_series_identity_resolves_every_registered_expiry(registry_with_data):
    for contract_spec, chain in zip(contract_specs, details):
        expected = registry_with_data.hash_contract(contract_spec)
        assert {
            registry_with_data.series_key(
                general_to_specific_contract_class(item.contract)
            )
            for item in chain
        } == {expected}


def test_series_identity_exposes_current_active_and_next(registry_with_data):
    contract_spec = contract_specs[0]
    series_key = registry_with_data.hash_contract(contract_spec)
    selector = registry_with_data.get_selector(contract_spec)

    assert registry_with_data.active_for_series(series_key) is selector.active_contract
    assert registry_with_data.current_for_series(series_key) == (
        selector.active_contract,
        selector.next_contract,
    )


def test_series_identity_rejects_unregistered_con_id(registry_with_data):
    with pytest.raises(KeyError, match="No registered futures series"):
        registry_with_data.series_key(ibi.Future(conId=999_999))


def test_get_details(registry_with_data):
    # details for one contract (corresponding to contract_spec)
    det = details[0]
    contract = det[0].contract
    assert registry_with_data.get_details(contract).details == det[0]


def test_qualified_member_finds_original_contract_spec_and_full_chain(
    registry_with_data,
):
    """Qualification supplies identity without another user-defined key."""
    member = details[0][-1].contract
    assert registry_with_data.contract_spec_for(member) == contract_specs[0]
    assert registry_with_data.get_selector(member) is registry_with_data.get_selector(
        contract_specs[0]
    )
    assert set(registry_with_data.contracts_for(member)) == {
        d.contract for d in details[0]
    }


def test_distinct_contract_specs_with_same_qualification_fail_atomically():
    """Optional declaration fields cannot create duplicate physical series."""
    registry = ContractRegistry(today=datetime(2025, 12, 16))
    first = ibi.Future("ES", exchange="CME")
    second = ibi.Future("ES", exchange="CME", currency="USD")
    registry.register_contract_spec(first)
    registry.register_contract_spec(second)
    with pytest.raises(
        ValueError, match="multiple registered contract specifications"
    ) as error:
        registry.reset_data([details[0], details[0]])
    assert str(first) in str(error.value)
    assert str(second) in str(error.value)
    assert not registry.selectors
    assert not registry.all_contracts


def test_contract_spec_registration_is_insulated_from_qualification_mutation():
    """Broker requests may mutate their copies, not the declaration identity."""
    registry = ContractRegistry()
    original = ibi.Future("ES", exchange="CME")
    registry.register_contract_spec(original)
    original.conId = 999
    request = registry.contract_specs[0]
    request.currency = "USD"
    assert registry.contract_specs == [ibi.Future("ES", exchange="CME")]
