import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from contract_details_for_registry import blueprints, details
from helpers import wait_for_condition
from sample_barDataList import sample_barDataList

from haymaker.base import ActiveNext, Atom
from haymaker.components.dataframe_aggregators import (
    FuturesPandasAggregator,
    VolumeGrouper,
    WrongStreamer,
)
from haymaker.components.streamers import HistoricalDataStreamer, MktDataStreamer
from haymaker.contract_registry import ContractRegistry
from haymaker.contract_selector import custom_bday
from haymaker.datastore import AsyncDataStore


@pytest.fixture(scope="module")
def registry_with_data():
    # this date ensures that active and next contracts are different
    # some tests rely on this
    registry = ContractRegistry(today=datetime(2025, 12, 12))
    for blueprint in blueprints:
        registry.register_blueprint(blueprint)

    registry.reset_data(details)
    return registry


@pytest.fixture(autouse=True)
def install_atom_runtime(atom_runtime):
    """Install the default Atom runtime for DataFrame aggregator tests."""

    return atom_runtime


@pytest.fixture
def registry_runtime(atom_runtime_factory, registry_with_data):
    """Install the populated futures registry for DataFrame aggregator tests."""

    return atom_runtime_factory(contract_registry=registry_with_data)


def make_aggregator() -> FuturesPandasAggregator:
    """Return an in-memory test aggregator with an explicit datastore."""

    return FuturesPandasAggregator(
        datastore=Mock(spec=AsyncDataStore), save_frequency=0
    )


def test_HistoricalDataStreamerAccepted():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )

    aggregator = make_aggregator()
    # test if no error raised
    assert aggregator.validate_source(streamer) is None


def test_wrong_streamer_fails():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = MktDataStreamer(contract=blueprint, tickList="212")
    aggregator = make_aggregator()
    with pytest.raises(WrongStreamer):
        aggregator.validate_source(streamer)


def test_onStart_accepts_atom_interface():
    """FuturesPandasAggregator accepts arbitrary data and named source."""
    streamer = HistoricalDataStreamer(
        contract=ibi.Future("NQ", exchange="CME"),
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    aggregator = make_aggregator()
    data = []

    with (
        patch.object(aggregator, "sync_with_streamer") as sync,
        patch.object(Atom, "onStart", autospec=True) as parent_on_start,
    ):
        aggregator.onStart(data, source=streamer)

    sync.assert_called_once_with(streamer)
    parent_on_start.assert_called_once_with(aggregator, data, streamer)


def test_sync_extracts_which_contract():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    streamer.which_contract = ActiveNext.NEXT
    aggregator = make_aggregator()
    aggregator.sync_with_streamer(streamer)
    assert aggregator.which_contract is ActiveNext.NEXT


def test_sync_extracts_blueprint():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    streamer.which_contract = ActiveNext.NEXT
    aggregator = make_aggregator()
    aggregator.sync_with_streamer(streamer)
    assert aggregator._contract_blueprint is blueprint


def test_FuturesPandasAggregator_has_the_same_contract_as_Streamer(
    registry_runtime,
):
    blueprint = ibi.Future("ES", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    streamer.which_contract = ActiveNext.NEXT
    # even though which_contract is mistakenly set differently on the aggregator
    aggregator = make_aggregator()
    aggregator.which_contract = ActiveNext.ACTIVE
    aggregator._contract_blueprint = blueprint
    # contracts are different before syncing
    assert aggregator.contract is not streamer.contract
    # ...it should get synced
    aggregator.sync_with_streamer(streamer)
    # ...into the same contract as Streamer
    assert aggregator.contract is streamer.contract


def test_params_extracted_from_streamer():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    aggregator = make_aggregator()
    aggregator.sync_with_streamer(streamer)
    assert isinstance(aggregator._streamer_params.get("contract"), ibi.Future)
    assert aggregator._streamer_params.get("durationStr") == "1D"
    assert aggregator._streamer_params.get("barSizeSetting") == "30 secs"
    assert aggregator._streamer_params.get("whatToShow") == "TRADES"


def test_sync_rejects_non_future_contract():
    """The futures-only contract requirement is checked once during startup."""

    streamer = HistoricalDataStreamer(
        contract=ibi.Stock("AAPL", exchange="SMART", currency="USD"),
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )

    with pytest.raises(TypeError, match="requires a futures Contract"):
        make_aggregator().sync_with_streamer(streamer)


def test_sync_accepts_continuous_future_after_contract_resolution(
    atom_runtime_factory,
):
    """A continuous-future blueprint resolves to a concrete Future at startup."""

    registry = ContractRegistry(today=datetime(2025, 12, 12))
    atom_runtime_factory(contract_registry=registry)
    streamer = HistoricalDataStreamer(
        contract=ibi.ContFuture("ES", exchange="CME"),
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    registry.reset_data([details[0]])

    aggregator = make_aggregator()
    aggregator.sync_with_streamer(streamer)

    assert isinstance(aggregator.contract, ibi.Future)


def test_injected_datastore_is_used_without_runtime_discovery(atom_runtime):
    """An aggregator should retain its injected datastore unchanged."""

    store = Mock(spec=AsyncDataStore)

    aggregator = FuturesPandasAggregator(datastore=store, save_frequency=0)

    assert aggregator.datastore is store
    assert aggregator.store is store
    atom_runtime.market_data_store_factory.assert_not_called()


def test_runtime_default_datastore_is_resolved_from_streamer(
    atom_runtime_factory,
):
    """The default store should use the connected streamer's request identity."""

    store = Mock(spec=AsyncDataStore)
    factory = Mock(return_value=store)
    atom_runtime_factory(market_data_store_factory=factory)
    streamer = HistoricalDataStreamer(
        contract=ibi.Future("NQ", exchange="CME"),
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
        useRTH=True,
    )
    aggregator = FuturesPandasAggregator(save_frequency=0)

    with patch.object(Atom, "onStart", autospec=True):
        aggregator.onStart({}, source=streamer)

    assert aggregator.datastore is True
    assert aggregator.store is store
    factory.assert_called_once_with(
        bar_size_setting="30 secs",
        what_to_show="TRADES",
        use_rth=True,
    )


@pytest.mark.parametrize("datastore", [False, None])
def test_disabled_datastore_is_rejected(datastore):
    """DataFrame aggregation requires stored history."""

    with pytest.raises(TypeError, match="True or an AsyncDataStore"):
        FuturesPandasAggregator(datastore=datastore)  # type: ignore[arg-type]


def test_runtime_default_store_is_unavailable_before_startup():
    """Default resolution requires request identity from the source streamer."""

    aggregator = FuturesPandasAggregator(save_frequency=0)

    with pytest.raises(RuntimeError, match="was not initialized"):
        _ = aggregator.store


def test_save_frequency_defaults_to_900_seconds():
    """Save cadence should be ordinary constructor policy."""

    aggregator = FuturesPandasAggregator(datastore=Mock(spec=AsyncDataStore))

    assert aggregator.save_frequency == 900


@pytest.mark.parametrize("save_frequency", [True, 1.5, "900"])
def test_save_frequency_rejects_non_integer_values(save_frequency):
    with pytest.raises(TypeError, match="save_frequency must be an int"):
        FuturesPandasAggregator(
            datastore=Mock(spec=AsyncDataStore),
            save_frequency=save_frequency,  # type: ignore[arg-type]
        )


def test_save_frequency_rejects_negative_values():
    with pytest.raises(ValueError, match="save_frequency must not be negative"):
        FuturesPandasAggregator(datastore=Mock(spec=AsyncDataStore), save_frequency=-1)


@pytest.mark.asyncio
async def test_repeated_start_does_not_create_duplicate_timer_setter():
    streamer = HistoricalDataStreamer(
        contract=ibi.Future("NQ", exchange="CME"),
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
    )
    aggregator = FuturesPandasAggregator(
        datastore=Mock(spec=AsyncDataStore), save_frequency=900
    )

    with patch.object(aggregator, "set_timer", new_callable=AsyncMock) as set_timer:
        aggregator.onStart({}, source=streamer)
        timer_task = aggregator._timer_task
        aggregator.onStart({}, source=streamer)

        assert aggregator._timer_task is timer_task
        assert timer_task is not None
        await timer_task

    set_timer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_save_data_awaits_datastore_append(atom_runtime):
    """Saving current data waits for append completion."""

    store = Mock(spec=AsyncDataStore)
    aggregator = FuturesPandasAggregator(datastore=store, save_frequency=0)
    aggregator.contract = ibi.Future(symbol="NQ", exchange="CME")
    aggregator._df = pd.DataFrame({"close": [1.0]})

    await aggregator.save_data()

    store.append.assert_awaited_once_with(aggregator.contract, aggregator._df)


@pytest.mark.asyncio
async def test_save_data_skips_overlapping_save(atom_runtime):
    store = Mock(spec=AsyncDataStore)
    aggregator = FuturesPandasAggregator(datastore=store, save_frequency=0)
    aggregator.contract = ibi.Future(symbol="NQ", exchange="CME")
    aggregator._df = pd.DataFrame({"close": [1.0]})
    aggregator._save_in_progress = True

    await aggregator.save_data()

    store.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_data_logs_best_effort_failure(atom_runtime, caplog):
    store = Mock(spec=AsyncDataStore)
    store.append.side_effect = RuntimeError("unavailable")
    aggregator = FuturesPandasAggregator(datastore=store, save_frequency=0)
    aggregator.contract = ibi.Future(symbol="NQ", exchange="CME")
    aggregator._df = pd.DataFrame({"close": [1.0]})

    with caplog.at_level(logging.ERROR):
        await aggregator.save_data()

    assert "failed to save aggregated data" in caplog.text
    assert aggregator._save_in_progress is False


@pytest.mark.asyncio
async def test_backfill_write_awaits_datastore_completion(atom_runtime):
    """A broker backfill waits for its datastore write."""

    store = Mock(spec=AsyncDataStore)
    store.read.return_value = None
    aggregator = FuturesPandasAggregator(datastore=store, save_frequency=0)
    aggregator.contract = ibi.Future(symbol="NQ", exchange="CME")
    back_contract = ibi.Future(symbol="ES", exchange="CME", localSymbol="ESZ5")
    bars = [{"date": datetime(2025, 12, 1), "close": 1.0}]
    aggregator._pull_history_from_broker = AsyncMock(return_value=bars)

    await aggregator._acquire_data_for_contract(
        back_contract,
        datetime(2025, 12, 1),
        datetime(2025, 12, 2),
    )

    store.write.assert_awaited_once()


def test_expiry_from_contract():
    gc = ibi.Future(
        conId=372852975,
        symbol="GC",
        lastTradeDateOrContractMonth="20250626",
        multiplier="100",
        exchange="COMEX",
        currency="USD",
        localSymbol="GCM5",
        tradingClass="GC",
    )
    assert FuturesPandasAggregator.expiry_from_contract(gc) == datetime(2025, 6, 26)


def test_back_contracts(registry_runtime):
    """
    Test it includes only contracts with expiry date earlier than
    current contract and contracts are sorted backwards by expiry
    date.
    """
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future("ES", exchange="CME")
    contracts = [details.contract for details in details[0]]
    previous_contracts = sorted(
        [
            contract
            for contract in contracts
            if FuturesPandasAggregator.expiry_from_contract(contract)
            <= FuturesPandasAggregator.expiry_from_contract(aggregator.contract)
        ],
        key=lambda x: FuturesPandasAggregator.expiry_from_contract(x),
        reverse=True,
    )
    assert list(aggregator._back_contracts()) == previous_contracts


def test_back_contracts_iterable_starting_with_current_contract(
    registry_runtime,
):
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future("ES", exchange="CME")
    for contract in aggregator._back_contracts():
        assert contract == aggregator.contract
        break


def test_back_contracts_iterable_going_backward(registry_runtime):
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future("ES", exchange="CME")
    previuos_contract = None
    for contract in aggregator._back_contracts():
        if previuos_contract is None:
            continue
        assert datetime.strptime(
            contract.lastTradeDateOrContractMonth, "%Y%m%d"
        ) < datetime.strptime(previuos_contract.lastTradeDateOrContractMonth, "%Y%m%d")
        previuos_contract = contract


def test_df_combined_correctly_in_append_data_non_overlapping():
    sample_df = pd.DataFrame(sample_barDataList).set_index("date")
    first_batch, last_batch = sample_df[:-5], sample_df[-5:]

    aggregator = make_aggregator()
    aggregator._df = first_batch

    with patch.object(aggregator, "save_data", new_callable=Mock):
        aggregator.append_data(last_batch)

    pd.testing.assert_frame_equal(aggregator._df, sample_df)


def test_df_combined_correctly_in_append_data_overlapping():
    sample_df = pd.DataFrame(sample_barDataList).set_index("date")
    first_batch, last_batch = sample_df[:-5], sample_df[-10:]

    aggregator = make_aggregator()
    aggregator._df = first_batch

    with patch.object(aggregator, "save_data", new_callable=Mock):
        aggregator.append_data(last_batch)

    pd.testing.assert_frame_equal(aggregator._df, sample_df)


@pytest.mark.asyncio
async def test_data_queued():
    """
    Test that rapidly emitted data will get processed in the right order.
    """
    first_batch = sample_barDataList[:-3]
    second_batch = sample_barDataList[:-2]
    third_batch = sample_barDataList[:-1]
    last_batch = sample_barDataList[:]

    class SourceAtom(Atom):
        pass

    class OutputAtom(Atom):
        def onData(self, data, *args):
            print(f"data on output: {len(data) if data else data}")

    aggregator = make_aggregator()
    source = SourceAtom()
    aggregator.contract = source.contract = ibi.Future(symbol="ES", exchange="CME")
    source += aggregator
    aggregator += OutputAtom()

    with patch.object(
        aggregator._queue, "processing_func", new_callable=AsyncMock
    ) as mock_process_data:
        source.dataEvent.emit(first_batch)
        source.dataEvent.emit(second_batch)
        source.dataEvent.emit(third_batch)
        source.dataEvent.emit(last_batch)

        await wait_for_condition(lambda: aggregator._queue._queue.empty())
        await asyncio.sleep(0.1)

        assert mock_process_data.call_count == 4
        for call, batch in zip(
            mock_process_data.call_args_list,
            [first_batch, second_batch, third_batch, last_batch],
        ):
            args, kwargs = call
            expected = args[0]
            assert expected == batch


@pytest.mark.parametrize(
    "conId,localSymbol,return_value",
    [
        (533620665, "ESH4", None),
        (551601561, "ESM4", None),
        (568550526, "ESU4", None),
        (495512557, "ESZ4", None),
        (603558932, "ESH5", None),
        (620731015, "ESM5", None),
        (637533641, "ESU5", None),
        (
            495512563,
            "ESZ5",
            (datetime(2025, 12, 10), datetime(2025, 12, 12)),
        ),
        # this is a contract with start date in the future
        (649180695, "ESH6", None),
    ],
)
def test_compute_date_range(registry_runtime, conId, localSymbol, return_value):
    """
    Here are the date ranges from `aggregator.contract_selector.date_ranges`.

    <FutureSelector active_contract=ESZ5 next_contract=ESH6>

    533620665 ESH4 (Timestamp('2023-12-19 00:00:00'), Timestamp('2024-03-12 00:00:00'))
    551601561 ESM4 (Timestamp('2024-03-12 00:00:00'), Timestamp('2024-06-17 00:00:00'))
    568550526 ESU4 (Timestamp('2024-06-17 00:00:00'), Timestamp('2024-09-17 00:00:00'))
    495512557 ESZ4 (Timestamp('2024-09-17 00:00:00'), Timestamp('2024-12-17 00:00:00'))
    603558932 ESH5 (Timestamp('2024-12-17 00:00:00'), Timestamp('2025-03-18 00:00:00'))
    620731015 ESM5 (Timestamp('2025-03-18 00:00:00'), Timestamp('2025-06-16 00:00:00'))
    637533641 ESU5 (Timestamp('2025-06-16 00:00:00'), Timestamp('2025-09-16 00:00:00'))
    495512563 ESZ5 (Timestamp('2025-09-16 00:00:00'), Timestamp('2025-12-16 00:00:00'))
    -- stop here for back contracts --
    649180695 ESH6 (Timestamp('2025-12-16 00:00:00'), Timestamp('2026-03-17 00:00:00'))
    ...

    today is 12/12/2025, required timedelta is 2 D, so we shouldn't go back before
    10/12/2025

    basically, we're requesting data between 10-12/12/2025

    """
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future("ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "2 D",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    # required timedelta will return 2D
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2025, 12, 12),
    ):
        date_range_or_none = aggregator._compute_date_range(
            ibi.Future(conId=conId, localSymbol=localSymbol)
        )
        assert date_range_or_none == return_value


@pytest.mark.parametrize(
    "conId,localSymbol,return_value",
    [
        (533620665, "ESH4", None),
        (551601561, "ESM4", None),
        (568550526, "ESU4", None),
        (495512557, "ESZ4", None),
        (603558932, "ESH5", None),
        (620731015, "ESM5", None),
        # start: 3 months back from now, end: contract end
        (
            637533641,
            "ESU5",
            (
                datetime(2025, 9, 13),
                datetime(2025, 9, 16),
            ),
        ),
        (
            495512563,
            "ESZ5",
            (
                datetime(2025, 9, 16),
                datetime(2025, 12, 12),
            ),
        ),
        # this is a contract with start date in the future
        (649180695, "ESH6", None),
    ],
)
def test_compute_date_range_longer_period(
    registry_runtime, conId, localSymbol, return_value
):
    """
    Here are the date ranges from `aggregator.contract_selector.date_ranges`.

    <FutureSelector active_contract=ESZ5 next_contract=ESH6>

    533620665 ESH4 (Timestamp('2023-12-19 00:00:00'), Timestamp('2024-03-12 00:00:00'))
    551601561 ESM4 (Timestamp('2024-03-12 00:00:00'), Timestamp('2024-06-17 00:00:00'))
    568550526 ESU4 (Timestamp('2024-06-17 00:00:00'), Timestamp('2024-09-17 00:00:00'))
    495512557 ESZ4 (Timestamp('2024-09-17 00:00:00'), Timestamp('2024-12-17 00:00:00'))
    603558932 ESH5 (Timestamp('2024-12-17 00:00:00'), Timestamp('2025-03-18 00:00:00'))
    620731015 ESM5 (Timestamp('2025-03-18 00:00:00'), Timestamp('2025-06-16 00:00:00'))
    637533641 ESU5 (Timestamp('2025-06-16 00:00:00'), Timestamp('2025-09-16 00:00:00'))
    495512563 ESZ5 (Timestamp('2025-09-16 00:00:00'), Timestamp('2025-12-16 00:00:00'))
    -- stop here for back contracts --
    649180695 ESH6 (Timestamp('2025-12-16 00:00:00'), Timestamp('2026-03-17 00:00:00'))
    ...

    today is 12/12/2025, required timedelta is 2 D, so we shouldn't go back before
    10/12/2025

    basically, we're requesting data between  3 months back from today

    """
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future("ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "3 M",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    # required timedelta will return 2D
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2025, 12, 12),
    ):
        date_range_or_none = aggregator._compute_date_range(
            ibi.Future(conId=conId, localSymbol=localSymbol)
        )
        assert date_range_or_none == return_value


def test_compute_date_range_uses_next_contract_ranges(registry_runtime):
    aggregator = make_aggregator()
    aggregator.which_contract = ActiveNext.NEXT
    aggregator.contract = ibi.Future("ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "2 D",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    contract = aggregator.contract

    assert contract.localSymbol == "ESH6"
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2025, 12, 12),
    ):
        assert aggregator._compute_date_range(contract) == (
            datetime(2025, 12, 11),
            datetime(2025, 12, 12),
        )


def test_aggregator_offset_by_durationStr_given_as_str(registry_runtime):
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2026, 2, 20),
    ):
        aggregator = make_aggregator()
        aggregator.contract = ibi.Future("ES", exchange="CME")
        aggregator._streamer_params = {
            "durationStr": "2 D",
            "barSizeSetting": "30 secs",
            "whatToShow": "TRADES",
            "useRTH": False,
        }
        # does it even work at all?
        assert isinstance(aggregator.offset_by_durationStr(), datetime)
        assert aggregator.offset_by_durationStr() == datetime(2026, 2, 18)


def test_aggregator_offset_by_durationStr_given_as_str_including_weekend(
    registry_runtime,
):
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2026, 4, 1),
    ):
        aggregator = make_aggregator()
        aggregator.contract = ibi.Future("ES", exchange="CME")
        aggregator._streamer_params = {
            "durationStr": "5 D",
            "barSizeSetting": "30 secs",
            "whatToShow": "TRADES",
            "useRTH": False,
        }
        # does it even work at all?
        assert isinstance(aggregator.offset_by_durationStr(), datetime)
        assert aggregator.offset_by_durationStr() == datetime(2026, 3, 25)


def test_aggregator_offset_by_durationStr_given_as_int(registry_runtime):
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2026, 2, 20),
    ):
        aggregator = make_aggregator()
        aggregator.contract = ibi.Future("ES", exchange="CME")
        aggregator._streamer_params = {
            "durationStr": 1000,
            "barSizeSetting": "30 secs",
            "whatToShow": "TRADES",
            "useRTH": False,
        }
        # does it even work at all?
        assert isinstance(aggregator.offset_by_durationStr(), datetime)
        # 1000 datapoints / 120 * 3600 = timedelta in seconds
        assert aggregator.offset_by_durationStr() == datetime(2026, 2, 20) - timedelta(
            seconds=30000
        )


def test_aggregator_offset_by_durationStr_given_as_int_longer_than_one_day(
    registry_runtime,
):
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2026, 2, 20),
    ):
        aggregator = make_aggregator()
        aggregator.contract = ibi.Future("ES", exchange="CME")
        aggregator._streamer_params = {
            "durationStr": 3000,
            "barSizeSetting": "30 secs",
            "whatToShow": "TRADES",
            "useRTH": False,
        }
        # does it even work at all?
        assert isinstance(aggregator.offset_by_durationStr(), datetime)
        # 3000 datapoints / 120 / 23 * 3600 * 24 = timedelta in seconds
        assert aggregator.offset_by_durationStr() == datetime(2026, 2, 20) - timedelta(
            seconds=round(3000 / 120 / 23 * 3600 * 24)
        )


def test_aggregator_offset_by_durationStr_given_as_int_including_weekend(
    registry_runtime,
):
    with patch(
        "haymaker.components.dataframe_aggregators.utc_now_naive",
        return_value=datetime(2026, 4, 1),
    ):
        aggregator = make_aggregator()
        aggregator.contract = ibi.Future("ES", exchange="CME")
        aggregator._streamer_params = {
            "durationStr": 10000,
            "barSizeSetting": "30 secs",
            "whatToShow": "TRADES",
            "useRTH": False,
        }
        # does it even work at all?
        assert isinstance(aggregator.offset_by_durationStr(), datetime)
        # 10000 datapoints / 120 / 23 * 3600 * 24 = timedelta in seconds
        delta = timedelta(seconds=313_043.47826087)
        assert aggregator.offset_by_durationStr() == datetime(
            2026, 4, 1
        ) - delta.days * custom_bday - timedelta(seconds=delta.seconds)


def test_aggregator_session_length(registry_runtime):
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    assert aggregator.session_length == timedelta(hours=23)


def test_aggregator_datapoints_from_str(registry_runtime):
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "5 D",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    assert aggregator.datapoints == 4 * 23 * 120


def test_aggregator_datapoints_from_int(registry_runtime):
    aggregator = make_aggregator()
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": 120,
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    assert aggregator.datapoints == 120


@pytest.mark.asyncio
async def test_pull_history_from_broker(registry_runtime):
    """Test if using streamer parameters."""

    input_contract = ibi.Future(symbol="ES", exchange="CME")
    bar_size_setting = "30 secs"
    what_to_show = "TRADES"
    useRTH = True

    streamer = HistoricalDataStreamer(
        input_contract,
        120,
        bar_size_setting,
        what_to_show,
        useRTH,
    )
    aggregator = make_aggregator()
    aggregator.contract = input_contract

    streamer += aggregator
    streamer.onStart({})

    es = ibi.Future(
        conId=495512557,
        symbol="ES",
        lastTradeDateOrContractMonth="20241220",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESZ4",
        tradingClass="ES",
    )
    aggregator.ib.reqHistoricalDataAsync = AsyncMock()

    await aggregator._pull_history_from_broker(es, datetime(2025, 12, 10, 9, 0))

    aggregator.ib.reqHistoricalDataAsync.assert_called_once()

    call_kwargs = aggregator.ib.reqHistoricalDataAsync.call_args.kwargs
    # these are all the same as on the streamer that aggregator is connected to
    assert call_kwargs["contract"] == es
    assert call_kwargs["whatToShow"] == what_to_show
    assert call_kwargs["barSizeSetting"] == bar_size_setting
    assert call_kwargs["useRTH"] == useRTH
    assert call_kwargs["durationStr"] == "3600 S"


def volume_frame(volumes: list[int]) -> pd.DataFrame:
    """Return a minimal date-indexed OHLCV frame for component tests."""

    index = pd.date_range("2026-01-01", periods=len(volumes), freq="min", name="date")
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": volumes,
        },
        index=index,
    )


@pytest.mark.parametrize("volumes", [[], [40]])
def test_volume_grouper_accepts_initial_frame_without_completed_group(volumes):
    """Short initial histories establish state without raising or emitting."""

    grouper = VolumeGrouper(100)
    emissions = []
    grouper.dataEvent += emissions.append

    grouper.onData(volume_frame(volumes))

    assert emissions == []


@pytest.mark.parametrize("label", ["left", "right"])
def test_volume_grouper_emits_first_group_when_it_later_completes(label):
    """An initially incomplete group should emit as soon as it reaches target."""

    grouper = VolumeGrouper(100, label=label)
    emissions = []
    grouper.dataEvent += emissions.append
    grouper.onData(volume_frame([40]))

    grouper.onData(volume_frame([40, 60]))

    assert len(emissions) == 1
    assert len(emissions[0]) == 1
    assert emissions[0].iloc[-1]["volume"] == 100


@pytest.mark.parametrize("volume", [0, -1])
def test_volume_grouper_rejects_non_positive_target(volume):
    """A volume threshold must be positive as documented."""

    with pytest.raises(ValueError, match="must be positive"):
        VolumeGrouper(volume)


@pytest.mark.parametrize("volume", [True, 100.0])
def test_volume_grouper_rejects_non_integer_target(volume):
    """Boolean and floating-point thresholds are not valid integer volumes."""

    with pytest.raises(TypeError, match="must be an int"):
        VolumeGrouper(volume)
