import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from contract_details_for_registry import blueprints, details
from helpers import wait_for_condition
from sample_barDataList import sample_barDataList

from haymaker.base import ActiveNext, Atom
from haymaker.contract_registry import ContractRegistry
from haymaker.dfaggregator import DfAggregator, WrongStreamer, custom_bday
from haymaker.streamers import HistoricalDataStreamer, MktDataStreamer


@pytest.fixture(scope="module")
def registry_with_data():
    # this date ensures that active and next contracts are different
    # some tests rely on this
    registry = ContractRegistry(today=datetime(2025, 12, 12))
    for blueprint in blueprints:
        registry.register_blueprint(blueprint)

    registry.reset_data(details)
    return registry


def test_HistoricalDataStreamerAccepted():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )

    aggregator = DfAggregator(save_frequency=0)
    # test if no error raised
    assert aggregator.sync_with_streamer(streamer) is None


def test_wrong_streamer_fails():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = MktDataStreamer(contract=blueprint, tickList="212")
    aggregator = DfAggregator(save_frequency=0)
    with pytest.raises(WrongStreamer):
        aggregator.sync_with_streamer(streamer)


def test_sync_extracts_which_contract():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    streamer.which_contract = ActiveNext.NEXT
    aggregator = DfAggregator(save_frequency=0)
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
    aggregator = DfAggregator(save_frequency=0)
    aggregator.sync_with_streamer(streamer)
    assert aggregator._contract_blueprint is blueprint


def test_DfAggregator_has_the_same_contract_as_Streamer(registry_with_data):
    HistoricalDataStreamer.contract_registry = registry_with_data
    DfAggregator.contract_registry = registry_with_data

    blueprint = ibi.Future("ES", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    streamer.which_contract = ActiveNext.NEXT
    # even though which_contract is mistakenly set as different on DfAggregator
    aggregator = DfAggregator(save_frequency=0)
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
    aggregator = DfAggregator(save_frequency=0)
    aggregator.sync_with_streamer(streamer)
    assert isinstance(aggregator._streamer_params.get("contract"), ibi.Future)
    assert aggregator._streamer_params.get("durationStr") == "1D"
    assert aggregator._streamer_params.get("barSizeSetting") == "30 secs"
    assert aggregator._streamer_params.get("whatToShow") == "TRADES"


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
    assert DfAggregator.expiry_from_contract(gc) == datetime(2025, 6, 26)


def test_back_contracts(registry_with_data):
    """
    Test it includes only contracts with expiry date earlier than
    current contract and contracts are sorted backwards by expiry
    date.
    """
    DfAggregator.contract_registry = registry_with_data
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future("ES", exchange="CME")
    contracts = [details.contract for details in details[0]]
    previous_contracts = sorted(
        [
            contract
            for contract in contracts
            if DfAggregator.expiry_from_contract(contract)
            <= DfAggregator.expiry_from_contract(aggregator.contract)
        ],
        key=lambda x: DfAggregator.expiry_from_contract(x),
        reverse=True,
    )
    assert list(aggregator._back_contracts()) == previous_contracts


def test_back_contracts_iterable_starting_with_current_contract(
    registry_with_data,
):
    DfAggregator.contract_registry = registry_with_data
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future("ES", exchange="CME")
    for contract in aggregator._back_contracts():
        assert contract == aggregator.contract
        break


def test_back_contracts_iterable_going_backward(registry_with_data):
    DfAggregator.contract_registry = registry_with_data
    aggregator = DfAggregator(save_frequency=0)
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

    aggregator = DfAggregator(save_frequency=0)
    aggregator._df = first_batch

    with patch.object(aggregator, "save_data", new_callable=Mock):
        aggregator.append_data(last_batch)

    pd.testing.assert_frame_equal(aggregator._df, sample_df)


def test_df_combined_correctly_in_append_data_overlapping():
    sample_df = pd.DataFrame(sample_barDataList).set_index("date")
    first_batch, last_batch = sample_df[:-5], sample_df[-10:]

    aggregator = DfAggregator(save_frequency=0)
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

    aggregator = DfAggregator(save_frequency=0)
    source = SourceAtom()
    aggregator.contract = source.contract = ibi.Future(symbol="ES", exchange="CME")
    print(aggregator.contract_selector)
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
def test_compute_date_range(registry_with_data, conId, localSymbol, return_value):
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
    DfAggregator.contract_registry = registry_with_data
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future("ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "2 D",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    # required timedelta will return 2D
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        # this is "now" used by the method:
        mock_dt.now.return_value = datetime(2025, 12, 12)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

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
    registry_with_data, conId, localSymbol, return_value
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
    DfAggregator.contract_registry = registry_with_data
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future("ES", exchange="CME")
    aggregator._streamer_params = {
        "durationStr": "3 M",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    # required timedelta will return 2D
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        # this is "now" used by the method:
        mock_dt.now.return_value = datetime(2025, 12, 12)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        date_range_or_none = aggregator._compute_date_range(
            ibi.Future(conId=conId, localSymbol=localSymbol)
        )
        assert date_range_or_none == return_value


def test_aggregator_offset_by_durationStr_given_as_str(registry_with_data):
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 2, 20)
        DfAggregator.contract_registry = registry_with_data
        aggregator = DfAggregator(save_frequency=0)
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
    registry_with_data,
):
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 1)
        DfAggregator.contract_registry = registry_with_data
        aggregator = DfAggregator(save_frequency=0)
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


def test_aggregator_offset_by_durationStr_given_as_int(registry_with_data):
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 2, 20)
        DfAggregator.contract_registry = registry_with_data
        aggregator = DfAggregator(save_frequency=0)
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
    registry_with_data,
):
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 2, 20)
        DfAggregator.contract_registry = registry_with_data
        aggregator = DfAggregator(save_frequency=0)
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
    registry_with_data,
):
    with patch("haymaker.dfaggregator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 1)
        DfAggregator.contract_registry = registry_with_data
        aggregator = DfAggregator(save_frequency=0)
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


def test_aggregator_session_length(registry_with_data):
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    aggregator.contract_registry = registry_with_data
    assert aggregator.session_length == timedelta(hours=23)


def test_aggregator_datapoints_from_str(registry_with_data):
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    aggregator.contract_registry = registry_with_data
    aggregator._streamer_params = {
        "durationStr": "5 D",
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    assert aggregator.datapoints == 4 * 23 * 120


def test_aggregator_datapoints_from_int(registry_with_data):
    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = ibi.Future(symbol="ES", exchange="CME")
    aggregator.contract_registry = registry_with_data
    aggregator._streamer_params = {
        "durationStr": 120,
        "barSizeSetting": "30 secs",
        "whatToShow": "TRADES",
        "useRTH": False,
    }
    assert aggregator.datapoints == 120


@pytest.mark.asyncio
async def test_pull_history_from_broker(Atom, registry_with_data):
    """Test if using streamer parameters."""

    input_contract = ibi.Future(symbol="ES", exchange="CME")
    bar_size_setting = "30 secs"
    what_to_show = "TRADES"
    useRTH = True

    streamer = HistoricalDataStreamer(
        input_contract,
        "2 D",
        bar_size_setting,
        what_to_show,
        useRTH,
    )
    streamer.contract_registry = registry_with_data

    aggregator = DfAggregator(save_frequency=0)
    aggregator.contract = input_contract
    aggregator.contract_registry = registry_with_data

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

    await aggregator._pull_history_from_broker(
        es, datetime(2025, 12, 10, 12, 0), datetime(2025, 12, 10, 9, 0)
    )

    aggregator.ib.reqHistoricalDataAsync.assert_called_once()

    call_kwargs = aggregator.ib.reqHistoricalDataAsync.call_args.kwargs
    # these are all the same as on the streamer that aggregator is connected to
    assert call_kwargs["contract"] == es
    assert call_kwargs["whatToShow"] == what_to_show
    assert call_kwargs["barSizeSetting"] == bar_size_setting
    assert call_kwargs["useRTH"] == useRTH
