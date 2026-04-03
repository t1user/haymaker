import asyncio
from unittest.mock import AsyncMock, patch

import ib_insync as ibi
import pytest
from helpers import wait_for_condition
from sample_barDataList import sample_barDataList

from haymaker.aggregators import BarAggregator, NoFilter, WrongStreamer
from haymaker.base import Atom as BaseAtom
from haymaker.base import Pipe
from haymaker.streamers import HistoricalDataStreamer, MktDataStreamer


def test_onStart_receives_streamer():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )

    aggregator = BarAggregator(NoFilter())

    streamer += aggregator
    with patch.object(aggregator, "sync_with_streamer") as mock_sync_with_streamer:
        streamer.onStart({})
        mock_sync_with_streamer.assert_called_once()
        mock_sync_with_streamer.assert_called_with(streamer)


def test_onStart_works_as_part_of_Pipe():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )
    aggregator = BarAggregator(NoFilter())

    streaming_pipe = Pipe(streamer, aggregator)

    class SourceAtom(BaseAtom):
        pass

    source_atom = SourceAtom()

    source_atom += streaming_pipe

    with patch.object(aggregator, "sync_with_streamer") as mock_sync_with_streamer:
        source_atom.onStart({})
        mock_sync_with_streamer.assert_called_once()
        mock_sync_with_streamer.assert_called_with(streamer)


def test_HistoricalDataStreamerAccepted():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract=blueprint,
        durationStr="1D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
    )

    aggregator = BarAggregator(NoFilter())
    # test if no error raised
    assert aggregator.sync_with_streamer(streamer) is None


def test_wrong_streamer_fails():
    blueprint = ibi.Future("NQ", exchange="CME")
    streamer = MktDataStreamer(contract=blueprint, tickList="212")
    aggregator = BarAggregator(NoFilter())
    with pytest.raises(WrongStreamer):
        aggregator.sync_with_streamer(streamer)


@pytest.fixture
def source_aggregator_output():
    """
    It's a setup that allows to emit either `startEvent` or
    `dataEvent` on `source` and then check whatever Aggregator emitted
    on `output`.  Output properties: `onStart_data` and `onData_data`
    have whatever `onStart` and `onData` methods recieved.
    """

    class SourceAtom(BaseAtom):
        pass

    class OutputAtom(BaseAtom):

        def __init__(self):
            self.reset_data()
            super().__init__()

        def onStart(self, data, *args):
            self.onStart_data = data
            self.onStart_counter += 1

        def onData(self, data, *args):
            self.onData_data = data
            self.onData_counter += 1

        def reset_data(self):
            self.onStart_data = None
            self.onData_data = None
            self.onStart_counter = 0
            self.onData_counter = 0

    # using NoFilter ensures that we return the same data as supplied
    # actual filter would be a layer of complication here
    aggregator = BarAggregator(NoFilter(), future_adjust_type="add")
    source = SourceAtom()
    output = OutputAtom()

    source.pipe(aggregator, output)
    return source, aggregator, output


def test_BarAggregator_passes_onStart_signal(source_aggregator_output):
    _, aggregator, output = source_aggregator_output

    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="ES", exchange="CME"), 1000, "1 min", "TRADES"
    )
    streamer += aggregator

    streamer.onStart({"test_data": True})

    assert output.onStart_data.get("test_data")


@pytest.mark.asyncio
async def test_BarAggregator__filter_empty(
    source_aggregator_output,
):
    source, aggregator, output = source_aggregator_output
    source.dataEvent.emit(sample_barDataList)

    await wait_for_condition(lambda: output.onData_data)
    assert len(output.onData_data) == len(sample_barDataList)
    for left, right in zip(output.onData_data, sample_barDataList):
        assert left == right
    assert output.onData_counter == 1


@pytest.mark.asyncio
async def test_BarAggregator__filter_not_empty_one_extra_bar(source_aggregator_output):
    source, aggregator, output = source_aggregator_output

    first_batch = sample_barDataList[:-1]

    source.dataEvent.emit(first_batch)
    await wait_for_condition(lambda: output.onData_data)

    # reset after first emit
    output.reset_data()

    # same as previous plus one extra bar
    # so only the last bar should end up in the filter
    source.dataEvent.emit(sample_barDataList)

    await wait_for_condition(lambda: output.onData_data)

    assert output.onData_counter == 1
    assert len(output.onData_data) == len(sample_barDataList)
    for left, right in zip(output.onData_data, sample_barDataList):
        assert left == right


@pytest.mark.asyncio
async def test_BarAggregator__filter_not_empty_batch_of_bars_after_restart(
    source_aggregator_output,
):
    source, aggregator, output = source_aggregator_output
    first_batch, _ = sample_barDataList[:-5], sample_barDataList[-5:]
    source.dataEvent.emit(first_batch)
    await wait_for_condition(lambda: output.onData_data)

    # reset after first emit
    output.reset_data()
    source.dataEvent.emit(sample_barDataList)
    await wait_for_condition(lambda: output.onData_data)

    assert len(output.onData_data) == len(sample_barDataList)
    for left, right in zip(output.onData_data, sample_barDataList):
        assert left == right
    assert output.onData_counter == 1


@pytest.mark.asyncio
async def test_BarAggregator__of_bars_after_restart__adjustment_required(
    source_aggregator_output,
):
    source, aggregator, output = source_aggregator_output

    adj = 5
    adjusted = ibi.BarDataList(
        [
            ibi.BarData(
                bar.date,
                bar.open + adj,
                bar.high + adj,
                bar.low + adj,
                bar.close + adj,
                bar.volume,
                bar.average + adj,
                bar.barCount,
            )
            for bar in sample_barDataList
        ]
    )

    first_batch, _ = sample_barDataList[:-5], sample_barDataList[-5:]
    # _, last_batch = adjusted[:-5], adjusted[-5:]

    source.dataEvent.emit(first_batch)
    await wait_for_condition(lambda: output.onData_data)

    # reset after first emit
    output.reset_data()

    # this should trigger adjustment
    aggregator.onContractChanged(ibi.Future(conId=5), ibi.Future(conId=6))

    adjusted_no_last_point = adjusted[:-1]
    source.dataEvent.emit(adjusted_no_last_point)
    await wait_for_condition(lambda: output.onData_data)

    assert len(output.onData_data) == len(adjusted_no_last_point)
    for left, right in zip(output.onData_data, adjusted_no_last_point):
        assert left == right
    assert output.onData_counter == 1

    # test that subsequent emits will not adjust
    output.reset_data()
    source.dataEvent.emit(adjusted)
    await wait_for_condition(lambda: output.onData_data)

    assert len(output.onData_data) == len(adjusted)
    for left, right in zip(output.onData_data, adjusted):
        assert left == right
    assert output.onData_counter == 1


@pytest.mark.asyncio
async def test_BarAggregator__of_bars_after_restart__adjustment_required_multiplication(
    source_aggregator_output,
):
    source, aggregator, output = source_aggregator_output

    aggregator.future_adjust_type = "mul"

    adj = 1.01
    adjusted = ibi.BarDataList(
        [
            ibi.BarData(
                bar.date,
                bar.open * adj,
                bar.high * adj,
                bar.low * adj,
                bar.close * adj,
                bar.volume,
                bar.average * adj,
                bar.barCount,
            )
            for bar in sample_barDataList
        ]
    )

    first_batch, _ = sample_barDataList[:-5], sample_barDataList[-5:]
    # _, last_batch = adjusted[:-5], adjusted[-5:]

    source.dataEvent.emit(first_batch)
    await wait_for_condition(lambda: output.onData_data)

    # reset after first emit
    output.reset_data()

    # this should trigger adjustment
    aggregator.onContractChanged(ibi.Future(conId=5), ibi.Future(conId=6))

    adjusted_no_last_point = adjusted[:-1]
    source.dataEvent.emit(adjusted_no_last_point)
    await wait_for_condition(lambda: output.onData_data)

    assert len(output.onData_data) == len(adjusted_no_last_point)
    for left, right in zip(output.onData_data, adjusted_no_last_point):
        assert left == right
    assert output.onData_counter == 1

    # test that subsequent emits will not adjust
    output.reset_data()
    source.dataEvent.emit(adjusted)
    await wait_for_condition(lambda: output.onData_data)

    assert len(output.onData_data) == len(adjusted)
    for left, right in zip(output.onData_data, adjusted):
        assert left == right
    assert output.onData_counter == 1


@pytest.mark.asyncio
async def test_BarAggregator_runs_backfill_if_run_first_time(source_aggregator_output):
    source, aggregator, output = source_aggregator_output

    with patch.object(aggregator, "backfill", new_callable=AsyncMock) as mock_backfill:
        source.dataEvent.emit(sample_barDataList)
        # wait for result then check if backfill was called
        await wait_for_condition(lambda: output.onData_counter)
        mock_backfill.assert_called_once()
        call_args = mock_backfill.call_args[0][0]
        # last datapoint is not being backfilled, it will be added subsequently
        assert list(call_args) == sample_barDataList[:-1]


@pytest.mark.asyncio
async def test_BarAggregator_backfill_not_run_if_not_necessary(
    source_aggregator_output,
):
    source, aggregator, output = source_aggregator_output
    first_batch, last_point = sample_barDataList[:-1], sample_barDataList[-1]
    source.dataEvent.emit(first_batch)
    await wait_for_condition(lambda: output.onData_data)

    with patch.object(aggregator, "backfill", new_callable=AsyncMock) as mock_backfill:
        output.reset_data()
        source.dataEvent.emit(last_point)
        await wait_for_condition(lambda: output.onData_data)
        mock_backfill.assert_not_called()


@pytest.mark.asyncio
async def test_BarAggregator_data_queued_during_backfill(source_aggregator_output):
    source, aggregator, output = source_aggregator_output

    print(f"{len(sample_barDataList)=}, last: {sample_barDataList[-1].date}")
    first_batch = sample_barDataList[:-5]
    with patch.object(
        aggregator, "backfill", wraps=aggregator.backfill
    ) as mock_backfill:

        # this will result with data being emitted before backfill is done
        source.dataEvent.emit(first_batch)
        for i in range(4, 0, -1):
            source.dataEvent.emit(sample_barDataList[:-i])
        source.dataEvent.emit(sample_barDataList)

        # 6 datapoints fed into aggregator
        await wait_for_condition(lambda: output.onData_counter == 6, timeout=1)
        print(output.onData_counter)
        # make sure we have correct data
        for should_be, is_ in zip(sample_barDataList, output.onData_data):
            assert should_be == is_

        # make sure we didn't backfill more than once
        assert mock_backfill.call_count == 1

        # make sure we backfilled with correct data
        # we don't backfill last bar
        mock_backfill.assert_called_once_with(first_batch[:-1])

        # make sure no additional emits from aggregator happened
        assert output.onData_counter == 6


@pytest.mark.asyncio
async def test_BarAggregator_no_duplicate_data_processed(source_aggregator_output):
    """
    If the same data is being re-emitted, we shouldn't send the same
    datapoint to the filter more than once.
    """
    source, aggregator, output = source_aggregator_output

    source.dataEvent.emit(sample_barDataList)
    source.dataEvent.emit(sample_barDataList)
    source.dataEvent.emit(sample_barDataList)

    # make sure all 3 emits were processed
    await wait_for_condition(lambda: output.onData_counter, timeout=1)
    await asyncio.sleep(0)

    assert len(output.onData_data) == len(sample_barDataList)
    assert len(aggregator.filter.bars) == len(sample_barDataList)

    for left, right in zip(output.onData_data, sample_barDataList):
        assert left == right

    # double check that the filter has what it's supposed to have
    for left, right in zip(aggregator.filter.bars, sample_barDataList):
        assert left == right


# =======================================================================
# LLM generated tests from here on
# =======================================================================


@pytest.mark.asyncio
async def test_BarAggregator_queued_items_strict_fifo_order(source_aggregator_output):
    """
    Test that items queued during backfill are processed in strict FIFO order
    """
    source, aggregator, output = source_aggregator_output

    # Track the order of _process calls
    process_order = []
    original_process = aggregator._process

    async def tracked_process(data_, *args, **kwargs):
        process_order.append(data_[-1].date)  # Record last bar timestamp
        return await original_process(data_, *args, **kwargs)

    # Create batches with distinct non-overlapping timestamps
    batch1 = sample_barDataList[:3]
    batch2 = sample_barDataList[3:6]
    batch3 = sample_barDataList[6:9]

    async def slow_backfill(*args, **kwargs):
        # Simulate slow backfill to ensure queuing happens
        await asyncio.sleep(0.05)
        return await original_backfill(*args, **kwargs)

    original_backfill = aggregator.backfill

    with patch.object(aggregator, "backfill", side_effect=slow_backfill):
        with patch.object(aggregator, "_process", side_effect=tracked_process):
            # Emit batches in rapid succession
            source.dataEvent.emit(batch1)
            await asyncio.sleep(0.01)  # Let backfill start
            source.dataEvent.emit(batch2)
            source.dataEvent.emit(batch3)

            # Wait for all to process
            await wait_for_condition(lambda: output.onData_counter == 3, timeout=1)

    # Verify processing happened in order: batch1, batch2, batch3
    assert len(process_order) == 3
    assert process_order[0] == batch1[-1].date
    assert process_order[1] == batch2[-1].date
    assert process_order[2] == batch3[-1].date


@pytest.mark.asyncio
async def test_BarAggregator_worker_task_recreation_after_crash(
    source_aggregator_output,
):
    """
    Test that if worker task crashes, it gets recreated on next data
    """
    source, aggregator, output = source_aggregator_output

    crash_count = 0
    original_process = aggregator._process

    async def crashing_process(*args, **kwargs):
        nonlocal crash_count
        crash_count += 1
        if crash_count == 1:
            raise RuntimeError("Simulated crash")
        return await original_process(*args, **kwargs)

    with patch.object(aggregator, "_process", side_effect=crashing_process):
        # First emit should crash
        source.dataEvent.emit(sample_barDataList[:3])
        await asyncio.sleep(0.1)  # Let it crash

        # Verify task is done/failed
        assert aggregator._worker_task is not None
        assert aggregator._worker_task.done()

        # Second emit should recreate task and succeed
        source.dataEvent.emit(sample_barDataList[3:6])
        await wait_for_condition(lambda: output.onData_counter > 0, timeout=1)

        assert crash_count == 2
        assert output.onData_counter == 1  # Only second emit succeeded


@pytest.mark.asyncio
async def test_BarAggregator_concurrent_backfills_blocked(source_aggregator_output):
    """
    Test that if two items are in queue, second waits for first backfill to complete
    """
    source, aggregator, output = source_aggregator_output

    backfill_call_times = []

    async def tracked_backfill(*args, **kwargs):
        backfill_call_times.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.05)  # Simulate slow backfill
        return await original_backfill(*args, **kwargs)

    original_backfill = aggregator.backfill

    with patch.object(aggregator, "backfill", side_effect=tracked_backfill):
        # Emit two batches that both trigger backfill
        source.dataEvent.emit(sample_barDataList[:5])
        source.dataEvent.emit(sample_barDataList)

        await wait_for_condition(lambda: output.onData_counter == 2, timeout=1)

    # Verify backfills happened sequentially, not concurrently
    assert len(backfill_call_times) == 2
    # Second backfill should start after first completes (0.05s delay)
    time_gap = backfill_call_times[1] - backfill_call_times[0]
    assert time_gap >= 0.04  # Allow small timing variance


@pytest.mark.asyncio
async def test_BarAggregator_single_bar_data(source_aggregator_output):
    """
    Test edge case where data_ has only one bar (data[:-1] would be empty)
    """
    source, aggregator, output = source_aggregator_output

    single_bar = ibi.BarDataList([sample_barDataList[0]])
    source.dataEvent.emit(single_bar)

    await wait_for_condition(lambda: output.onData_counter, timeout=1)

    # Should still work - backfills empty list, then emits the single bar
    assert output.onData_counter == 1
    assert len(output.onData_data) == 1
    assert output.onData_data[0] == sample_barDataList[0]


@pytest.mark.asyncio
async def test_BarAggregator_empty_barDataList(source_aggregator_output):
    """
    Test edge case where empty BarDataList is emitted
    """
    source, aggregator, output = source_aggregator_output

    empty_bars = ibi.BarDataList([])
    source.dataEvent.emit(empty_bars)

    # Should handle gracefully and not crash
    await asyncio.sleep(0.1)

    # Worker task should still be running (not crashed)
    assert aggregator._worker_task is not None
    assert not aggregator._worker_task.done()

    # No data should have been emitted
    assert output.onData_counter == 0


@pytest.mark.asyncio
async def test_BarAggregator_out_of_order_bars_rejected(source_aggregator_output):
    """
    Test that bars arriving out of chronological order are rejected
    """
    source, aggregator, output = source_aggregator_output

    # Process normal sequence first
    source.dataEvent.emit(sample_barDataList)
    await wait_for_condition(lambda: output.onData_counter, timeout=1)

    initial_filter_length = len(aggregator.filter.bars)
    output.reset_data()

    # Try to emit an older bar
    old_bars = sample_barDataList[:-3]
    source.dataEvent.emit(old_bars)
    await asyncio.sleep(0)

    # Should not emit or add to filter
    assert output.onData_counter == 0
    assert output.onData_data is None
    assert len(aggregator.filter.bars) == initial_filter_length


@pytest.mark.asyncio
async def test_BarAggregator_duplicate_bar_rejected(source_aggregator_output):
    """
    Test that duplicate bars (same timestamp) are not processed twice
    """
    source, aggregator, output = source_aggregator_output

    source.dataEvent.emit(sample_barDataList)
    await wait_for_condition(lambda: output.onData_counter, timeout=1)

    initial_filter_length = len(aggregator.filter.bars)
    last_bar_date = aggregator._last_data_point
    output.reset_data()

    # Emit the same bars again
    duplicate = sample_barDataList
    source.dataEvent.emit(duplicate)
    await asyncio.sleep(0)

    # Should not emit or add to filter
    assert output.onData_counter == 0
    assert output.onData_data is None
    assert len(aggregator.filter.bars) == initial_filter_length
    assert aggregator._last_data_point == last_bar_date


@pytest.mark.asyncio
async def test_BarAggregator_backfill_exception_blocks_future_processing(
    source_aggregator_output,
):
    """
    Test that if backfill raises exception during stream.list(),
    event stays cleared and blocks future processing.
    """
    source, aggregator, output = source_aggregator_output

    async def failing_stream_list(*args, **kwargs):
        raise RuntimeError("Stream list failed")

    with patch("eventkit.Sequence.list", side_effect=failing_stream_list):
        source.dataEvent.emit(sample_barDataList)
        await asyncio.sleep(0.1)

    # Worker task should have crashed
    assert aggregator._worker_task.done()

    # Retrieve the exception to suppress warning
    with pytest.raises(RuntimeError, match="Stream list failed"):
        await aggregator._worker_task

    # Event should be cleared (backfill failed)
    assert not aggregator._backfill_event.is_set()

    # No data emitted
    assert output.onData_counter == 0


@pytest.mark.asyncio
async def test_BarAggregator_backfill_sets_event_on_success(source_aggregator_output):
    """
    Test that backfill properly sets event on successful completion
    """
    source, aggregator, output = source_aggregator_output

    # Initially event should be set
    assert aggregator._backfill_event.is_set()

    source.dataEvent.emit(sample_barDataList)

    # During/immediately after emit, event may be cleared
    await asyncio.sleep(0.01)

    # Wait for processing to complete
    await wait_for_condition(lambda: output.onData_counter, timeout=1)

    # After successful backfill, event should be set again
    assert aggregator._backfill_event.is_set()


@pytest.mark.asyncio
async def test_BarAggregator_multiple_bars_same_timestamp_batch(
    source_aggregator_output,
):
    """
    Test handling of a batch where multiple bars have the same
    timestamp (data quality issue)
    """
    source, aggregator, output = source_aggregator_output

    # Create a batch with duplicate timestamps
    duplicate_timestamp_batch = ibi.BarDataList(
        [
            sample_barDataList[0],
            sample_barDataList[1],
            sample_barDataList[1],  # Duplicate
            sample_barDataList[2],
        ]
    )

    source.dataEvent.emit(duplicate_timestamp_batch)
    await wait_for_condition(lambda: output.onData_counter, timeout=1)

    # Should process without crashing
    # Behavior depends on filter implementation (NoFilter passes everything through)
    assert output.onData_counter == 1
    assert len(output.onData_data) == len(duplicate_timestamp_batch)
