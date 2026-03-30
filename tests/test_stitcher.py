import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.contract_selector import FutureSelector
from haymaker.stitcher import FuturesStitcher, NonOverLappingDfsError


def test_offset():
    sticher = FuturesStitcher({Mock(): Mock()})
    assert sticher.offset(5, 7) == 2


def test_offset_negative():
    sticher = FuturesStitcher({Mock(): Mock()})
    assert sticher.offset(5, 4) == -1


def test_offset_mul():
    sticher = FuturesStitcher({Mock(): Mock()}, "mul")
    assert sticher.offset(0.5, 1) == 2


def test_offset_None():
    sticher = FuturesStitcher({Mock(): Mock()}, None)
    assert sticher.offset(5, 7) == 0


def test_sticher_doesnt_accept_wrong_adjust_type():
    with pytest.raises(AssertionError):
        FuturesStitcher([(Mock(), Mock())], "xxx")


def test_adjust():
    sticher = FuturesStitcher({Mock(): Mock()})
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 1)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [2, 3, 4],
                "high": [3, 4, 5],
                "low": [1, 2, 3],
                "close": [4, 5, 6],
                "volume": [234, 235, 236],
                "average": [2, 3, 4],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_negative():
    sticher = FuturesStitcher({Mock(): Mock()})
    df = pd.DataFrame(
        {
            "open": [2, 3, 4],
            "high": [3, 4, 5],
            "low": [1, 2, 3],
            "close": [4, 5, 6],
            "volume": [234, 235, 236],
            "average": [2, 3, 4],
            "barCount": [2, 2, 2],
        }
    )

    adjusted = sticher.adjust(df, -1)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0, 1, 2],
                "close": [3, 4, 5],
                "volume": [234, 235, 236],
                "average": [1, 2, 3],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_mul():
    sticher = FuturesStitcher({Mock(): Mock()}, "mul")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 2)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [2, 4, 6],
                "high": [4, 6, 8],
                "low": [0, 2, 4],
                "close": [6, 8, 10],
                "volume": [234, 235, 236],
                "average": [2, 4, 6],
                "barCount": [2, 2, 2],
            }
        ),
    )


def test_adjust_mul_less_than_zero():
    sticher = FuturesStitcher({Mock(): Mock()}, "mul")
    df = pd.DataFrame(
        {
            "open": [2.0, 4.0, 6.0],
            "high": [4.0, 6.0, 8.0],
            "low": [0, 2.0, 4.0],
            "close": [6.0, 8.0, 10.0],
            "volume": [234, 235, 236],
            "average": [2.0, 4.0, 6.0],
            "barCount": [2.0, 2.0, 2.0],
        }
    )
    adjusted = sticher.adjust(df, 0.5)
    pd.testing.assert_frame_equal(
        adjusted,
        pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0, 1.0, 2.0],
                "close": [3.0, 4.0, 5.0],
                "volume": [234, 235, 236],
                "average": [1.0, 2.0, 3.0],
                "barCount": [2.0, 2.0, 2.0],
            }
        ),
    )


def test_adjust_none():
    sticher = FuturesStitcher({Mock(): Mock()}, None)
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [3, 4, 5],
            "volume": [234, 235, 236],
            "average": [1, 2, 3],
            "barCount": [2, 2, 2],
        }
    )
    adjusted = sticher.adjust(df, 0)
    pd.testing.assert_frame_equal(adjusted, df)


def test_params_passed_from_FuturesStitcher_to_FutureSelector_both_not_None():
    sticher = FuturesStitcher({Mock(): Mock()}, roll_bdays=666, roll_margin_bdays=999)
    assert sticher._selector.roll_bdays == 666
    assert sticher._selector.roll_margin_bdays == 999


def test_params_passed_from_FuturesStitcher_to_FutureSelector_one_not_None():
    sticher = FuturesStitcher({Mock(): Mock()}, roll_bdays=666)
    assert sticher._selector.roll_bdays == 666
    assert sticher._selector.roll_margin_bdays is not None


def test_params_passed_from_FuturesStitcher_to_FutureSelector_other_one_not_None():
    sticher = FuturesStitcher({Mock(): Mock()}, roll_margin_bdays=666)
    assert sticher._selector.roll_margin_bdays == 666
    assert sticher._selector.roll_bdays is not None


def test_params_passed_from_FuturesStitcher_to_FutureSelector_both_None():
    sticher = FuturesStitcher({Mock(): Mock()}, roll_margin_bdays=666)
    assert sticher._selector.roll_bdays is not None
    assert sticher._selector.roll_margin_bdays is not None


@pytest.fixture(scope="module")
def FuturesStitcher_source() -> dict[ibi.Future, pd.DataFrame]:
    """
    Return a dict[ibi.Future, df] typically returned from db.
    """
    contract_df_dict = {}
    test_dir = Path(__file__).parent
    with open(Path(test_dir / "data" / "contract_list.pickle"), "rb") as f:
        contract_list = pickle.load(f)
    for contract in contract_list:
        p = Path(test_dir / "data" / f"df_{contract.localSymbol}.pickle")
        with p.open("rb") as f:
            df = pickle.load(f)
            contract_df_dict[contract] = df
    return contract_df_dict


@pytest.fixture(scope="module")
def add_sticher(FuturesStitcher_source):
    return FuturesStitcher(FuturesStitcher_source, "add", debug=True)


@pytest.fixture(scope="module")
def mul_sticher(FuturesStitcher_source):
    return FuturesStitcher(FuturesStitcher_source, "mul", debug=True)


@pytest.fixture(scope="module")
def none_sticher(FuturesStitcher_source):
    return FuturesStitcher(FuturesStitcher_source, None, debug=True)


def test_FuturesStitcher_all_dfs_used(FuturesStitcher_source, add_sticher):
    source = FuturesStitcher_source
    dfs = add_sticher._dfs
    assert len(source) == len(dfs)


def test_FuturesStitcher_last_df_not_adjusted(FuturesStitcher_source, add_sticher):
    source = FuturesStitcher_source
    latest_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-1]
    last_input_df = source[latest_contract]
    assert last_input_df.iloc[-1].close == add_sticher.data.iloc[-1].close


def test_FuturesStitcher_last_but_one_df_adjusted(FuturesStitcher_source, add_sticher):
    source = FuturesStitcher_source
    last_but_one_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-2]
    input_df = source[last_but_one_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = add_sticher.data.loc[datapoint_index]
    offset = add_sticher.inspect()["offset"].iloc[-1]
    assert datapoint_input["close"] == datapoint_output["close"] - offset


def test_FuturesStitcher_third_from_last_df_adjusted(
    FuturesStitcher_source, add_sticher
):
    source = FuturesStitcher_source
    third_from_last_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-3]
    input_df = source[third_from_last_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = add_sticher.data.loc[datapoint_index]
    offsets = add_sticher.inspect()["offset"]
    assert (
        datapoint_input["close"]
        == datapoint_output["close"] - offsets.iloc[-1] - offsets.iloc[-2]
    )


def test_FuturesStitcher_first_df_cummulative_adjustment_correct(
    FuturesStitcher_source, add_sticher
):
    source = FuturesStitcher_source
    # source is: dict[ibi.Contract, pd.DataFrame]
    first_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[0]
    first_input_df = source[first_contract]
    output_df = add_sticher.data
    datapoint_index = output_df.index[0]

    input_datapoint = first_input_df.loc[datapoint_index]
    output_datapoint = output_df.loc[datapoint_index]

    cummulative_adjustment = add_sticher.inspect()["offset"].sum()
    assert (
        output_datapoint["close"] == input_datapoint["close"] + cummulative_adjustment
    )


def test_FuturesStitcher_last_but_one_df_adjusted_mul(
    FuturesStitcher_source, mul_sticher
):
    source = FuturesStitcher_source
    last_but_one_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-2]
    input_df = source[last_but_one_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = mul_sticher.data.loc[datapoint_index]
    offset = mul_sticher.inspect()["offset"].iloc[-1]
    assert datapoint_input["close"] * offset == datapoint_output["close"]


def test_FuturesStitcher_third_from_last_df_adjusted_mul(
    FuturesStitcher_source, mul_sticher
):
    source = FuturesStitcher_source
    third_from_last_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-3]
    input_df = source[third_from_last_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = mul_sticher.data.loc[datapoint_index]
    offsets = mul_sticher.inspect()["offset"]
    assert datapoint_output["close"] == pytest.approx(
        datapoint_input["close"] * offsets.iloc[-1] * offsets.iloc[-2]
    )


def test_FuturesStitcher_first_df_cummulative_adjustment_correct_mul(
    FuturesStitcher_source, mul_sticher
):
    source = FuturesStitcher_source
    # source is: dict[ibi.Contract, pd.DataFrame]
    first_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[0]
    first_input_df = source[first_contract]
    output_df = mul_sticher.data
    datapoint_index = output_df.index[0]

    input_datapoint = first_input_df.loc[datapoint_index]
    output_datapoint = output_df.loc[datapoint_index]

    cummulative_adjustment = mul_sticher.inspect()["offset"].product()
    assert output_datapoint["close"] == pytest.approx(
        input_datapoint["close"] * cummulative_adjustment
    )


def test_FuturesStitcher_last_but_one_df_adjusted_none(
    FuturesStitcher_source, none_sticher
):
    source = FuturesStitcher_source
    last_but_one_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-2]
    input_df = source[last_but_one_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = none_sticher.data.loc[datapoint_index]
    assert datapoint_input["close"] == datapoint_output["close"]


def test_FuturesStitcher_third_from_last_df_adjusted_none(
    FuturesStitcher_source, none_sticher
):
    source = FuturesStitcher_source
    third_from_last_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-3]
    input_df = source[third_from_last_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = none_sticher.data.loc[datapoint_index]

    assert datapoint_output["close"] == datapoint_input["close"]


def test_FuturesStitcher_first_df_cummulative_adjustment_correct_none(
    FuturesStitcher_source, none_sticher
):
    source = FuturesStitcher_source
    # source is: dict[ibi.Contract, pd.DataFrame]
    first_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[0]
    first_input_df = source[first_contract]
    output_df = none_sticher.data
    datapoint_index = output_df.index[0]

    input_datapoint = first_input_df.loc[datapoint_index]
    output_datapoint = output_df.loc[datapoint_index]

    assert output_datapoint["close"] == input_datapoint["close"]


def test_resulting_df_monotonic(add_sticher):
    assert add_sticher.data.index.is_monotonic_increasing


def test_resulting_df_no_index_duplicates(add_sticher):
    df = add_sticher.data
    print(df[df.duplicated()])
    assert df[df.index.duplicated()].empty


def test_resulting_df_bounds_correct(add_sticher):
    """
    Test that the resulting df covers all the relevant period as per
    supplied dfs and contract ranges.
    """
    first_output_point = add_sticher.data.index[0]
    last_output_point = add_sticher.data.index[-1]

    ranges = add_sticher._selector.date_ranges
    contracts = list(ranges)
    first = contracts[0]
    last = contracts[-1]

    start_date = ranges[first][0].replace(tzinfo=add_sticher.tz_info)
    end_date = ranges[last][1].replace(tzinfo=add_sticher.tz_info)

    first_input_point = (
        add_sticher.source[first]
        .loc[start_date - add_sticher._buffer_offset :]
        .index[0]
    )
    last_input_point = add_sticher.source[last].loc[:end_date].index[-1]

    assert first_input_point == first_output_point
    assert last_input_point == last_output_point


# ======================================
# LLM generated below
# ======================================


def test_make_roll_timestamp_intraday_tz_aware(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-12 09:00", periods=3, freq="1h", tz="US/Eastern"),
    )
    ts = sticher_for_sync._make_roll_timestamp(roll_day, df)
    assert ts == pd.Timestamp("2024-03-11 10:00", tz="US/Eastern")
    assert ts.hour == sticher_for_sync.roll_hour


def test_make_roll_timestamp_intraday_tz_naive(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-12 09:00", periods=3, freq="1h"),
    )
    ts = sticher_for_sync._make_roll_timestamp(roll_day, df)
    assert ts == pd.Timestamp("2024-03-11 10:00")
    assert ts.tzinfo is None
    assert ts.hour == sticher_for_sync.roll_hour


def test_make_roll_timestamp_daily_no_hour_forced(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-10", periods=3, freq="B"),
    )
    ts = sticher_for_sync._make_roll_timestamp(roll_day, df)
    assert ts.hour == 0


def test_make_roll_timestamp_daily_tz_aware_no_hour_forced(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-10", periods=3, freq="B", tz="US/Eastern"),
    )
    ts = sticher_for_sync._make_roll_timestamp(roll_day, df)
    assert ts.hour == 0
    assert ts.tzinfo is not None


def test_make_roll_timestamp_daily_tz_naive_no_hour_forced(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-10", periods=3, freq="B"),
    )
    ts = sticher_for_sync._make_roll_timestamp(roll_day, df)
    assert ts.hour == 0
    assert ts.tzinfo is None


def test_make_roll_timestamp_respects_custom_roll_hour():
    sticher = FuturesStitcher({Mock(): Mock()}, roll_hour=14)
    roll_day = datetime(2024, 3, 12)
    df = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-12 09:00", periods=3, freq="1h", tz="US/Eastern"),
    )
    ts = sticher._make_roll_timestamp(roll_day, df)
    assert ts.hour == 14


@pytest.fixture
def make_intraday_df():
    def _make(tz=None, start="2024-03-12 09:00", periods=48, freq="1h"):
        index = pd.date_range(start, periods=periods, freq=freq, tz=tz)
        return pd.DataFrame({"close": range(periods)}, index=index)

    return _make


@pytest.fixture
def sticher_for_sync(FuturesStitcher_source):
    return FuturesStitcher(FuturesStitcher_source, "add")


def test_sync_point_exact_match_tz_aware(make_intraday_df, sticher_for_sync):
    roll_day = datetime(2024, 3, 13)
    # use US/Eastern directly so roll_ts matches index exactly
    df0 = make_intraday_df(
        tz="US/Eastern", start="2024-03-12 10:00", periods=48, freq="1h"
    )
    df1 = make_intraday_df(
        tz="US/Eastern", start="2024-03-12 10:00", periods=48, freq="1h"
    )
    sync = sticher_for_sync._sync_point(df0, df1, roll_day)
    assert sync == pd.Timestamp("2024-03-12 10:00", tz="US/Eastern")


def test_sync_point_exact_match_tz_naive(make_intraday_df, sticher_for_sync):
    roll_day = datetime(2024, 3, 13)
    df0 = make_intraday_df(tz=None, start="2024-03-12 09:00", periods=24, freq="1h")
    df1 = make_intraday_df(tz=None, start="2024-03-12 09:00", periods=24, freq="1h")
    sync = sticher_for_sync._sync_point(df0, df1, roll_day)
    assert sync == pd.Timestamp("2024-03-12 10:00")


def test_sync_point_falls_back_to_common_index(make_intraday_df, sticher_for_sync):
    roll_day = datetime(2024, 3, 13)
    # df0 ends before roll_ts, df1 starts after — no exact match
    df0 = make_intraday_df(tz="UTC", start="2024-03-12 13:00", periods=3, freq="1h")
    df1 = make_intraday_df(tz="UTC", start="2024-03-12 14:00", periods=5, freq="1h")
    sync = sticher_for_sync._sync_point(df0, df1, roll_day)
    assert sync in df0.index
    assert sync in df1.index


def test_sync_point_non_overlapping_raises(make_intraday_df, sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df0 = make_intraday_df(tz="UTC", start="2024-03-10 09:00", periods=3, freq="1h")
    df1 = make_intraday_df(tz="UTC", start="2024-03-13 09:00", periods=3, freq="1h")
    with pytest.raises(NonOverLappingDfsError):
        sticher_for_sync._sync_point(df0, df1, roll_day)


def test_sync_point_daily_data_no_hour_forced(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    index = pd.date_range("2024-03-10", periods=5, freq="B", tz="UTC")
    df0 = pd.DataFrame({"close": range(5)}, index=index)
    df1 = pd.DataFrame({"close": range(5)}, index=index)
    sync = sticher_for_sync._sync_point(df0, df1, roll_day)
    # for daily data roll_ts should not have hour forced to roll_hour
    assert sync.hour == 0


def test_sync_point_present_in_df0_but_not_df1_falls_back(sticher_for_sync):
    roll_day = datetime(2024, 3, 13)
    # df0 has 15:00 UTC (roll_ts), df1 skips it but shares 14:00 and 14:30
    df0 = pd.DataFrame(
        {"close": range(4)},
        index=pd.DatetimeIndex(
            [
                "2024-03-12 14:00",
                "2024-03-12 14:30",
                "2024-03-12 15:00",
                "2024-03-12 16:00",
            ],
            tz="UTC",
        ),
    )
    df1 = pd.DataFrame(
        {"close": range(4)},
        index=pd.DatetimeIndex(
            [
                "2024-03-12 14:00",
                "2024-03-12 14:30",
                "2024-03-12 16:00",
                "2024-03-12 17:00",
            ],
            tz="UTC",
        ),
    )
    sync = sticher_for_sync._sync_point(df0, df1, roll_day)
    assert sync == pd.Timestamp("2024-03-12 10:00", tz="US/Eastern")
    assert sync in df0.index
    assert sync in df1.index


def test_sync_point_tz_naive_no_overlap_raises(sticher_for_sync):
    roll_day = datetime(2024, 3, 12)
    df0 = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-10 09:00", periods=3, freq="1h"),
    )
    df1 = pd.DataFrame(
        {"close": range(3)},
        index=pd.date_range("2024-03-13 09:00", periods=3, freq="1h"),
    )
    with pytest.raises(NonOverLappingDfsError):
        sticher_for_sync._sync_point(df0, df1, roll_day)


def test_offset_None_returns_zero():
    sticher = FuturesStitcher({Mock(): Mock()}, None)
    assert sticher.offset(5, 7) == 0


def test_inspect_returns_none_without_debug(FuturesStitcher_source):
    sticher = FuturesStitcher(FuturesStitcher_source, debug=False)
    assert sticher.inspect() is None


def test_inspect_returns_dataframe_with_debug(add_sticher):
    result = add_sticher.inspect()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_inspect_columns(add_sticher):
    result = add_sticher.inspect()
    expected_columns = {
        "roll_day",
        "sync_point",
        "old_contract",
        "new_contract",
        "old_end",
        "new_beginning",
        "old_close",
        "new_close",
        "offset",
    }
    assert expected_columns.issubset(set(result.columns))


def test_inspect_sorted_by_sync_point(add_sticher):
    result = add_sticher.inspect()
    assert result["sync_point"].is_monotonic_increasing


def test_inspect_row_count(FuturesStitcher_source, add_sticher):
    # one row per roll, so n_contracts - 1
    assert len(add_sticher.inspect()) == len(FuturesStitcher_source) - 1


def test_first_df_starts_at_start_date(FuturesStitcher_source, add_sticher):
    ranges = add_sticher._selector.date_ranges
    first_contract = list(ranges)[0]
    start_date = ranges[first_contract][0].replace(tzinfo=add_sticher.tz_info)
    _, first_df, _ = add_sticher._dfs[0]
    assert first_df.index[0] >= start_date - add_sticher._buffer_offset


def test_explicit_selector_used(FuturesStitcher_source):
    selector = FutureSelector.from_contracts(list(FuturesStitcher_source.keys()))
    sticher = FuturesStitcher(FuturesStitcher_source, selector=selector)
    assert sticher._selector is selector


def test_resulting_df_monotonic_mul(mul_sticher):
    assert mul_sticher.data.index.is_monotonic_increasing


def test_resulting_df_no_duplicates_mul(mul_sticher):
    assert mul_sticher.data[mul_sticher.data.index.duplicated()].empty


def test_resulting_df_monotonic_none(none_sticher):
    assert none_sticher.data.index.is_monotonic_increasing


def test_resulting_df_no_duplicates_none(none_sticher):
    assert none_sticher.data[none_sticher.data.index.duplicated()].empty
