import math
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.sticher import FuturesSticher


def test_offset():
    sticher = FuturesSticher({Mock(): Mock()})
    assert sticher.offset(5, 7) == 2


def test_offset_negative():
    sticher = FuturesSticher({Mock(): Mock()})
    assert sticher.offset(5, 4) == -1


def test_offset_mul():
    sticher = FuturesSticher({Mock(): Mock()}, "mul")
    assert sticher.offset(0.5, 1) == 2


def test_offset_None():
    sticher = FuturesSticher({Mock(): Mock()}, None)
    assert sticher.offset(5, 7) == 0


def test_sticher_doesnt_accept_wrong_adjust_type():
    with pytest.raises(AssertionError):
        FuturesSticher([(Mock(), Mock())], "xxx")


def test_adjust():
    sticher = FuturesSticher({Mock(): Mock()})
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
    sticher = FuturesSticher({Mock(): Mock()})
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
    sticher = FuturesSticher({Mock(): Mock()}, "mul")
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
    sticher = FuturesSticher({Mock(): Mock()}, "mul")
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
    sticher = FuturesSticher({Mock(): Mock()}, None)
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


def test_params_passed_from_FuturesSticher_to_FutureSelector_both_not_None():
    sticher = FuturesSticher({Mock(): Mock()}, roll_bdays=666, roll_margin_bdays=999)
    assert sticher._selector.roll_bdays == 666
    assert sticher._selector.roll_margin_bdays == 999


def test_params_passed_from_FuturesSticher_to_FutureSelector_one_not_None():
    sticher = FuturesSticher({Mock(): Mock()}, roll_bdays=666)
    assert sticher._selector.roll_bdays == 666
    assert sticher._selector.roll_margin_bdays is not None


def test_params_passed_from_FuturesSticher_to_FutureSelector_other_one_not_None():
    sticher = FuturesSticher({Mock(): Mock()}, roll_margin_bdays=666)
    assert sticher._selector.roll_margin_bdays == 666
    assert sticher._selector.roll_bdays is not None


def test_params_passed_from_FuturesSticher_to_FutureSelector_both_None():
    sticher = FuturesSticher({Mock(): Mock()}, roll_margin_bdays=666)
    assert sticher._selector.roll_bdays is not None
    assert sticher._selector.roll_margin_bdays is not None


@pytest.fixture(scope="module")
def FuturesSticher_source() -> dict[ibi.Future, pd.DataFrame]:
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
def add_sticher(FuturesSticher_source):
    return FuturesSticher(FuturesSticher_source, "add")


@pytest.fixture(scope="module")
def mul_sticher(FuturesSticher_source):
    return FuturesSticher(FuturesSticher_source, "mul")


def test_FuturesSticher_all_dfs_used(FuturesSticher_source, add_sticher):
    source = FuturesSticher_source
    dfs = add_sticher._dfs
    assert len(source) == len(dfs)


def test_FuturesSticher_last_df_not_adjusted(FuturesSticher_source, add_sticher):
    source = FuturesSticher_source
    latest_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-1]
    last_input_df = source[latest_contract]
    assert last_input_df.iloc[-1].close == add_sticher.data.iloc[-1].close


def test_FuturesSticher_last_but_one_df_adjusted(FuturesSticher_source, add_sticher):
    source = FuturesSticher_source
    last_but_one_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-2]
    input_df = source[last_but_one_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = add_sticher.data.loc[datapoint_index]
    assert (
        datapoint_input["close"] == datapoint_output["close"] - add_sticher._offsets[-1]
    )


def test_FuturesSticher_third_from_last_df_adjusted(FuturesSticher_source, add_sticher):
    source = FuturesSticher_source
    third_from_last_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-3]
    input_df = source[third_from_last_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = add_sticher.data.loc[datapoint_index]
    assert (
        datapoint_input["close"]
        == datapoint_output["close"]
        - add_sticher._offsets[-1]
        - add_sticher._offsets[-2]
    )


def test_FuturesSticher_correct_offset_calculated(add_sticher):
    last_df = add_sticher._dfs[-1]
    previous_df = add_sticher._dfs[-2]

    sync_index = previous_df.index[-1]

    offset = last_df.close.loc[sync_index] - previous_df.close.loc[sync_index]

    test_df = add_sticher.data.loc[:sync_index]

    assert test_df["close"].iloc[-5] == previous_df["close"].iloc[-5] + offset


def test_FuturesSticher_first_df_cummulative_adjustment_correct(
    FuturesSticher_source, add_sticher
):
    source = FuturesSticher_source
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

    cummulative_adjustment = sum(add_sticher._offsets)
    assert (
        output_datapoint["close"] == input_datapoint["close"] + cummulative_adjustment
    )


def test_FuturesSticher_last_but_one_df_adjusted_mul(
    FuturesSticher_source, mul_sticher
):
    source = FuturesSticher_source
    last_but_one_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-2]
    input_df = source[last_but_one_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = mul_sticher.data.loc[datapoint_index]
    assert (
        datapoint_input["close"] * mul_sticher._offsets[-1] == datapoint_output["close"]
    )


def test_FuturesSticher_third_from_last_df_adjusted_mul(
    FuturesSticher_source, mul_sticher
):
    source = FuturesSticher_source
    third_from_last_contract = sorted(
        list(source),
        key=lambda x: datetime.strptime(x.lastTradeDateOrContractMonth, "%Y%m%d"),
    )[-3]
    input_df = source[third_from_last_contract]
    mid_point = int(len(input_df.index) / 2)
    datapoint_index = input_df.index[mid_point]
    datapoint_input = input_df.loc[datapoint_index]
    datapoint_output = mul_sticher.data.loc[datapoint_index]
    assert (
        datapoint_output["close"]
        == datapoint_input["close"]
        * mul_sticher._offsets[-1]
        * mul_sticher._offsets[-2]
    )


def test_FuturesSticher_correct_offset_calculated_mul(mul_sticher):
    last_df = mul_sticher._dfs[-1]
    previous_df = mul_sticher._dfs[-2]

    sync_index = previous_df.index[-1]

    offset = last_df.close.loc[sync_index] / previous_df.close.loc[sync_index]

    test_df = mul_sticher.data.loc[:sync_index]

    assert test_df["close"].iloc[-5] == previous_df["close"].iloc[-5] * offset


def test_FuturesSticher_first_df_cummulative_adjustment_correct_mul(
    FuturesSticher_source, mul_sticher
):
    source = FuturesSticher_source
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

    cummulative_adjustment = math.prod(mul_sticher._offsets)
    assert output_datapoint["close"] == pytest.approx(
        input_datapoint["close"] * cummulative_adjustment
    )


def test_resulting_df_monotonic(add_sticher):
    assert add_sticher.data.index.is_monotonic_increasing


def test_resulting_df_no_index_duplicates(add_sticher):
    df = add_sticher.data
    print(df[df.duplicated()])
    assert df[df.index.duplicated()].empty


def test_resulting_df_bounds_correct(add_sticher):
    first_output_point = add_sticher.data.index[0]
    last_output_point = add_sticher.data.index[-1]

    first_input_point = add_sticher._dfs[0].index[0]
    last_input_point = add_sticher._dfs[-1].index[-1]

    assert first_input_point == first_output_point
    assert last_input_point == last_output_point


def test_adjustment_correct(add_sticher):
    # last but one df close point
    unadjusted_index = add_sticher._dfs[-2].index[-1]
    unadjusted_close = add_sticher._dfs[-2].iloc[-1].close

    adjusted_close = add_sticher.data.loc[unadjusted_index].close

    assert adjusted_close - unadjusted_close == add_sticher._offsets[-1]
