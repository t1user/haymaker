import logging
import pickle
from pathlib import Path
from unittest.mock import Mock

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.base import Atom
from haymaker.sticher import FuturesSticher, Sticher


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
    for i in range(1, 6):
        p = Path(test_dir / "data" / f"contract_df_dict_part_{i}.pickle")
        with p.open("rb") as f:
            contract_df_dict.update(pickle.load(f))
    return contract_df_dict


@pytest.fixture(scope="module")
def one_sticher(FuturesSticher_source):
    return FuturesSticher(FuturesSticher_source)


def test_FuturesSticher_last_df_not_adjusted(one_sticher):
    assert one_sticher._dfs[-1].close[-1] == one_sticher.data.iloc[-1].close


def test_FuturesSticher_first_df_cummulative_adjustment_correct(one_sticher):
    first_input_df = one_sticher._dfs[0]
    output_df = one_sticher.data
    cummulative_adjustment = sum(one_sticher._offsets)
    assert (
        output_df.iloc[0].close - first_input_df.iloc[0].close == cummulative_adjustment
    )


def test_resulting_df_monotonic(one_sticher):
    assert one_sticher.data.index.is_monotonic_increasing


def test_resulting_df_no_index_duplicates(one_sticher):
    df = one_sticher.data
    print(df[df.duplicated()])
    assert df[df.index.duplicated()].empty


def test_resulting_df_bounds_correct(one_sticher):
    first_output_point = one_sticher.data.index[0]
    last_output_point = one_sticher.data.index[-1]

    first_input_point = one_sticher._dfs[0].index[0]
    last_input_point = one_sticher._dfs[-1].index[-1]

    assert first_input_point == first_output_point
    assert last_input_point == last_output_point


def test_all_data_points_used(one_sticher):
    input_data_points = sum([len(df) for df in one_sticher._dfs])
    output_data_points = len(one_sticher.data)
    # number of offsets = number of joints, which drop duplicate point
    assert input_data_points - len(one_sticher._offsets) == output_data_points


def test_date_range_for_every_contract(one_sticher):
    assert len(one_sticher.source) == len(one_sticher._date_ranges)


def test_adjustment_correct(one_sticher):
    # last but one df close point
    unadjusted_index = one_sticher._dfs[-2].index[-1]
    unadjusted_close = one_sticher._dfs[-2].iloc[-1].close

    adjusted_close = one_sticher.data.loc[unadjusted_index].close

    assert adjusted_close - unadjusted_close == one_sticher._offsets[-1]


# ########################################
# Sticher from here on
# ########################################


def test_Sticher_cannot_set_contract_in_int():
    """Make sure user is not able to manually set contract."""
    with pytest.raises(TypeError):
        Sticher(saver=Mock(), contract=ibi.Contract(symbol="empty contract"))


def test_new_Sticher_has_no_contract():
    """Contract on Sticher needs to be set in `onStart` so that it's
    alligned with Streamer it's connected to."""
    sticher = Sticher(saver=Mock())
    assert sticher.contract is None


def test_Sticher_raises_if_contract_not_passed_onStart(caplog):
    sticher = Sticher(saver=Mock())

    class Source(Atom):
        pass

    source = Source()
    source += sticher

    # emit without contract
    source.startEvent.emit({})

    # Eventkit captures errors and logs then instead of raising
    assert any(
        "received no contract onStart" in record.message for record in caplog.records
    )
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_Sticher_warns_when_contract_reset(caplog):
    caplog.set_level(logging.DEBUG)
    sticher = Sticher(saver=Mock())
    sticher.contract = ibi.Contract()

    class Source(Atom):
        pass

    source = Source()
    source += sticher

    source.startEvent.emit({"contract": ibi.Contract(conId=666)})

    # a log was generated
    print(caplog.records)
    assert len(caplog.records) > 0
